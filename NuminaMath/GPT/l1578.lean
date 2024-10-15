import Mathlib

namespace NUMINAMATH_GPT_triangle_area_ratio_l1578_157866

open Set 

variables {X Y Z W : Type} 
variable [LinearOrder X]

noncomputable def ratio_areas (XW WZ : ℕ) (h : ℕ) : ℚ :=
  (8 * h : ℚ) / (12 * h)

theorem triangle_area_ratio (XW WZ : ℕ) (h : ℕ)
  (hXW : XW = 8)
  (hWZ : WZ = 12) :
  ratio_areas XW WZ h = 2 / 3 :=
by
  rw [hXW, hWZ]
  unfold ratio_areas
  norm_num
  sorry

end NUMINAMATH_GPT_triangle_area_ratio_l1578_157866


namespace NUMINAMATH_GPT_equation_of_parametrized_curve_l1578_157819

theorem equation_of_parametrized_curve :
  ∀ t : ℝ, let x := 3 * t + 6 
           let y := 5 * t - 8 
           ∃ (m b : ℝ), y = m * x + b ∧ m = 5 / 3 ∧ b = -18 :=
by
  sorry

end NUMINAMATH_GPT_equation_of_parametrized_curve_l1578_157819


namespace NUMINAMATH_GPT_fans_who_received_all_three_l1578_157843

theorem fans_who_received_all_three (n : ℕ) :
  (∀ k, k ≤ 5000 → (k % 100 = 0 → k % 40 = 0 → k % 60 = 0 → k / 600 ≤ n)) ∧
  (∀ k, k ≤ 5000 → (k % 100 = 0 → k % 40 = 0 → k % 60 = 0 → k / 600 ≤ 8)) :=
by
  sorry

end NUMINAMATH_GPT_fans_who_received_all_three_l1578_157843


namespace NUMINAMATH_GPT_age_of_25th_student_l1578_157810

variable (total_students : ℕ) (total_average : ℕ)
variable (group1_students : ℕ) (group1_average : ℕ)
variable (group2_students : ℕ) (group2_average : ℕ)

theorem age_of_25th_student 
  (h1 : total_students = 25) 
  (h2 : total_average = 25)
  (h3 : group1_students = 10)
  (h4 : group1_average = 22)
  (h5 : group2_students = 14)
  (h6 : group2_average = 28) : 
  (total_students * total_average) =
  (group1_students * group1_average) + (group2_students * group2_average) + 13 :=
by sorry

end NUMINAMATH_GPT_age_of_25th_student_l1578_157810


namespace NUMINAMATH_GPT_gcd_g_x_l1578_157879

def g (x : ℕ) : ℕ := (5 * x + 7) * (11 * x + 3) * (17 * x + 8) * (4 * x + 5)

theorem gcd_g_x (x : ℕ) (hx : 17280 ∣ x) : Nat.gcd (g x) x = 120 :=
by sorry

end NUMINAMATH_GPT_gcd_g_x_l1578_157879


namespace NUMINAMATH_GPT_hours_between_dates_not_thirteen_l1578_157875

def total_hours (start_date: ℕ × ℕ × ℕ × ℕ) (end_date: ℕ × ℕ × ℕ × ℕ) (days_in_dec: ℕ) : ℕ :=
  let (start_year, start_month, start_day, start_hour) := start_date
  let (end_year, end_month, end_day, end_hour) := end_date
  (days_in_dec - start_day) * 24 - start_hour + end_day * 24 + end_hour

theorem hours_between_dates_not_thirteen :
  let start_date := (2015, 12, 30, 23)
  let end_date := (2016, 1, 1, 12)
  let days_in_dec := 31
  total_hours start_date end_date days_in_dec ≠ 13 :=
by
  sorry

end NUMINAMATH_GPT_hours_between_dates_not_thirteen_l1578_157875


namespace NUMINAMATH_GPT_mb_less_than_neg_one_point_five_l1578_157815

theorem mb_less_than_neg_one_point_five (m b : ℚ) (h1 : m = 3/4) (h2 : b = -2) : m * b < -1.5 :=
by {
  -- sorry skips the proof
  sorry
}

end NUMINAMATH_GPT_mb_less_than_neg_one_point_five_l1578_157815


namespace NUMINAMATH_GPT_purely_imaginary_value_of_m_third_quadrant_value_of_m_l1578_157817

theorem purely_imaginary_value_of_m (m : ℝ) :
  (2 * m^2 - 3 * m - 2 = 0) ∧ (m^2 - 2 * m ≠ 0) → m = -1/2 :=
by
  sorry

theorem third_quadrant_value_of_m (m : ℝ) :
  (2 * m^2 - 3 * m - 2 < 0) ∧ (m^2 - 2 * m < 0) → 0 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_value_of_m_third_quadrant_value_of_m_l1578_157817


namespace NUMINAMATH_GPT_symmetric_line_condition_l1578_157808

theorem symmetric_line_condition (x y : ℝ) :
  (∀ x y : ℝ, x - 2 * y - 3 = 0 → -y + 2 * x - 3 = 0) →
  (∀ x y : ℝ, x + y = 0 → ∃ a b c : ℝ, 2 * x - y - 3 = 0) :=
sorry

end NUMINAMATH_GPT_symmetric_line_condition_l1578_157808


namespace NUMINAMATH_GPT_maximum_contribution_l1578_157840

theorem maximum_contribution (total_contribution : ℕ) (num_people : ℕ) (individual_min_contribution : ℕ) :
  total_contribution = 20 → num_people = 10 → individual_min_contribution = 1 → 
  ∃ (max_contribution : ℕ), max_contribution = 11 := by
  intro h1 h2 h3
  existsi 11
  sorry

end NUMINAMATH_GPT_maximum_contribution_l1578_157840


namespace NUMINAMATH_GPT_Pat_worked_days_eq_57_l1578_157865

def Pat_earnings (x : ℕ) : ℤ := 100 * x
def Pat_food_costs (x : ℕ) : ℤ := 20 * (70 - x)
def total_balance (x : ℕ) : ℤ := Pat_earnings x - Pat_food_costs x

theorem Pat_worked_days_eq_57 (x : ℕ) (h : total_balance x = 5440) : x = 57 :=
by
  sorry

end NUMINAMATH_GPT_Pat_worked_days_eq_57_l1578_157865


namespace NUMINAMATH_GPT_obtuse_angle_at_515_l1578_157857

-- Definitions derived from conditions
def minuteHandDegrees (minute: ℕ) : ℝ := minute * 6.0
def hourHandDegrees (hour: ℕ) (minute: ℕ) : ℝ := hour * 30.0 + (minute * 0.5)

-- Main statement to be proved
theorem obtuse_angle_at_515 : 
  let hour := 5
  let minute := 15
  let minute_pos := minuteHandDegrees minute
  let hour_pos := hourHandDegrees hour minute
  let angle := abs (minute_pos - hour_pos)
  angle = 67.5 :=
by
  sorry

end NUMINAMATH_GPT_obtuse_angle_at_515_l1578_157857


namespace NUMINAMATH_GPT_even_function_a_is_0_l1578_157855

def f (a : ℝ) (x : ℝ) : ℝ := (a+1) * x^2 + 3 * a * x + 1

theorem even_function_a_is_0 (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 :=
by sorry

end NUMINAMATH_GPT_even_function_a_is_0_l1578_157855


namespace NUMINAMATH_GPT_intersection_of_M_N_equals_0_1_open_interval_l1578_157829

def M : Set ℝ := { x | x ≥ 0 }
def N : Set ℝ := { x | x^2 < 1 }

theorem intersection_of_M_N_equals_0_1_open_interval :
  M ∩ N = { x | 0 ≤ x ∧ x < 1 } := 
sorry

end NUMINAMATH_GPT_intersection_of_M_N_equals_0_1_open_interval_l1578_157829


namespace NUMINAMATH_GPT_cos_F_l1578_157892

theorem cos_F (D E F : ℝ) (hDEF : D + E + F = 180)
  (sin_D : Real.sin D = 4 / 5)
  (cos_E : Real.cos E = 12 / 13) :
  Real.cos F = -16 / 65 :=
  sorry

end NUMINAMATH_GPT_cos_F_l1578_157892


namespace NUMINAMATH_GPT_box_internal_volume_in_cubic_feet_l1578_157813

def box_length := 26 -- inches
def box_width := 26 -- inches
def box_height := 14 -- inches
def wall_thickness := 1 -- inch

def external_volume := box_length * box_width * box_height -- cubic inches
def internal_length := box_length - 2 * wall_thickness
def internal_width := box_width - 2 * wall_thickness
def internal_height := box_height - 2 * wall_thickness
def internal_volume := internal_length * internal_width * internal_height -- cubic inches

def cubic_inches_to_cubic_feet (v : ℕ) : ℕ := v / 1728

theorem box_internal_volume_in_cubic_feet : cubic_inches_to_cubic_feet internal_volume = 4 := by
  sorry

end NUMINAMATH_GPT_box_internal_volume_in_cubic_feet_l1578_157813


namespace NUMINAMATH_GPT_proof_problem_statement_l1578_157825

noncomputable def proof_problem (x y: ℝ) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ (∀ n : ℕ, n > 0 → (⌊x / y⌋ : ℝ) = ⌊↑n * x⌋ / ⌊↑n * y⌋) →
  (x = y ∨ (∃ k : ℤ, k ≠ 0 ∧ (x = k * y ∨ y = k * x)))

-- The formal statement of the problem
theorem proof_problem_statement (x y : ℝ) :
  proof_problem x y := by
  sorry

end NUMINAMATH_GPT_proof_problem_statement_l1578_157825


namespace NUMINAMATH_GPT_angle_bisector_b_c_sum_l1578_157816

theorem angle_bisector_b_c_sum (A B C : ℝ × ℝ)
  (hA : A = (4, -3))
  (hB : B = (-6, 21))
  (hC : C = (10, 7)) :
  ∃ b c : ℝ, (3 * x + b * y + c = 0) ∧ (b + c = correct_answer) :=
by
  sorry

end NUMINAMATH_GPT_angle_bisector_b_c_sum_l1578_157816


namespace NUMINAMATH_GPT_sum_lent_l1578_157844

theorem sum_lent (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ) 
  (h1 : r = 0.06) (h2 : t = 8) (h3 : I = P - 520) : P * r * t = I → P = 1000 :=
by
  -- Given conditions
  intros
  -- Sorry placeholder
  sorry

end NUMINAMATH_GPT_sum_lent_l1578_157844


namespace NUMINAMATH_GPT_domain_of_function_l1578_157893

def domain_sqrt_log : Set ℝ :=
  {x | (2 - x ≥ 0) ∧ ((2 * x - 1) / (3 - x) > 0)}

theorem domain_of_function :
  domain_sqrt_log = {x | (1/2 < x) ∧ (x ≤ 2)} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1578_157893


namespace NUMINAMATH_GPT_jill_earnings_l1578_157898

theorem jill_earnings :
  ∀ (hourly_wage : ℝ) (tip_rate : ℝ) (num_shifts : ℕ) (hours_per_shift : ℕ) (avg_orders_per_hour : ℝ),
  hourly_wage = 4.00 →
  tip_rate = 0.15 →
  num_shifts = 3 →
  hours_per_shift = 8 →
  avg_orders_per_hour = 40 →
  (num_shifts * hours_per_shift * hourly_wage + num_shifts * hours_per_shift * avg_orders_per_hour * tip_rate = 240) :=
by
  intros hourly_wage tip_rate num_shifts hours_per_shift avg_orders_per_hour
  intros hwage_eq trip_rate_eq nshifts_eq hshift_eq avgorder_eq
  sorry

end NUMINAMATH_GPT_jill_earnings_l1578_157898


namespace NUMINAMATH_GPT_birds_left_after_a_week_l1578_157876

def initial_chickens := 300
def initial_turkeys := 200
def initial_guinea_fowls := 80
def daily_chicken_loss := 20
def daily_turkey_loss := 8
def daily_guinea_fowl_loss := 5
def days_in_a_week := 7

def remaining_chickens := initial_chickens - daily_chicken_loss * days_in_a_week
def remaining_turkeys := initial_turkeys - daily_turkey_loss * days_in_a_week
def remaining_guinea_fowls := initial_guinea_fowls - daily_guinea_fowl_loss * days_in_a_week

def total_remaining_birds := remaining_chickens + remaining_turkeys + remaining_guinea_fowls

theorem birds_left_after_a_week : total_remaining_birds = 349 := by
  sorry

end NUMINAMATH_GPT_birds_left_after_a_week_l1578_157876


namespace NUMINAMATH_GPT_total_nails_l1578_157839

def num_planks : Nat := 1
def nails_per_plank : Nat := 3
def additional_nails : Nat := 8

theorem total_nails : (num_planks * nails_per_plank + additional_nails) = 11 :=
by
  sorry

end NUMINAMATH_GPT_total_nails_l1578_157839


namespace NUMINAMATH_GPT_probability_triangle_l1578_157890

noncomputable def points : List (ℕ × ℕ) := [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2), (3, 3)]

def collinear (p1 p2 p3 : (ℕ × ℕ)) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

def is_triangle (p1 p2 p3 : (ℕ × ℕ)) : Prop := ¬ collinear p1 p2 p3

axiom collinear_ACEF : collinear (0, 0) (1, 1) (2, 2) ∧ collinear (0, 0) (1, 1) (3, 3) ∧ collinear (1, 1) (2, 2) (3, 3)
axiom collinear_BCD : collinear (2, 0) (1, 1) (0, 2)

theorem probability_triangle : 
  let total := 20
  let collinear_ACEF := 4
  let collinear_BCD := 1
  (total - collinear_ACEF - collinear_BCD) / total = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_triangle_l1578_157890


namespace NUMINAMATH_GPT_exponent_division_simplification_l1578_157860

theorem exponent_division_simplification :
  ((18^18 / 18^17)^2 * 9^2) / 3^4 = 324 :=
by
  sorry

end NUMINAMATH_GPT_exponent_division_simplification_l1578_157860


namespace NUMINAMATH_GPT_solution_to_functional_equation_l1578_157806

noncomputable def find_functions (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (x / 2) + (x / 2) * (deriv f x)

theorem solution_to_functional_equation :
  ∀ (f : ℝ → ℝ), (∀ x : ℝ, f x = f (x / 2) + (x / 2) * (deriv f x)) ↔ (∃ c b : ℝ, ∀ x : ℝ, f x = c * x + b) :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_to_functional_equation_l1578_157806


namespace NUMINAMATH_GPT_max_chain_triangles_l1578_157835

theorem max_chain_triangles (n : ℕ) (h : n > 0) : 
  ∃ k, k = n^2 - n + 1 := 
sorry

end NUMINAMATH_GPT_max_chain_triangles_l1578_157835


namespace NUMINAMATH_GPT_total_interest_correct_l1578_157834

-- Definitions
def total_amount : ℝ := 3500
def P1 : ℝ := 1550
def P2 : ℝ := total_amount - P1
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

-- Total interest calculation
noncomputable def interest1 : ℝ := P1 * rate1
noncomputable def interest2 : ℝ := P2 * rate2
noncomputable def total_interest : ℝ := interest1 + interest2

-- Theorem statement
theorem total_interest_correct : total_interest = 144 := 
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_total_interest_correct_l1578_157834


namespace NUMINAMATH_GPT_line_hyperbola_unique_intersection_l1578_157877

theorem line_hyperbola_unique_intersection (k : ℝ) :
  (∃ (x y : ℝ), k * x - y - 2 * k = 0 ∧ x^2 - y^2 = 2 ∧ 
  ∀ y₁, y₁ ≠ y → k * x - y₁ - 2 * k ≠ 0 ∧ x^2 - y₁^2 ≠ 2) ↔ (k = 1 ∨ k = -1) :=
by
  sorry

end NUMINAMATH_GPT_line_hyperbola_unique_intersection_l1578_157877


namespace NUMINAMATH_GPT_area_rectangle_l1578_157801

theorem area_rectangle 
    (x y : ℝ)
    (h1 : 5 * x + 4 * y = 10)
    (h2 : 3 * x = 2 * y) :
    5 * (x * y) = 3000 / 121 :=
by
  sorry

end NUMINAMATH_GPT_area_rectangle_l1578_157801


namespace NUMINAMATH_GPT_balls_into_boxes_l1578_157888

theorem balls_into_boxes :
  let balls := 7
  let boxes := 3
  let min_balls_per_box := 1
  let remaining_balls := balls - boxes * min_balls_per_box
  let combination := Nat.choose (remaining_balls + boxes - 1) (boxes - 1) 
  combination = 15 :=
by
  let balls := 7
  let boxes := 3
  let min_balls_per_box := 1
  let remaining_balls := balls - boxes * min_balls_per_box
  let combination := Nat.choose (remaining_balls + boxes - 1) (boxes - 1)
  show combination = 15
  sorry

end NUMINAMATH_GPT_balls_into_boxes_l1578_157888


namespace NUMINAMATH_GPT_combination_exists_l1578_157874

theorem combination_exists 
  (S T Ti : ℝ) (x y z : ℝ)
  (h : 3 * S + 4 * T + 2 * Ti = 40) :
  ∃ x y z : ℝ, x * S + y * T + z * Ti = 60 :=
sorry

end NUMINAMATH_GPT_combination_exists_l1578_157874


namespace NUMINAMATH_GPT_parabola_vertex_l1578_157880

theorem parabola_vertex :
  (∃ x y : ℝ, y^2 + 6 * y + 4 * x - 7 = 0 ∧ (x, y) = (4, -3)) :=
sorry

end NUMINAMATH_GPT_parabola_vertex_l1578_157880


namespace NUMINAMATH_GPT_n_gon_angle_condition_l1578_157867

theorem n_gon_angle_condition (n : ℕ) (h1 : 150 * (n-1) + (30 * n - 210) = 180 * (n-2)) (h2 : 30 * n - 210 < 150) (h3 : 30 * n - 210 > 0) :
  n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11 :=
by
  sorry

end NUMINAMATH_GPT_n_gon_angle_condition_l1578_157867


namespace NUMINAMATH_GPT_find_y_given_conditions_l1578_157862

theorem find_y_given_conditions (x y : ℝ) (h1 : x^(3 * y) = 27) (h2 : x = 3) : y = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_y_given_conditions_l1578_157862


namespace NUMINAMATH_GPT_circular_garden_radius_l1578_157889

theorem circular_garden_radius
  (r : ℝ) -- radius of the circular garden
  (h : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2) :
  r = 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_circular_garden_radius_l1578_157889


namespace NUMINAMATH_GPT_area_of_rectangle_l1578_157849

theorem area_of_rectangle (length : ℝ) (width : ℝ) (h_length : length = 47.3) (h_width : width = 24) : 
  length * width = 1135.2 := 
by 
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l1578_157849


namespace NUMINAMATH_GPT_tan_sum_identity_l1578_157884

theorem tan_sum_identity (theta : Real) (h : Real.tan theta = 1 / 3) :
  Real.tan (theta + Real.pi / 4) = 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_identity_l1578_157884


namespace NUMINAMATH_GPT_determine_x_l1578_157895

theorem determine_x (x : ℚ) : 
  x + 5 / 8 = 2 + 3 / 16 - 2 / 3 → 
  x = 43 / 48 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_x_l1578_157895


namespace NUMINAMATH_GPT_expand_expression_l1578_157856

theorem expand_expression (x y : ℤ) : (x + 7) * (3 * y + 8) = 3 * x * y + 8 * x + 21 * y + 56 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1578_157856


namespace NUMINAMATH_GPT_determine_b_perpendicular_l1578_157872

theorem determine_b_perpendicular :
  ∀ (b : ℝ),
  (b * 2 + (-3) * (-1) + 2 * 4 = 0) → 
  b = -11/2 :=
by
  intros b h
  sorry

end NUMINAMATH_GPT_determine_b_perpendicular_l1578_157872


namespace NUMINAMATH_GPT_set_M_real_l1578_157899

noncomputable def set_M : Set ℂ := {z : ℂ | (z - 1) ^ 2 = Complex.abs (z - 1) ^ 2}

theorem set_M_real :
  set_M = {z : ℂ | ∃ x : ℝ, z = x} :=
by
  sorry

end NUMINAMATH_GPT_set_M_real_l1578_157899


namespace NUMINAMATH_GPT_habitat_limits_are_correct_l1578_157859

-- Definitions of the conditions
def colonyA_doubling_days : ℕ := 22
def colonyB_tripling_days : ℕ := 30
def tripling_interval : ℕ := 2

-- Definitions to confirm they grow as described
def is_colonyA_habitat_limit_reached (days : ℕ) : Prop := days = colonyA_doubling_days
def is_colonyB_habitat_limit_reached (days : ℕ) : Prop := days = colonyB_tripling_days

-- Proof statement
theorem habitat_limits_are_correct :
  (is_colonyA_habitat_limit_reached colonyA_doubling_days) ∧ (is_colonyB_habitat_limit_reached colonyB_tripling_days) :=
by
  sorry

end NUMINAMATH_GPT_habitat_limits_are_correct_l1578_157859


namespace NUMINAMATH_GPT_solve_for_s_l1578_157826

theorem solve_for_s (s : ℝ) (t : ℝ) (h1 : t = 8 * s^2) (h2 : t = 4.8) : s = Real.sqrt 0.6 ∨ s = -Real.sqrt 0.6 := by
  sorry

end NUMINAMATH_GPT_solve_for_s_l1578_157826


namespace NUMINAMATH_GPT_find_other_endpoint_diameter_l1578_157886

-- Define the given conditions
def center : ℝ × ℝ := (1, 2)
def endpoint_A : ℝ × ℝ := (4, 6)

-- Define a function to find the other endpoint
def other_endpoint (center endpoint_A : ℝ × ℝ) : ℝ × ℝ := 
  let vector_CA := (center.1 - endpoint_A.1, center.2 - endpoint_A.2)
  let vector_CB := (-vector_CA.1, -vector_CA.2)
  (center.1 + vector_CB.1, center.2 + vector_CB.2)

-- State the theorem
theorem find_other_endpoint_diameter : 
  ∀ center endpoint_A, other_endpoint center endpoint_A = (4, 6) :=
by
  intro center endpoint_A
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_find_other_endpoint_diameter_l1578_157886


namespace NUMINAMATH_GPT_total_students_in_class_l1578_157803

theorem total_students_in_class (B G : ℕ) (h1 : G = 160) (h2 : 5 * G = 8 * B) : B + G = 260 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_total_students_in_class_l1578_157803


namespace NUMINAMATH_GPT_cuboid_length_l1578_157851

variable (L W H V : ℝ)

theorem cuboid_length (W_eq : W = 4) (H_eq : H = 6) (V_eq : V = 96) (Volume_eq : V = L * W * H) : L = 4 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_length_l1578_157851


namespace NUMINAMATH_GPT_real_roots_exactly_three_l1578_157868

theorem real_roots_exactly_three (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * |x| + 2 = m) → (∃ a b c : ℝ, 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (a^2 - 2 * |a| + 2 = m) ∧ 
  (b^2 - 2 * |b| + 2 = m) ∧ 
  (c^2 - 2 * |c| + 2 = m)) → 
  m = 2 := 
sorry

end NUMINAMATH_GPT_real_roots_exactly_three_l1578_157868


namespace NUMINAMATH_GPT_new_rope_length_l1578_157824

-- Define the given constants and conditions
def rope_length_initial : ℝ := 12
def additional_area : ℝ := 1511.7142857142858
noncomputable def pi_approx : ℝ := Real.pi

-- Define the proof statement
theorem new_rope_length :
  let r2 := Real.sqrt ((additional_area / pi_approx) + rope_length_initial ^ 2)
  r2 = 25 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_new_rope_length_l1578_157824


namespace NUMINAMATH_GPT_natasha_time_reach_top_l1578_157864

variable (t : ℝ) (d_up d_total T : ℝ)

def time_to_reach_top (T d_up d_total t : ℝ) : Prop :=
  d_total = 2 * d_up ∧
  d_up = 1.5 * t ∧
  T = t + 2 ∧
  2 = d_total / T

theorem natasha_time_reach_top (T : ℝ) (h : time_to_reach_top T (1.5 * 4) (3 * 4) 4) : T = 4 :=
by
  sorry

end NUMINAMATH_GPT_natasha_time_reach_top_l1578_157864


namespace NUMINAMATH_GPT_simplify_144_over_1296_times_36_l1578_157833

theorem simplify_144_over_1296_times_36 :
  (144 / 1296) * 36 = 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_144_over_1296_times_36_l1578_157833


namespace NUMINAMATH_GPT_number_of_senior_citizen_tickets_l1578_157887

theorem number_of_senior_citizen_tickets 
    (A S : ℕ)
    (h1 : A + S = 529)
    (h2 : 25 * A + 15 * S = 9745) 
    : S = 348 := 
by
  sorry

end NUMINAMATH_GPT_number_of_senior_citizen_tickets_l1578_157887


namespace NUMINAMATH_GPT_person_picking_number_who_announced_6_is_1_l1578_157836

theorem person_picking_number_who_announced_6_is_1
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ)
  (h₁ : a₁₀ + a₂ = 2)
  (h₂ : a₁ + a₃ = 4)
  (h₃ : a₂ + a₄ = 6)
  (h₄ : a₃ + a₅ = 8)
  (h₅ : a₄ + a₆ = 10)
  (h₆ : a₅ + a₇ = 12)
  (h₇ : a₆ + a₈ = 14)
  (h₈ : a₇ + a₉ = 16)
  (h₉ : a₈ + a₁₀ = 18)
  (h₁₀ : a₉ + a₁ = 20) :
  a₆ = 1 :=
by
  sorry

end NUMINAMATH_GPT_person_picking_number_who_announced_6_is_1_l1578_157836


namespace NUMINAMATH_GPT_correct_transformation_l1578_157845

theorem correct_transformation (x : ℝ) :
  (6 * ((2 * x + 1) / 3) - 6 * ((10 * x + 1) / 6) = 6) ↔ (4 * x + 2 - 10 * x - 1 = 6) :=
by
  sorry

end NUMINAMATH_GPT_correct_transformation_l1578_157845


namespace NUMINAMATH_GPT_birds_per_cup_l1578_157847

theorem birds_per_cup :
  ∀ (C B S T : ℕ) (H1 : C = 2) (H2 : S = 1 / 2 * C) (H3 : T = 21) (H4 : B = 14),
    ((C - S) * B = T) :=
by
  sorry

end NUMINAMATH_GPT_birds_per_cup_l1578_157847


namespace NUMINAMATH_GPT_three_five_seven_sum_fraction_l1578_157897

theorem three_five_seven_sum_fraction :
  (3 * 5 * 7) * ((1 / 3) + (1 / 5) + (1 / 7)) = 71 :=
by
  sorry

end NUMINAMATH_GPT_three_five_seven_sum_fraction_l1578_157897


namespace NUMINAMATH_GPT_min_value_g_geq_6_min_value_g_eq_6_l1578_157814

noncomputable def g (x : ℝ) : ℝ :=
  x + (x / (x^2 + 2)) + (x * (x + 5) / (x^2 + 3)) + (3 * (x + 3) / (x * (x^2 + 3)))

theorem min_value_g_geq_6 : ∀ x > 0, g x ≥ 6 :=
by
  sorry

theorem min_value_g_eq_6 : ∃ x > 0, g x = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_g_geq_6_min_value_g_eq_6_l1578_157814


namespace NUMINAMATH_GPT_candy_bar_price_l1578_157809

theorem candy_bar_price (total_money bread_cost candy_bar_price remaining_money : ℝ) 
    (h1 : total_money = 32)
    (h2 : bread_cost = 3)
    (h3 : remaining_money = 18)
    (h4 : total_money - bread_cost - candy_bar_price - (1 / 3) * (total_money - bread_cost - candy_bar_price) = remaining_money) :
    candy_bar_price = 1.33 := 
sorry

end NUMINAMATH_GPT_candy_bar_price_l1578_157809


namespace NUMINAMATH_GPT_min_value_frac_inv_sum_l1578_157842

theorem min_value_frac_inv_sum (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + 3 * y = 1) : 
  ∃ (minimum_value : ℝ), minimum_value = 4 + 2 * Real.sqrt 3 ∧ (∀ (a b : ℝ), 0 < a → 0 < b → a + 3 * b = 1 → (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3) := 
sorry

end NUMINAMATH_GPT_min_value_frac_inv_sum_l1578_157842


namespace NUMINAMATH_GPT_max_4x_3y_l1578_157871

theorem max_4x_3y (x y : ℝ) (h : x^2 + y^2 = 16 * x + 8 * y + 8) : 4 * x + 3 * y ≤ 63 :=
sorry

end NUMINAMATH_GPT_max_4x_3y_l1578_157871


namespace NUMINAMATH_GPT_equivalent_fraction_l1578_157858

theorem equivalent_fraction : (8 / (5 * 46)) = (0.8 / 23) := 
by sorry

end NUMINAMATH_GPT_equivalent_fraction_l1578_157858


namespace NUMINAMATH_GPT_pickup_carries_10_bags_per_trip_l1578_157802

def total_weight : ℕ := 10000
def weight_one_bag : ℕ := 50
def number_of_trips : ℕ := 20
def total_bags : ℕ := total_weight / weight_one_bag
def bags_per_trip : ℕ := total_bags / number_of_trips

theorem pickup_carries_10_bags_per_trip : bags_per_trip = 10 := by
  sorry

end NUMINAMATH_GPT_pickup_carries_10_bags_per_trip_l1578_157802


namespace NUMINAMATH_GPT_b_joined_after_x_months_l1578_157807

-- Establish the given conditions as hypotheses
theorem b_joined_after_x_months
  (a_start_capital : ℝ)
  (b_start_capital : ℝ)
  (profit_ratio : ℝ)
  (months_in_year : ℝ)
  (a_capital_time : ℝ)
  (b_capital_time : ℝ)
  (a_profit_ratio : ℝ)
  (b_profit_ratio : ℝ)
  (x : ℝ)
  (h1 : a_start_capital = 3500)
  (h2 : b_start_capital = 9000)
  (h3 : profit_ratio = 2 / 3)
  (h4 : months_in_year = 12)
  (h5 : a_capital_time = 12)
  (h6 : b_capital_time = 12 - x)
  (h7 : a_profit_ratio = 2)
  (h8 : b_profit_ratio = 3)
  (h_ratio : (a_start_capital * a_capital_time) / (b_start_capital * b_capital_time) = profit_ratio) :
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_b_joined_after_x_months_l1578_157807


namespace NUMINAMATH_GPT_range_of_k_l1578_157805

theorem range_of_k (k : ℝ) :
  (∀ x : ℤ, ((x^2 - x - 2 > 0) ∧ (2*x^2 + (2*k + 5)*x + 5*k < 0)) ↔ (x = -2)) -> 
  (-3 ≤ k ∧ k < 2) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_k_l1578_157805


namespace NUMINAMATH_GPT_gcd_60_90_l1578_157894

theorem gcd_60_90 : Nat.gcd 60 90 = 30 := by
  let f1 : 60 = 2 ^ 2 * 3 * 5 := by sorry
  let f2 : 90 = 2 * 3 ^ 2 * 5 := by sorry
  sorry

end NUMINAMATH_GPT_gcd_60_90_l1578_157894


namespace NUMINAMATH_GPT_find_K_l1578_157852

def satisfies_conditions (K m n h : ℕ) : Prop :=
  K ∣ (m^h - 1) ∧ K ∣ (n ^ ((m^h - 1) / K) + 1)

def odd (n : ℕ) : Prop := n % 2 = 1

theorem find_K (r : ℕ) (h : ℕ := 2^r) :
    ∀ K : ℕ, (∃ (m : ℕ), odd m ∧ m > 1 ∧ ∃ (n : ℕ), satisfies_conditions K m n h) ↔
    (∃ s t : ℕ, K = 2^(r + s) * t ∧ 2 ∣ t) := sorry

end NUMINAMATH_GPT_find_K_l1578_157852


namespace NUMINAMATH_GPT_gcd_lcm_identity_l1578_157830

theorem gcd_lcm_identity (a b: ℕ) (h_lcm: (Nat.lcm a b) = 4620) (h_gcd: Nat.gcd a b = 33) (h_a: a = 231) : b = 660 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_identity_l1578_157830


namespace NUMINAMATH_GPT_find_smaller_integer_l1578_157811

theorem find_smaller_integer : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ y = x + 8 ∧ x * y = 80 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_integer_l1578_157811


namespace NUMINAMATH_GPT_expression_values_l1578_157831

variable {a b c : ℚ}

theorem expression_values (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) :
  (a / abs a + b / abs b + c / abs c - (a * b * c) / abs (a * b * c) = 2) ∨ 
  (a / abs a + b / abs b + c / abs c - (a * b * c) / abs (a * b * c) = -2) := 
sorry

end NUMINAMATH_GPT_expression_values_l1578_157831


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1578_157861

-- Definitions based on conditions
def hyperbola (x y : ℝ) (a : ℝ) := x^2 / a^2 - y^2 / 5 = 1

-- Main theorem
theorem hyperbola_eccentricity (a : ℝ) (c : ℝ) (h_focus : c = 3) (h_hyperbola : hyperbola 0 0 a) (focus_condition : c^2 = a^2 + 5) :
  c / a = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1578_157861


namespace NUMINAMATH_GPT_same_side_of_line_l1578_157838

open Real

theorem same_side_of_line (a : ℝ) :
  let O := (0, 0)
  let A := (1, 1)
  (O.1 + O.2 < a ↔ A.1 + A.2 < a) →
  a < 0 ∨ a > 2 := by
  sorry

end NUMINAMATH_GPT_same_side_of_line_l1578_157838


namespace NUMINAMATH_GPT_each_partner_percentage_l1578_157882

-- Defining the conditions as variables
variables (total_profit majority_share combined_amount : ℝ) (num_partners : ℕ)

-- Given conditions
def majority_owner_received_25_percent_of_total : total_profit * 0.25 = majority_share := sorry
def remaining_profit_distribution : total_profit - majority_share = 60000 := sorry
def combined_share_of_three : majority_share + 30000 = combined_amount := sorry
def total_profit_amount : total_profit = 80000 := sorry
def number_of_partners : num_partners = 4 := sorry

-- The theorem to be proven
theorem each_partner_percentage :
  ∃ (percent : ℝ), percent = 25 :=
sorry

end NUMINAMATH_GPT_each_partner_percentage_l1578_157882


namespace NUMINAMATH_GPT_basketball_players_taking_chemistry_l1578_157870

variable (total_players : ℕ) (taking_biology : ℕ) (taking_both : ℕ)

theorem basketball_players_taking_chemistry (h1 : total_players = 20) 
                                           (h2 : taking_biology = 8) 
                                           (h3 : taking_both = 4) 
                                           (h4 : ∀p, p ≤ total_players) :
  total_players - taking_biology + taking_both = 16 :=
by sorry

end NUMINAMATH_GPT_basketball_players_taking_chemistry_l1578_157870


namespace NUMINAMATH_GPT_seven_digit_number_insertion_l1578_157883

theorem seven_digit_number_insertion (num : ℕ) (h : num = 52115) : (∃ (count : ℕ), count = 21) :=
by 
  sorry

end NUMINAMATH_GPT_seven_digit_number_insertion_l1578_157883


namespace NUMINAMATH_GPT_largest_divisor_of_n_l1578_157804

theorem largest_divisor_of_n (n : ℕ) (h1 : 0 < n) (h2 : 127 ∣ n^3) : 127 ∣ n :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_n_l1578_157804


namespace NUMINAMATH_GPT_exists_f_m_eq_n_plus_2017_l1578_157818

theorem exists_f_m_eq_n_plus_2017 (m : ℕ) (h : m > 0) :
  (∃ f : ℤ → ℤ, ∀ n : ℤ, (f^[m] n = n + 2017)) ↔ (m = 1 ∨ m = 2017) :=
by
  sorry

end NUMINAMATH_GPT_exists_f_m_eq_n_plus_2017_l1578_157818


namespace NUMINAMATH_GPT_equal_papers_per_cousin_l1578_157837

-- Given conditions
def haley_origami_papers : Float := 48.0
def cousins_count : Float := 6.0

-- Question and expected answer
def papers_per_cousin (total_papers : Float) (cousins : Float) : Float :=
  total_papers / cousins

-- Proof statement asserting the correct answer
theorem equal_papers_per_cousin :
  papers_per_cousin haley_origami_papers cousins_count = 8.0 :=
sorry

end NUMINAMATH_GPT_equal_papers_per_cousin_l1578_157837


namespace NUMINAMATH_GPT_math_problem_l1578_157881

noncomputable def f (x : ℝ) : ℝ := sorry

theorem math_problem (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (1 - x) = f (1 + x))
  (h2 : ∀ x : ℝ, f (-x) = -f x)
  (h3 : ∀ {x y : ℝ}, (0 ≤ x → x < y → y ≤ 1 → f x < f y)) :
  (f 0 = 0) ∧ 
  (∀ x : ℝ, f (x + 2) = f (-x)) ∧ 
  (∀ x : ℝ, x = -1 ∨ ∀ ε > 0, ε ≠ (x + 1))
:= sorry

end NUMINAMATH_GPT_math_problem_l1578_157881


namespace NUMINAMATH_GPT_problem_1_problem_2_l1578_157873

namespace ProofProblems

def U : Set ℝ := {y | true}

def E : Set ℝ := {y | y > 2}

def F : Set ℝ := {y | ∃ (x : ℝ), (-1 < x ∧ x < 2 ∧ y = x^2 - 2*x)}

def complement (A : Set ℝ) : Set ℝ := {y | y ∉ A}

theorem problem_1 : 
  (complement E ∩ F) = {y | -1 ≤ y ∧ y ≤ 2} := 
  sorry

def G (a : ℝ) : Set ℝ := {y | ∃ (x : ℝ), (0 < x ∧ x < a ∧ y = Real.log x / Real.log 2)}

theorem problem_2 (a : ℝ) :
  (∀ y, (y ∈ G a → y < 3)) → a ≥ 8 :=
  sorry

end ProofProblems

end NUMINAMATH_GPT_problem_1_problem_2_l1578_157873


namespace NUMINAMATH_GPT_sum_first_60_terms_l1578_157891

theorem sum_first_60_terms {a : ℕ → ℤ}
  (h : ∀ n, a (n + 1) + (-1)^n * a n = 2 * n - 1) :
  (Finset.range 60).sum a = 1830 :=
sorry

end NUMINAMATH_GPT_sum_first_60_terms_l1578_157891


namespace NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l1578_157848

-- Definitions and conditions
def a : ℤ := -58  -- First term
def d : ℤ := 7   -- Common difference
def l : ℤ := 78  -- Last term

-- Statement of the problem
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 20 := 
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l1578_157848


namespace NUMINAMATH_GPT_smallest_cookie_packages_l1578_157800

/-- The smallest number of cookie packages Zoey can buy in order to buy an equal number of cookie
and milk packages. -/
theorem smallest_cookie_packages (n : ℕ) (h1 : ∃ k : ℕ, 5 * k = 7 * n) : n = 7 :=
sorry

end NUMINAMATH_GPT_smallest_cookie_packages_l1578_157800


namespace NUMINAMATH_GPT_senior_students_in_sample_l1578_157820

theorem senior_students_in_sample 
  (total_students : ℕ) (total_seniors : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : total_seniors = 500)
  (h3 : sample_size = 200) : 
  (total_seniors * sample_size / total_students = 50) :=
by {
  sorry
}

end NUMINAMATH_GPT_senior_students_in_sample_l1578_157820


namespace NUMINAMATH_GPT_problem1_problem2_l1578_157854

theorem problem1 (m : ℝ) (H : m > 0) (p : ∀ x : ℝ, (x+1)*(x-5) ≤ 0 → 1 - m ≤ x ∧ x ≤ 1 + m) : m ≥ 4 :=
sorry

theorem problem2 (x : ℝ) (m : ℝ) (H : m = 5) (disj : ∀ x : ℝ, ((x+1)*(x-5) ≤ 0 ∨ (1 - m ≤ x ∧ x ≤ 1 + m))
) (conj : ¬ ∃ x : ℝ, (x+1)*(x-5) ≤ 0 ∧ (1 - m ≤ x ∧ x ≤ 1 + m)) : (-4 ≤ x ∧ x < -1) ∨ (5 < x ∧ x < 6) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1578_157854


namespace NUMINAMATH_GPT_construction_better_than_logistics_l1578_157869

theorem construction_better_than_logistics 
  (applications_computer : ℕ := 215830)
  (applications_mechanical : ℕ := 200250)
  (applications_marketing : ℕ := 154676)
  (applications_logistics : ℕ := 74570)
  (applications_trade : ℕ := 65280)
  (recruitments_computer : ℕ := 124620)
  (recruitments_marketing : ℕ := 102935)
  (recruitments_mechanical : ℕ := 89115)
  (recruitments_construction : ℕ := 76516)
  (recruitments_chemical : ℕ := 70436) :
  applications_construction / recruitments_construction < applications_logistics / recruitments_logistics→ 
  (applications_computer / recruitments_computer < applications_chemical / recruitments_chemical) :=
sorry

end NUMINAMATH_GPT_construction_better_than_logistics_l1578_157869


namespace NUMINAMATH_GPT_weight_of_hollow_golden_sphere_l1578_157821

theorem weight_of_hollow_golden_sphere : 
  let diameter := 12
  let thickness := 0.3
  let pi := (3 : Real)
  let outer_radius := diameter / 2
  let inner_radius := (outer_radius - thickness)
  let outer_volume := (4 / 3) * pi * outer_radius^3
  let inner_volume := (4 / 3) * pi * inner_radius^3
  let gold_volume := outer_volume - inner_volume
  let weight_per_cubic_inch := 1
  let weight := gold_volume * weight_per_cubic_inch
  weight = 123.23 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_hollow_golden_sphere_l1578_157821


namespace NUMINAMATH_GPT_no_32_people_class_exists_30_people_class_l1578_157846

-- Definition of the conditions: relationship between boys and girls
def friends_condition (B G : ℕ) : Prop :=
  3 * B = 2 * G

-- The first problem statement: No 32 people class
theorem no_32_people_class : ¬ ∃ (B G : ℕ), friends_condition B G ∧ B + G = 32 := 
sorry

-- The second problem statement: There is a 30 people class
theorem exists_30_people_class : ∃ (B G : ℕ), friends_condition B G ∧ B + G = 30 := 
sorry

end NUMINAMATH_GPT_no_32_people_class_exists_30_people_class_l1578_157846


namespace NUMINAMATH_GPT_total_embroidery_time_l1578_157832

-- Defining the constants as given in the problem
def stitches_per_minute : ℕ := 4
def stitches_per_flower : ℕ := 60
def stitches_per_unicorn : ℕ := 180
def stitches_per_godzilla : ℕ := 800
def num_flowers : ℕ := 50
def num_unicorns : ℕ := 3
def num_godzillas : ℕ := 1 -- Implicitly from the problem statement

-- Total time calculation as a Lean theorem
theorem total_embroidery_time : 
  (stitches_per_godzilla * num_godzillas + 
   stitches_per_unicorn * num_unicorns + 
   stitches_per_flower * num_flowers) / stitches_per_minute = 1085 := 
by
  sorry

end NUMINAMATH_GPT_total_embroidery_time_l1578_157832


namespace NUMINAMATH_GPT_inequality_problem_l1578_157853

theorem inequality_problem (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : m + n < 0) : m < -n ∧ -n < n ∧ n < -m :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_problem_l1578_157853


namespace NUMINAMATH_GPT_set_non_neg_even_set_primes_up_to_10_eq_sol_set_l1578_157863

noncomputable def non_neg_even (x : ℕ) : Prop := x % 2 = 0 ∧ x ≤ 10
def primes_up_to_10 (x : ℕ) : Prop := Nat.Prime x ∧ x ≤ 10
def eq_sol (x : ℤ) : Prop := x^2 + 2*x - 15 = 0

theorem set_non_neg_even :
  {x : ℕ | non_neg_even x} = {0, 2, 4, 6, 8, 10} := by
  sorry

theorem set_primes_up_to_10 :
  {x : ℕ | primes_up_to_10 x} = {2, 3, 5, 7} := by
  sorry

theorem eq_sol_set :
  {x : ℤ | eq_sol x} = {-5, 3} := by
  sorry

end NUMINAMATH_GPT_set_non_neg_even_set_primes_up_to_10_eq_sol_set_l1578_157863


namespace NUMINAMATH_GPT_remainder_when_divided_by_6_l1578_157823

theorem remainder_when_divided_by_6 (n : ℕ) (h₁ : n = 482157)
  (odd_n : n % 2 ≠ 0) (div_by_3 : n % 3 = 0) : n % 6 = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_6_l1578_157823


namespace NUMINAMATH_GPT_number_of_friends_with_pears_l1578_157850

-- Each friend either carries pears or oranges
def total_friends : Nat := 15
def friends_with_oranges : Nat := 6
def friends_with_pears : Nat := total_friends - friends_with_oranges

theorem number_of_friends_with_pears :
  friends_with_pears = 9 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_number_of_friends_with_pears_l1578_157850


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l1578_157841

variable (a b : ℝ)

-- Conditions: Points lie on the line y = 2x + 1
def point_M (a : ℝ) : Prop := a = 2 * 2 + 1
def point_N (b : ℝ) : Prop := b = 2 * 3 + 1

-- Prove that a < b given the conditions
theorem relationship_between_a_and_b (hM : point_M a) (hN : point_N b) : a < b := 
sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l1578_157841


namespace NUMINAMATH_GPT_framed_painting_ratio_l1578_157822

def painting_width := 20
def painting_height := 30

def smaller_dimension := painting_width + 2 * 5
def larger_dimension := painting_height + 4 * 5

noncomputable def ratio := (smaller_dimension : ℚ) / (larger_dimension : ℚ)

theorem framed_painting_ratio :
  ratio = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_framed_painting_ratio_l1578_157822


namespace NUMINAMATH_GPT_right_triangle_legs_sum_l1578_157896

-- Definitions
def sum_of_legs (a b : ℕ) : ℕ := a + b

-- Main theorem statement
theorem right_triangle_legs_sum (x : ℕ) (h : x^2 + (x + 1)^2 = 53^2) :
  sum_of_legs x (x + 1) = 75 :=
sorry

end NUMINAMATH_GPT_right_triangle_legs_sum_l1578_157896


namespace NUMINAMATH_GPT_smallest_unwritable_number_l1578_157885

theorem smallest_unwritable_number :
  ∀ a b c d : ℕ, 11 ≠ (2^a - 2^b) / (2^c - 2^d) := sorry

end NUMINAMATH_GPT_smallest_unwritable_number_l1578_157885


namespace NUMINAMATH_GPT_mark_purchased_cans_l1578_157828

theorem mark_purchased_cans : ∀ (J M : ℕ), 
    (J = 40) → 
    (100 - J = 6 * M / 5) → 
    M = 27 := by
  sorry

end NUMINAMATH_GPT_mark_purchased_cans_l1578_157828


namespace NUMINAMATH_GPT_initial_ratio_milk_water_l1578_157878

theorem initial_ratio_milk_water (M W : ℕ) 
  (h1 : M + W = 60) 
  (h2 : ∀ k, k = M → M * 2 = W + 60) : (M:ℚ) / (W:ℚ) = 4 / 1 :=
by
  sorry

end NUMINAMATH_GPT_initial_ratio_milk_water_l1578_157878


namespace NUMINAMATH_GPT_compute_fraction_l1578_157812

noncomputable def distinct_and_sum_zero (w x y z : ℝ) : Prop :=
w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ w + x + y + z = 0

theorem compute_fraction (w x y z : ℝ) (h : distinct_and_sum_zero w x y z) :
  (w * y + x * z) / (w^2 + x^2 + y^2 + z^2) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_compute_fraction_l1578_157812


namespace NUMINAMATH_GPT_solve_quadratic_substitution_l1578_157827

theorem solve_quadratic_substitution : 
  (∀ x : ℝ, (2 * x - 5) ^ 2 - 2 * (2 * x - 5) - 3 = 0 ↔ x = 2 ∨ x = 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_substitution_l1578_157827
