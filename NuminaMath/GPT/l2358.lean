import Mathlib

namespace NUMINAMATH_GPT_parallel_transitivity_l2358_235856

variable (Line Plane : Type)
variable (m n : Line)
variable (α : Plane)

-- Definitions for parallelism
variable (parallel : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Conditions
variable (m_n_parallel : parallel m n)
variable (m_alpha_parallel : parallelLinePlane m α)
variable (n_outside_alpha : ¬ parallelLinePlane n α)

-- Proposition to be proved
theorem parallel_transitivity (m n : Line) (α : Plane) 
  (h1 : parallel m n) 
  (h2 : parallelLinePlane m α) 
  : parallelLinePlane n α :=
sorry 

end NUMINAMATH_GPT_parallel_transitivity_l2358_235856


namespace NUMINAMATH_GPT_equal_roots_of_quadratic_l2358_235872

theorem equal_roots_of_quadratic (k : ℝ) : 
  ( ∀ x : ℝ, 2 * k * x^2 + 7 * k * x + 2 = 0 → x = x ) ↔ k = 16 / 49 :=
by
  sorry

end NUMINAMATH_GPT_equal_roots_of_quadratic_l2358_235872


namespace NUMINAMATH_GPT_min_value_correct_l2358_235801

noncomputable def min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  Real.sqrt ((a^2 + 2 * b^2) * (4 * a^2 + b^2)) / (a * b)

theorem min_value_correct (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_value a b ha hb ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_correct_l2358_235801


namespace NUMINAMATH_GPT_grass_field_width_l2358_235806

theorem grass_field_width (w : ℝ) (length_field : ℝ) (path_width : ℝ) (area_path : ℝ) :
  length_field = 85 → path_width = 2.5 → area_path = 1450 →
  (90 * (w + path_width * 2) - length_field * w = area_path) → w = 200 :=
by
  intros h_length_field h_path_width h_area_path h_eq
  sorry

end NUMINAMATH_GPT_grass_field_width_l2358_235806


namespace NUMINAMATH_GPT_largest_8_11_double_l2358_235888

def is_8_11_double (M : ℕ) : Prop :=
  let digits_8 := (Nat.digits 8 M)
  let M_11 := Nat.ofDigits 11 digits_8
  M_11 = 2 * M

theorem largest_8_11_double : ∃ (M : ℕ), is_8_11_double M ∧ ∀ (N : ℕ), is_8_11_double N → N ≤ M :=
sorry

end NUMINAMATH_GPT_largest_8_11_double_l2358_235888


namespace NUMINAMATH_GPT_alyssa_turnips_l2358_235808

theorem alyssa_turnips (k a t: ℕ) (h1: k = 6) (h2: t = 15) (h3: t = k + a) : a = 9 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_alyssa_turnips_l2358_235808


namespace NUMINAMATH_GPT_simplify_expression_l2358_235869

variable (x y z : ℝ)

-- Statement of the problem to be proved.
theorem simplify_expression :
  (15 * x + 45 * y - 30 * z) + (20 * x - 10 * y + 5 * z) - (5 * x + 35 * y - 15 * z) = 
  (30 * x - 10 * z) :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_simplify_expression_l2358_235869


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_simplify_expr3_l2358_235880

theorem simplify_expr1 : -2.48 + 4.33 + (-7.52) + (-4.33) = -10 := by
  sorry

theorem simplify_expr2 : (7/13) * (-9) + (7/13) * (-18) + (7/13) = -14 := by
  sorry

theorem simplify_expr3 : -((20 + 1/19) * 38) = -762 := by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_simplify_expr3_l2358_235880


namespace NUMINAMATH_GPT_david_on_sixth_platform_l2358_235891

theorem david_on_sixth_platform 
  (h₁ : walter_initial_fall = 4)
  (h₂ : walter_additional_fall = 3 * walter_initial_fall)
  (h₃ : total_fall = walter_initial_fall + walter_additional_fall)
  (h₄ : total_platforms = 8)
  (h₅ : total_height = total_fall)
  (h₆ : platform_height = total_height / total_platforms)
  (h₇ : david_fall_distance = walter_initial_fall)
  : (total_height - david_fall_distance) / platform_height = 6 := 
  by sorry

end NUMINAMATH_GPT_david_on_sixth_platform_l2358_235891


namespace NUMINAMATH_GPT_circle_radius_of_complex_roots_l2358_235854

theorem circle_radius_of_complex_roots (z : ℂ) (hz : (z - 1)^3 = 8 * z^3) : 
  ∃ r : ℝ, r = 1 / Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_of_complex_roots_l2358_235854


namespace NUMINAMATH_GPT_average_age_of_students_l2358_235803

variable (A : ℕ) -- We define A as a natural number representing average age

-- Define the conditions
def num_students : ℕ := 32
def staff_age : ℕ := 49
def new_average_age := A + 1

-- Definition of total age including the staff
def total_age_with_staff := 33 * new_average_age

-- Original condition stated as an equality
def condition : Prop := num_students * A + staff_age = total_age_with_staff

-- Theorem statement asserting that the average age A is 16 given the condition
theorem average_age_of_students : condition A → A = 16 :=
by sorry

end NUMINAMATH_GPT_average_age_of_students_l2358_235803


namespace NUMINAMATH_GPT_number_of_clients_l2358_235831

-- Definitions from the problem
def cars : ℕ := 18
def selections_per_client : ℕ := 3
def selections_per_car : ℕ := 3

-- Theorem statement: Prove that the number of clients is 18
theorem number_of_clients (total_cars : ℕ) (cars_selected_by_each_client : ℕ) (each_car_selected : ℕ)
  (h_cars : total_cars = cars)
  (h_select_each : cars_selected_by_each_client = selections_per_client)
  (h_selected_car : each_car_selected = selections_per_car) :
  (total_cars * each_car_selected) / cars_selected_by_each_client = 18 :=
by
  rw [h_cars, h_select_each, h_selected_car]
  sorry

end NUMINAMATH_GPT_number_of_clients_l2358_235831


namespace NUMINAMATH_GPT_number_of_dots_on_A_number_of_dots_on_B_number_of_dots_on_C_number_of_dots_on_D_l2358_235874

variable (A B C D : ℕ)
variable (dice : ℕ → ℕ)

-- Conditions
axiom dice_faces : ∀ {i : ℕ}, 1 ≤ i ∧ i ≤ 6 → ∃ j, dice i = j
axiom opposite_faces_sum : ∀ {i j : ℕ}, dice i + dice j = 7
axiom configuration : True -- Placeholder for the specific arrangement configuration

-- Questions and Proof Statements
theorem number_of_dots_on_A :
  A = 3 := sorry

theorem number_of_dots_on_B :
  B = 5 := sorry

theorem number_of_dots_on_C :
  C = 6 := sorry

theorem number_of_dots_on_D :
  D = 5 := sorry

end NUMINAMATH_GPT_number_of_dots_on_A_number_of_dots_on_B_number_of_dots_on_C_number_of_dots_on_D_l2358_235874


namespace NUMINAMATH_GPT_evaluate_expression_l2358_235810

theorem evaluate_expression : 
  (900 * 900) / ((306 * 306) - (294 * 294)) = 112.5 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2358_235810


namespace NUMINAMATH_GPT_smallest_integer_square_l2358_235830

theorem smallest_integer_square (x : ℤ) (h : x^2 = 2 * x + 75) : x = -7 :=
  sorry

end NUMINAMATH_GPT_smallest_integer_square_l2358_235830


namespace NUMINAMATH_GPT_line_y_intercept_l2358_235894

theorem line_y_intercept (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : x2 = 6) (h4 : y2 = 9) :
  ∃ b : ℝ, b = -9 := 
by
  sorry

end NUMINAMATH_GPT_line_y_intercept_l2358_235894


namespace NUMINAMATH_GPT_percentage_difference_l2358_235846

variable (x y z : ℝ)

theorem percentage_difference (h1 : y = 1.75 * x) (h2 : z = 0.60 * y) :
  (1 - x / z) * 100 = 4.76 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l2358_235846


namespace NUMINAMATH_GPT_fraction_people_over_65_l2358_235876

theorem fraction_people_over_65 (T : ℕ) (F : ℕ) : 
  (3:ℚ) / 7 * T = 24 ∧ 50 < T ∧ T < 100 → T = 56 ∧ ∃ F : ℕ, (F / 56 : ℚ) = F / (T : ℚ) :=
by 
  sorry

end NUMINAMATH_GPT_fraction_people_over_65_l2358_235876


namespace NUMINAMATH_GPT_least_number_to_subtract_l2358_235820

-- Define the problem and prove that this number, when subtracted, makes the original number divisible by 127.
theorem least_number_to_subtract (n : ℕ) (h₁ : n = 100203) (h₂ : 127 > 0) : 
  ∃ k : ℕ, (100203 - 72) = 127 * k :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l2358_235820


namespace NUMINAMATH_GPT_sunzi_system_l2358_235893

variable (x y : ℝ)

theorem sunzi_system :
  (y - x = 4.5) ∧ (x - (1/2) * y = 1) :=
by
  sorry

end NUMINAMATH_GPT_sunzi_system_l2358_235893


namespace NUMINAMATH_GPT_percentage_cleared_land_l2358_235898

theorem percentage_cleared_land (T C : ℝ) (hT : T = 6999.999999999999) (hC : 0.20 * C + 0.70 * C + 630 = C) :
  (C / T) * 100 = 90 :=
by {
  sorry
}

end NUMINAMATH_GPT_percentage_cleared_land_l2358_235898


namespace NUMINAMATH_GPT_zinc_percentage_in_1_gram_antacid_l2358_235845

theorem zinc_percentage_in_1_gram_antacid :
  ∀ (z1 z2 : ℕ → ℤ) (total_zinc : ℤ),
    z1 0 = 2 ∧ z2 0 = 2 ∧ z1 1 = 1 ∧ total_zinc = 650 ∧
    (z1 0) * 2 * 5 / 100 + (z2 1) * 3 = total_zinc / 100 →
    (z2 1) * 100 = 15 :=
by
  sorry

end NUMINAMATH_GPT_zinc_percentage_in_1_gram_antacid_l2358_235845


namespace NUMINAMATH_GPT_problem_1_problem_2_l2358_235800

noncomputable def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x + a < 0}

theorem problem_1 (a : ℝ) :
  a = -2 →
  A ∩ B a = {x | (1 / 2 : ℝ) ≤ x ∧ x < 2} :=
by
  intro ha
  sorry

theorem problem_2 (a : ℝ) :
  (A ∩ B a) = A → a < -3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2358_235800


namespace NUMINAMATH_GPT_solve_for_x_l2358_235822

theorem solve_for_x (x : ℝ) (h : 3 * x + 20 = (1 / 3) * (7 * x + 45)) : x = -7.5 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2358_235822


namespace NUMINAMATH_GPT_value_of_f_5_l2358_235863

-- Define the function f
def f (x y : ℕ) : ℕ := 2 * x ^ 2 + y

-- Given conditions
variable (some_value : ℕ)
axiom h1 : f some_value 52 = 60
axiom h2 : f 5 52 = 102

-- Proof statement
theorem value_of_f_5 : f 5 52 = 102 := by
  sorry

end NUMINAMATH_GPT_value_of_f_5_l2358_235863


namespace NUMINAMATH_GPT_fraction_of_boys_among_attendees_l2358_235839

def boys : ℕ := sorry
def girls : ℕ := boys
def teachers : ℕ := boys / 2

def boys_attending : ℕ := (4 * boys) / 5
def girls_attending : ℕ := girls / 2
def teachers_attending : ℕ := teachers / 10

theorem fraction_of_boys_among_attendees :
  (boys_attending : ℚ) / (boys_attending + girls_attending + teachers_attending) = 16 / 27 := sorry

end NUMINAMATH_GPT_fraction_of_boys_among_attendees_l2358_235839


namespace NUMINAMATH_GPT_part1_part2_l2358_235804

-- Given conditions for part (Ⅰ)
variables {a_n : ℕ → ℝ} {S_n : ℕ → ℝ}

-- The general formula for the sequence {a_n}
theorem part1 (a3_eq : a_n 3 = 1 / 8)
  (arith_seq : S_n 2 + 1 / 16 = 2 * S_n 3 - S_n 4) :
  ∀ n, a_n n = (1 / 2)^n := sorry

-- Given conditions for part (Ⅱ)
variables {b_n : ℕ → ℝ} {T_n : ℕ → ℝ}

-- The sum of the first n terms of the sequence {b_n}
theorem part2 (h_general : ∀ n, a_n n = (1 / 2)^n)
  (b_formula : ∀ n, b_n n = a_n n * (Real.log (a_n n) / Real.log (1 / 2))) :
  ∀ n, T_n n = 2 - (n + 2) / 2^n := sorry

end NUMINAMATH_GPT_part1_part2_l2358_235804


namespace NUMINAMATH_GPT_value_of_a3_a6_a9_l2358_235884

variable (a : ℕ → ℝ) (d : ℝ)

-- Condition: The common difference is 2
axiom common_difference : d = 2

-- Condition: a_1 + a_4 + a_7 = -50
axiom sum_a1_a4_a7 : a 1 + a 4 + a 7 = -50

-- The goal: a_3 + a_6 + a_9 = -38
theorem value_of_a3_a6_a9 : a 3 + a 6 + a 9 = -38 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_a3_a6_a9_l2358_235884


namespace NUMINAMATH_GPT_cargo_arrival_day_l2358_235816

-- Definitions based on conditions
def navigation_days : Nat := 21
def customs_days : Nat := 4
def warehouse_days_from_today : Nat := 2
def departure_days_ago : Nat := 30

-- Definition represents the total transit time
def total_transit_days : Nat := navigation_days + customs_days + warehouse_days_from_today

-- Theorem to prove the cargo always arrives at the rural warehouse 1 day after leaving the port in Vancouver
theorem cargo_arrival_day : 
  (departure_days_ago - total_transit_days + warehouse_days_from_today = 1) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_cargo_arrival_day_l2358_235816


namespace NUMINAMATH_GPT_correct_pronoun_possessive_l2358_235805

theorem correct_pronoun_possessive : 
  (∃ (pronoun : String), 
    pronoun = "whose" ∧ 
    pronoun = "whose" ∨ pronoun = "who" ∨ pronoun = "that" ∨ pronoun = "which") := 
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_correct_pronoun_possessive_l2358_235805


namespace NUMINAMATH_GPT_crayons_per_row_correct_l2358_235859

-- Declare the given conditions
def total_crayons : ℕ := 210
def num_rows : ℕ := 7

-- Define the expected number of crayons per row
def crayons_per_row : ℕ := 30

-- The desired proof statement: Prove that dividing total crayons by the number of rows yields the expected crayons per row.
theorem crayons_per_row_correct : total_crayons / num_rows = crayons_per_row :=
by sorry

end NUMINAMATH_GPT_crayons_per_row_correct_l2358_235859


namespace NUMINAMATH_GPT_intersection_eq_interval_l2358_235885

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x < 5}

theorem intersection_eq_interval : M ∩ N = {x | 1 < x ∧ x < 5} :=
sorry

end NUMINAMATH_GPT_intersection_eq_interval_l2358_235885


namespace NUMINAMATH_GPT_difference_between_min_and_max_l2358_235836

noncomputable 
def minValue (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ := 0

noncomputable
def maxValue (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ := 1.5

theorem difference_between_min_and_max (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  maxValue x z hx hz - minValue x z hx hz = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_min_and_max_l2358_235836


namespace NUMINAMATH_GPT_max_min_magnitude_of_sum_l2358_235837

open Real

-- Define the vectors a and b and their magnitudes
variables {a b : ℝ × ℝ}
variable (h_a : ‖a‖ = 5)
variable (h_b : ‖b‖ = 2)

-- Define the constant 7 and 3 for the max and min values
noncomputable def max_magnitude : ℝ := 7
noncomputable def min_magnitude : ℝ := 3

-- State the theorem
theorem max_min_magnitude_of_sum (h_a : ‖a‖ = 5) (h_b : ‖b‖ = 2) :
  ‖a + b‖ ≤ max_magnitude ∧ ‖a + b‖ ≥ min_magnitude :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_max_min_magnitude_of_sum_l2358_235837


namespace NUMINAMATH_GPT_major_axis_double_minor_axis_l2358_235833

-- Define the radius of the right circular cylinder.
def cylinder_radius := 2

-- Define the minor axis length based on the cylinder's radius.
def minor_axis_length := 2 * cylinder_radius

-- Define the major axis length as double the minor axis length.
def major_axis_length := 2 * minor_axis_length

-- State the theorem to prove the major axis length.
theorem major_axis_double_minor_axis : major_axis_length = 8 := by
  sorry

end NUMINAMATH_GPT_major_axis_double_minor_axis_l2358_235833


namespace NUMINAMATH_GPT_arithmetic_sequence_condition_l2358_235871

theorem arithmetic_sequence_condition (a : ℕ → ℝ) (h : 2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36) : 
a 6 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_condition_l2358_235871


namespace NUMINAMATH_GPT_modulus_Z_l2358_235807

theorem modulus_Z (Z : ℂ) (h : Z * (2 - 3 * Complex.I) = 6 + 4 * Complex.I) : Complex.abs Z = 2 := 
sorry

end NUMINAMATH_GPT_modulus_Z_l2358_235807


namespace NUMINAMATH_GPT_total_money_correct_l2358_235861

-- Define the number of pennies and quarters Sam has
def pennies : ℕ := 9
def quarters : ℕ := 7

-- Define the value of one penny and one quarter
def penny_value : ℝ := 0.01
def quarter_value : ℝ := 0.25

-- Calculate the total value of pennies and quarters Sam has
def total_value : ℝ := pennies * penny_value + quarters * quarter_value

-- Proof problem: Prove that the total value of money Sam has is $1.84
theorem total_money_correct : total_value = 1.84 :=
sorry

end NUMINAMATH_GPT_total_money_correct_l2358_235861


namespace NUMINAMATH_GPT_total_flowers_l2358_235886

theorem total_flowers (R T L : ℕ) 
  (hR : R = 58)
  (hT : R = T + 15)
  (hL : R = L - 25) :
  R + T + L = 184 :=
by 
  sorry

end NUMINAMATH_GPT_total_flowers_l2358_235886


namespace NUMINAMATH_GPT_minimum_positive_announcements_l2358_235866

theorem minimum_positive_announcements (x y : ℕ) (h : x * (x - 1) = 132) (positive_products negative_products : ℕ)
  (hp : positive_products = y * (y - 1)) (hn : negative_products = (x - y) * (x - y - 1)) 
  (h_sum : positive_products + negative_products = 132) : 
  y = 2 :=
by sorry

end NUMINAMATH_GPT_minimum_positive_announcements_l2358_235866


namespace NUMINAMATH_GPT_smaller_solid_volume_l2358_235826

noncomputable def cube_edge_length : ℝ := 2

def point (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def D := point 0 0 0
def M := point 1 2 0
def N := point 2 0 1

-- Define the condition for the plane that passes through D, M, and N
def plane (p r q : ℝ × ℝ × ℝ) (x y z : ℝ) : Prop :=
  let (px, py, pz) := p
  let (rx, ry, rz) := r
  let (qx, qy, qz) := q
  2 * x - 4 * y - 8 * z = 0

-- Predicate to test if point is on a plane
def on_plane (pt : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := pt
  plane D M N x y z

-- Volume of the smaller solid
theorem smaller_solid_volume :
  ∃ V : ℝ, V = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_smaller_solid_volume_l2358_235826


namespace NUMINAMATH_GPT_sum_of_coefficients_eq_zero_l2358_235892

theorem sum_of_coefficients_eq_zero 
  (A B C D E F : ℝ) :
  (∀ x, (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) 
  = A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
by sorry

end NUMINAMATH_GPT_sum_of_coefficients_eq_zero_l2358_235892


namespace NUMINAMATH_GPT_minimum_sum_of_distances_squared_l2358_235857

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -2, y := 0 }
def B : Point := { x := 2, y := 0 }

-- Define the moving point P on the circle
def on_circle (P : Point) : Prop :=
  (P.x - 3)^2 + (P.y - 4)^2 = 4

-- Distance squared between two points
def dist_squared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the sum of squared distances from P to points A and B
def sum_distances_squared (P : Point) : ℝ :=
  dist_squared P A + dist_squared P B

-- Statement of the proof problem
theorem minimum_sum_of_distances_squared :
  ∃ P : Point, on_circle P ∧ sum_distances_squared P = 26 :=
sorry

end NUMINAMATH_GPT_minimum_sum_of_distances_squared_l2358_235857


namespace NUMINAMATH_GPT_solution_points_satisfy_equation_l2358_235890

theorem solution_points_satisfy_equation (x y : ℝ) :
  x^2 * (y + y^2) = y^3 + x^4 → (y = x ∨ y = -x ∨ y = x^2) := sorry

end NUMINAMATH_GPT_solution_points_satisfy_equation_l2358_235890


namespace NUMINAMATH_GPT_impossible_result_l2358_235817

noncomputable def f (a b : ℝ) (c : ℤ) (x : ℝ) : ℝ :=
  a * Real.sin x + b * x + c

theorem impossible_result (a b : ℝ) (c : ℤ) :
  ¬(f a b c 1 = 1 ∧ f a b c (-1) = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_impossible_result_l2358_235817


namespace NUMINAMATH_GPT_solve_cubic_eq_l2358_235842

theorem solve_cubic_eq (x : ℝ) : x^3 + (2 - x)^3 = 8 ↔ x = 0 ∨ x = 2 := 
by 
  { sorry }

end NUMINAMATH_GPT_solve_cubic_eq_l2358_235842


namespace NUMINAMATH_GPT_state_tax_percentage_l2358_235824

theorem state_tax_percentage (weekly_salary federal_percent health_insurance life_insurance parking_fee final_paycheck : ℝ)
  (h_weekly_salary : weekly_salary = 450)
  (h_federal_percent : federal_percent = 1/3)
  (h_health_insurance : health_insurance = 50)
  (h_life_insurance : life_insurance = 20)
  (h_parking_fee : parking_fee = 10)
  (h_final_paycheck : final_paycheck = 184) :
  (36 / 450) * 100 = 8 :=
by
  sorry

end NUMINAMATH_GPT_state_tax_percentage_l2358_235824


namespace NUMINAMATH_GPT_muffin_banana_costs_l2358_235848

variable (m b : ℕ) -- Using natural numbers for non-negativity

theorem muffin_banana_costs (h : 3 * (3 * m + 5 * b) = 4 * m + 10 * b) : m = b :=
by
  sorry

end NUMINAMATH_GPT_muffin_banana_costs_l2358_235848


namespace NUMINAMATH_GPT_find_alpha_polar_equation_l2358_235828

noncomputable def alpha := (3 * Real.pi) / 4

theorem find_alpha (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) (A B : ℝ × ℝ)
  (hP : P = (2, 1))
  (hl_def : ∀ t, l t = (2 + t * Real.cos alpha, 1 + t * Real.sin alpha))
  (hA : ∃ t, l t = (A.1, 0) ∧ A.1 > 0)
  (hB : ∃ t, l t = (0, B.2) ∧ B.2 > 0)
  (h_cond : dist P A * dist P B = 4) : alpha = (3 * Real.pi) / 4 :=
sorry

theorem polar_equation (l : ℝ → ℝ × ℝ)
  (hl_def : ∀ t, l t = (2 + t * Real.cos alpha, 1 + t * Real.sin alpha))
  (h_alpha : alpha = (3 * Real.pi) / 4)
  (h_polar : ∀ ρ θ, l ρ = (ρ * Real.cos θ, ρ * Real.sin θ))
  : ∀ ρ θ, ρ * (Real.cos θ + Real.sin θ) = 3 :=
sorry

end NUMINAMATH_GPT_find_alpha_polar_equation_l2358_235828


namespace NUMINAMATH_GPT_solve_for_x_l2358_235815

theorem solve_for_x (x : ℝ) :
  (x - 2)^6 + (x - 6)^6 = 64 → x = 3 ∨ x = 5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solve_for_x_l2358_235815


namespace NUMINAMATH_GPT_find_age_l2358_235853

theorem find_age (A : ℤ) (h : 4 * (A + 4) - 4 * (A - 4) = A) : A = 32 :=
by sorry

end NUMINAMATH_GPT_find_age_l2358_235853


namespace NUMINAMATH_GPT_cleaner_used_after_30_minutes_l2358_235840

-- Define function to calculate the total amount of cleaner used
def total_cleaner_used (time: ℕ) (rate1: ℕ) (time1: ℕ) (rate2: ℕ) (time2: ℕ) (rate3: ℕ) (time3: ℕ) : ℕ :=
  (rate1 * time1) + (rate2 * time2) + (rate3 * time3)

-- The main theorem statement
theorem cleaner_used_after_30_minutes : total_cleaner_used 30 2 15 3 10 4 5 = 80 := by
  -- insert proof here
  sorry

end NUMINAMATH_GPT_cleaner_used_after_30_minutes_l2358_235840


namespace NUMINAMATH_GPT_rank_from_left_l2358_235858

theorem rank_from_left (total_students : ℕ) (rank_from_right : ℕ) (h1 : total_students = 20) (h2 : rank_from_right = 13) : 
  (total_students - rank_from_right + 1 = 8) :=
by
  sorry

end NUMINAMATH_GPT_rank_from_left_l2358_235858


namespace NUMINAMATH_GPT_roll_seven_dice_at_least_one_pair_no_three_l2358_235843

noncomputable def roll_seven_dice_probability : ℚ :=
  let total_outcomes := (6^7 : ℚ)
  let one_pair_case := (6 * 21 * 120 : ℚ)
  let two_pairs_case := (15 * 21 * 10 * 24 : ℚ)
  let successful_outcomes := one_pair_case + two_pairs_case
  successful_outcomes / total_outcomes

theorem roll_seven_dice_at_least_one_pair_no_three :
  roll_seven_dice_probability = 315 / 972 :=
by
  unfold roll_seven_dice_probability
  -- detailed steps to show the proof would go here
  sorry

end NUMINAMATH_GPT_roll_seven_dice_at_least_one_pair_no_three_l2358_235843


namespace NUMINAMATH_GPT_total_hours_charged_l2358_235814

variables (K P M : ℕ)

theorem total_hours_charged (h1 : P = 2 * K) 
                            (h2 : P = (1/3 : ℚ) * (M : ℚ)) 
                            (h3 : M = K + 85) : 
  K + P + M = 153 := 
by 
  sorry

end NUMINAMATH_GPT_total_hours_charged_l2358_235814


namespace NUMINAMATH_GPT_number_of_raised_beds_l2358_235841

def length_feed := 8
def width_feet := 4
def height_feet := 1
def cubic_feet_per_bag := 4
def total_bags_needed := 16

theorem number_of_raised_beds :
  ∀ (length_feed width_feet height_feet : ℕ) (cubic_feet_per_bag total_bags_needed : ℕ),
    (length_feed * width_feet * height_feet) / cubic_feet_per_bag = 8 →
    total_bags_needed / (8 : ℕ) = 2 :=
by sorry

end NUMINAMATH_GPT_number_of_raised_beds_l2358_235841


namespace NUMINAMATH_GPT_billy_reads_60_pages_per_hour_l2358_235860

theorem billy_reads_60_pages_per_hour
  (free_time_per_day : ℕ)
  (days : ℕ)
  (video_games_time_percentage : ℝ)
  (books : ℕ)
  (pages_per_book : ℕ)
  (remaining_time_percentage : ℝ)
  (total_free_time := free_time_per_day * days)
  (time_playing_video_games := video_games_time_percentage * total_free_time)
  (time_reading := remaining_time_percentage * total_free_time)
  (total_pages := books * pages_per_book)
  (pages_per_hour := total_pages / time_reading) :
  free_time_per_day = 8 →
  days = 2 →
  video_games_time_percentage = 0.75 →
  remaining_time_percentage = 0.25 →
  books = 3 →
  pages_per_book = 80 →
  pages_per_hour = 60 :=
by
  intros
  sorry

end NUMINAMATH_GPT_billy_reads_60_pages_per_hour_l2358_235860


namespace NUMINAMATH_GPT_sin_300_eq_neg_sqrt_three_div_two_l2358_235823

theorem sin_300_eq_neg_sqrt_three_div_two :
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_300_eq_neg_sqrt_three_div_two_l2358_235823


namespace NUMINAMATH_GPT_natural_number_triplets_l2358_235825

theorem natural_number_triplets :
  ∀ (a b c : ℕ), a^3 + b^3 + c^3 = (a * b * c)^2 → 
    (a = 3 ∧ b = 2 ∧ c = 1) ∨ (a = 3 ∧ b = 1 ∧ c = 2) ∨ 
    (a = 2 ∧ b = 3 ∧ c = 1) ∨ (a = 2 ∧ b = 1 ∧ c = 3) ∨ 
    (a = 1 ∧ b = 3 ∧ c = 2) ∨ (a = 1 ∧ b = 2 ∧ c = 3) := 
by
  sorry

end NUMINAMATH_GPT_natural_number_triplets_l2358_235825


namespace NUMINAMATH_GPT_unique_coprime_solution_l2358_235834

theorem unique_coprime_solution 
  (p : ℕ) (a b m r : ℕ) 
  (hp : Nat.Prime p) 
  (hp_odd : p % 2 = 1)
  (hp_nmid_ab : ¬ (p ∣ a * b))
  (hab_gt_m2 : a * b > m^2) :
  ∃! (x y : ℕ), Nat.Coprime x y ∧ (a * x^2 + b * y^2 = m * p ^ r) := 
sorry

end NUMINAMATH_GPT_unique_coprime_solution_l2358_235834


namespace NUMINAMATH_GPT_sin_beta_value_sin2alpha_over_cos2alpha_plus_cos2alpha_value_l2358_235813

open Real

noncomputable def problem_conditions (α β : ℝ) : Prop :=
  0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧
  cos α = 3/5 ∧ cos (β + α) = 5/13

theorem sin_beta_value 
  {α β : ℝ} (h : problem_conditions α β) : 
  sin β = 16 / 65 :=
sorry

theorem sin2alpha_over_cos2alpha_plus_cos2alpha_value
  {α β : ℝ} (h : problem_conditions α β) : 
  (sin (2 * α)) / (cos α^2 + cos (2 * α)) = 12 :=
sorry

end NUMINAMATH_GPT_sin_beta_value_sin2alpha_over_cos2alpha_plus_cos2alpha_value_l2358_235813


namespace NUMINAMATH_GPT_masha_can_pay_with_5_ruble_coins_l2358_235868

theorem masha_can_pay_with_5_ruble_coins (p c n : ℤ) (h : 2 * p + c + 7 * n = 100) : (p + 3 * c + n) % 5 = 0 :=
  sorry

end NUMINAMATH_GPT_masha_can_pay_with_5_ruble_coins_l2358_235868


namespace NUMINAMATH_GPT_problem_a_l2358_235832

def part_a : Prop :=
  ∃ (tokens : Finset (Fin 4 × Fin 4)), 
    tokens.card = 7 ∧ 
    (∀ (rows : Finset (Fin 4)) (cols : Finset (Fin 4)), rows.card = 2 → cols.card = 2 → 
      ∃ (token : (Fin 4 × Fin 4)), token ∈ tokens ∧ token.1 ∉ rows ∧ token.2 ∉ cols)

theorem problem_a : part_a :=
  sorry

end NUMINAMATH_GPT_problem_a_l2358_235832


namespace NUMINAMATH_GPT_sand_problem_l2358_235850

-- Definitions based on conditions
def initial_sand := 1050
def sand_lost_first := 32
def sand_lost_second := 67
def sand_lost_third := 45
def sand_lost_fourth := 54

-- Total sand lost
def total_sand_lost := sand_lost_first + sand_lost_second + sand_lost_third + sand_lost_fourth

-- Sand remaining
def sand_remaining := initial_sand - total_sand_lost

-- Theorem stating the proof problem
theorem sand_problem : sand_remaining = 852 :=
by
-- Skipping proof as per instructions
sorry

end NUMINAMATH_GPT_sand_problem_l2358_235850


namespace NUMINAMATH_GPT_values_of_a_and_b_l2358_235811

def is_root (a b x : ℝ) : Prop := x^2 - 2*a*x + b = 0

noncomputable def A : Set ℝ := {-1, 1}
noncomputable def B (a b : ℝ) : Set ℝ := {x | is_root a b x}

theorem values_of_a_and_b (a b : ℝ) (h_nonempty : Set.Nonempty (B a b)) (h_union : A ∪ B a b = A) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1) :=
sorry

end NUMINAMATH_GPT_values_of_a_and_b_l2358_235811


namespace NUMINAMATH_GPT_length_vector_eq_three_l2358_235882

theorem length_vector_eq_three (A B : ℝ) (hA : A = -1) (hB : B = 2) : |B - A| = 3 :=
by
  sorry

end NUMINAMATH_GPT_length_vector_eq_three_l2358_235882


namespace NUMINAMATH_GPT_simplify_expression_correct_l2358_235887

def simplify_expression (x : ℝ) : Prop :=
  2 * x - 3 * (2 - x) + 4 * (2 + 3 * x) - 5 * (1 - 2 * x) = 27 * x - 3

theorem simplify_expression_correct (x : ℝ) : simplify_expression x :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_correct_l2358_235887


namespace NUMINAMATH_GPT_water_depth_when_upright_l2358_235851
-- Import the entire Mathlib library

-- Define the conditions and question as a theorem
theorem water_depth_when_upright (height : ℝ) (diameter : ℝ) (horizontal_depth : ℝ) :
  height = 20 → diameter = 6 → horizontal_depth = 4 → water_depth = 5.3 :=
by
  intro h1 h2 h3
  -- The proof would go here, but we insert sorry to skip it
  sorry

end NUMINAMATH_GPT_water_depth_when_upright_l2358_235851


namespace NUMINAMATH_GPT_evaluate_expression_l2358_235852

variables {a b c d e : ℝ}

theorem evaluate_expression (a b c d e : ℝ) : a * b^c - d + e = a * (b^c - (d + e)) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2358_235852


namespace NUMINAMATH_GPT_slope_of_CD_l2358_235812

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10 * x - 2 * y + 40 = 0

-- Theorem statement
theorem slope_of_CD :
  ∃ C D : ℝ × ℝ,
    (circle1 C.1 C.2) ∧ (circle2 C.1 C.2) ∧ (circle1 D.1 D.2) ∧ (circle2 D.1 D.2) ∧
    (∃ m : ℝ, m = -2 / 3) := 
  sorry

end NUMINAMATH_GPT_slope_of_CD_l2358_235812


namespace NUMINAMATH_GPT_hexagonal_prism_surface_area_l2358_235895

theorem hexagonal_prism_surface_area (h : ℝ) (a : ℝ) (H_h : h = 6) (H_a : a = 4) : 
  let base_area := 2 * (3 * a^2 * (Real.sqrt 3) / 2)
  let lateral_area := 6 * a * h
  let total_area := lateral_area + base_area
  total_area = 48 * (3 + Real.sqrt 3) :=
by
  -- let base_area := 2 * (3 * a^2 * (Real.sqrt 3) / 2)
  -- let lateral_area := 6 * a * h
  -- let total_area := lateral_area + base_area
  -- total_area = 48 * (3 + Real.sqrt 3)
  sorry

end NUMINAMATH_GPT_hexagonal_prism_surface_area_l2358_235895


namespace NUMINAMATH_GPT_sum_of_integers_l2358_235897

theorem sum_of_integers (m n p q : ℤ) 
(h1 : m ≠ n) (h2 : m ≠ p) 
(h3 : m ≠ q) (h4 : n ≠ p) 
(h5 : n ≠ q) (h6 : p ≠ q) 
(h7 : (5 - m) * (5 - n) * (5 - p) * (5 - q) = 9) : 
m + n + p + q = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l2358_235897


namespace NUMINAMATH_GPT_least_number_division_remainder_4_l2358_235847

theorem least_number_division_remainder_4 : 
  ∃ n : Nat, (n % 6 = 4) ∧ (n % 130 = 4) ∧ (n % 9 = 4) ∧ (n % 18 = 4) ∧ n = 2344 :=
by
  sorry

end NUMINAMATH_GPT_least_number_division_remainder_4_l2358_235847


namespace NUMINAMATH_GPT_pairwise_coprime_triples_l2358_235878

open Nat

theorem pairwise_coprime_triples (a b c : ℕ) 
  (h1 : a.gcd b = 1) (h2 : a.gcd c = 1) (h3 : b.gcd c = 1)
  (h4 : (a + b) ∣ c) (h5 : (a + c) ∣ b) (h6 : (b + c) ∣ a) :
  { (a, b, c) | (a = 1 ∧ b = 1 ∧ (c = 1 ∨ c = 2)) ∨ (a = 1 ∧ b = 2 ∧ c = 3) } :=
by
  -- Proof omitted for conciseness
  sorry

end NUMINAMATH_GPT_pairwise_coprime_triples_l2358_235878


namespace NUMINAMATH_GPT_isabelle_weeks_needed_l2358_235873

def total_ticket_cost : ℕ := 20 + 10 + 10
def total_savings : ℕ := 5 + 5
def weekly_earnings : ℕ := 3
def amount_needed : ℕ := total_ticket_cost - total_savings
def weeks_needed : ℕ := amount_needed / weekly_earnings

theorem isabelle_weeks_needed 
  (ticket_cost_isabelle : ℕ := 20)
  (ticket_cost_brother : ℕ := 10)
  (savings_brothers : ℕ := 5)
  (savings_isabelle : ℕ := 5)
  (earnings_weekly : ℕ := 3)
  (total_cost := ticket_cost_isabelle + 2 * ticket_cost_brother)
  (total_savings := savings_brothers + savings_isabelle)
  (needed_amount := total_cost - total_savings)
  (weeks := needed_amount / earnings_weekly) :
  weeks = 10 :=
  by
  sorry

end NUMINAMATH_GPT_isabelle_weeks_needed_l2358_235873


namespace NUMINAMATH_GPT_speed_ratio_thirteen_l2358_235855

noncomputable section

def speed_ratio (vNikita vCar : ℝ) : ℝ := vCar / vNikita

theorem speed_ratio_thirteen :
  ∀ (vNikita vCar : ℝ),
  (65 * vNikita = 5 * vCar) →
  speed_ratio vNikita vCar = 13 :=
by
  intros vNikita vCar h
  unfold speed_ratio
  sorry

end NUMINAMATH_GPT_speed_ratio_thirteen_l2358_235855


namespace NUMINAMATH_GPT_sum_of_diffs_is_10_l2358_235802

-- Define the number of fruits each person has
def Sharon_plums : ℕ := 7
def Allan_plums : ℕ := 10
def Dave_oranges : ℕ := 12

-- Define the differences in the number of fruits
def diff_Sharon_Allan : ℕ := Allan_plums - Sharon_plums
def diff_Sharon_Dave : ℕ := Dave_oranges - Sharon_plums
def diff_Allan_Dave : ℕ := Dave_oranges - Allan_plums

-- Define the sum of these differences
def sum_of_diffs : ℕ := diff_Sharon_Allan + diff_Sharon_Dave + diff_Allan_Dave

-- State the theorem to be proved
theorem sum_of_diffs_is_10 : sum_of_diffs = 10 := by
  sorry

end NUMINAMATH_GPT_sum_of_diffs_is_10_l2358_235802


namespace NUMINAMATH_GPT_number_of_primary_schools_l2358_235849

theorem number_of_primary_schools (A B total : ℕ) (h1 : A = 2 * 400)
  (h2 : B = 2 * 340) (h3 : total = 1480) (h4 : total = A + B) :
  2 + 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_primary_schools_l2358_235849


namespace NUMINAMATH_GPT_sum_of_remainders_mod_500_l2358_235827

theorem sum_of_remainders_mod_500 : 
  (5 ^ (5 ^ (5 ^ 5)) + 2 ^ (2 ^ (2 ^ 2))) % 500 = 49 := by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_mod_500_l2358_235827


namespace NUMINAMATH_GPT_boys_seen_l2358_235899

theorem boys_seen (total_eyes : ℕ) (eyes_per_boy : ℕ) (h1 : total_eyes = 46) (h2 : eyes_per_boy = 2) : total_eyes / eyes_per_boy = 23 := 
by 
  sorry

end NUMINAMATH_GPT_boys_seen_l2358_235899


namespace NUMINAMATH_GPT_expression_positive_l2358_235881

variable {a b c : ℝ}

theorem expression_positive (h₀ : 0 < a ∧ a < 2) (h₁ : -2 < b ∧ b < 0) : 0 < b + a^2 :=
by
  sorry

end NUMINAMATH_GPT_expression_positive_l2358_235881


namespace NUMINAMATH_GPT_inequality_solution_nonempty_l2358_235821

theorem inequality_solution_nonempty (a : ℝ) :
  (∃ x : ℝ, x ^ 2 - a * x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_nonempty_l2358_235821


namespace NUMINAMATH_GPT_verify_z_relationship_l2358_235867

variable {x y z : ℝ}

theorem verify_z_relationship (h1 : x > y) (h2 : y > 1) :
  z = (x + 3) - 2 * (y - 5) → z = x - 2 * y + 13 :=
by
  intros
  sorry

end NUMINAMATH_GPT_verify_z_relationship_l2358_235867


namespace NUMINAMATH_GPT_simplify_abs_eq_l2358_235889

variable {x : ℚ}

theorem simplify_abs_eq (hx : |1 - x| = 1 + |x|) : |x - 1| = 1 - x :=
by
  sorry

end NUMINAMATH_GPT_simplify_abs_eq_l2358_235889


namespace NUMINAMATH_GPT_final_weight_is_200_l2358_235818

def initial_weight : ℕ := 220
def percentage_lost : ℕ := 10
def weight_gained : ℕ := 2

theorem final_weight_is_200 :
  initial_weight - (initial_weight * percentage_lost / 100) + weight_gained = 200 := by
  sorry

end NUMINAMATH_GPT_final_weight_is_200_l2358_235818


namespace NUMINAMATH_GPT_fraction_of_tomatoes_eaten_l2358_235883

theorem fraction_of_tomatoes_eaten (original : ℕ) (remaining : ℕ) (birds_ate : ℕ) (h1 : original = 21) (h2 : remaining = 14) (h3 : birds_ate = original - remaining) :
  (birds_ate : ℚ) / original = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_tomatoes_eaten_l2358_235883


namespace NUMINAMATH_GPT_circle_value_l2358_235838

theorem circle_value (c d s : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 - 8*x + y^2 + 16*y = -100 ↔ (x - c)^2 + (y + d)^2 = s^2)
  (h2 : c = 4)
  (h3 : d = -8)
  (h4 : s = 2 * Real.sqrt 5) :
  c + d + s = -4 + 2 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_circle_value_l2358_235838


namespace NUMINAMATH_GPT_min_value_x_plus_4_div_x_plus_1_l2358_235862

theorem min_value_x_plus_4_div_x_plus_1 (x : ℝ) (h : x > -1) : ∃ m, m = 3 ∧ ∀ y, y = x + 4 / (x + 1) → y ≥ m :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_4_div_x_plus_1_l2358_235862


namespace NUMINAMATH_GPT_total_price_before_increase_l2358_235879

-- Conditions
def original_price_candy_box (c_or: ℝ) := 10 = c_or * 1.25
def original_price_soda_can (s_or: ℝ) := 15 = s_or * 1.50

-- Goal
theorem total_price_before_increase :
  ∃ (c_or s_or : ℝ), original_price_candy_box c_or ∧ original_price_soda_can s_or ∧ c_or + s_or = 25 :=
by
  sorry

end NUMINAMATH_GPT_total_price_before_increase_l2358_235879


namespace NUMINAMATH_GPT_smallest_n_congruent_5n_eq_n5_mod_7_l2358_235844

theorem smallest_n_congruent_5n_eq_n5_mod_7 : ∃ (n : ℕ), n > 0 ∧ (∀ m > 0, 5^m % 7 ≠ m^5 % 7 → m ≥ n) :=
by
  use 6
  -- Proof steps here which are skipped
  sorry

end NUMINAMATH_GPT_smallest_n_congruent_5n_eq_n5_mod_7_l2358_235844


namespace NUMINAMATH_GPT_proof_by_contradiction_conditions_l2358_235865

theorem proof_by_contradiction_conditions:
  (∃ (neg_conclusion known_conditions ax_thms_defs original_conclusion : Prop),
    (neg_conclusion ∧ known_conditions ∧ ax_thms_defs) → False)
:= sorry

end NUMINAMATH_GPT_proof_by_contradiction_conditions_l2358_235865


namespace NUMINAMATH_GPT_find_all_pos_integers_l2358_235835

theorem find_all_pos_integers (M : ℕ) (h1 : M > 0) (h2 : M < 10) :
  (5 ∣ (1989^M + M^1989)) ↔ (M = 1) ∨ (M = 4) :=
by
  sorry

end NUMINAMATH_GPT_find_all_pos_integers_l2358_235835


namespace NUMINAMATH_GPT_train_crossing_time_l2358_235896

noncomputable def train_speed_kmph : ℕ := 72
noncomputable def platform_length_m : ℕ := 300
noncomputable def crossing_time_platform_s : ℕ := 33
noncomputable def train_speed_mps : ℕ := (train_speed_kmph * 5) / 18

theorem train_crossing_time (L : ℕ) (hL : L + platform_length_m = train_speed_mps * crossing_time_platform_s) :
  L / train_speed_mps = 18 :=
  by
    have : train_speed_mps = 20 := by
      sorry
    have : L = 360 := by
      sorry
    sorry

end NUMINAMATH_GPT_train_crossing_time_l2358_235896


namespace NUMINAMATH_GPT_a5_value_l2358_235870

def seq (a : ℕ → ℤ) (a1 : a 1 = 2) (rec : ∀ n, a (n + 1) = 2 * a n - 1) : Prop := True

theorem a5_value : 
  ∀ (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (recurrence : ∀ n, a (n + 1) = 2 * a n - 1),
  seq a h1 recurrence → a 5 = 17 :=
by
  intros a h1 recurrence seq_a
  sorry

end NUMINAMATH_GPT_a5_value_l2358_235870


namespace NUMINAMATH_GPT_expand_polynomial_l2358_235875

theorem expand_polynomial :
  (x^2 - 3 * x + 3) * (x^2 + 3 * x + 1) = x^4 - 5 * x^2 + 6 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l2358_235875


namespace NUMINAMATH_GPT_central_angle_measure_l2358_235864

-- Given the problem definitions
variables (A : ℝ) (x : ℝ)

-- Condition: The probability of landing in the region is 1/8
def probability_condition : Prop :=
  (1 / 8 : ℝ) = (x / 360)

-- The final theorem to prove
theorem central_angle_measure (h : probability_condition x) : x = 45 := 
  sorry

end NUMINAMATH_GPT_central_angle_measure_l2358_235864


namespace NUMINAMATH_GPT_minimum_difference_l2358_235829

def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem minimum_difference (x y z : ℤ) 
  (hx : even x) (hy : odd y) (hz : odd z)
  (hxy : x < y) (hyz : y < z) (hzx : z - x = 9) : y - x = 1 := 
sorry

end NUMINAMATH_GPT_minimum_difference_l2358_235829


namespace NUMINAMATH_GPT_combinations_count_l2358_235809

theorem combinations_count:
  let valid_a (a: ℕ) := a < 1000 ∧ a % 29 = 7
  let valid_b (b: ℕ) := b < 1000 ∧ b % 47 = 22
  let valid_c (c: ℕ) (a b: ℕ) := c < 1000 ∧ c = (a + b) % 23 
  ∃ (a b c: ℕ), valid_a a ∧ valid_b b ∧ valid_c c a b :=
  sorry

end NUMINAMATH_GPT_combinations_count_l2358_235809


namespace NUMINAMATH_GPT_undefined_expression_l2358_235819

theorem undefined_expression (a : ℝ) : (a = 3 ∨ a = -3) ↔ (a^2 - 9 = 0) := 
by
  sorry

end NUMINAMATH_GPT_undefined_expression_l2358_235819


namespace NUMINAMATH_GPT_rope_cutting_impossible_l2358_235877

/-- 
Given a rope initially cut into 5 pieces, and then some of these pieces were each cut into 
5 parts, with this process repeated several times, it is not possible for the total 
number of pieces to be exactly 2019.
-/ 
theorem rope_cutting_impossible (n : ℕ) : 5 + 4 * n ≠ 2019 := 
sorry

end NUMINAMATH_GPT_rope_cutting_impossible_l2358_235877
