import Mathlib

namespace NUMINAMATH_GPT_ned_trips_l688_68846

theorem ned_trips : 
  ∀ (carry_capacity : ℕ) (table1 : ℕ) (table2 : ℕ) (table3 : ℕ) (table4 : ℕ),
  carry_capacity = 5 →
  table1 = 7 →
  table2 = 10 →
  table3 = 12 →
  table4 = 3 →
  (table1 + table2 + table3 + table4 + carry_capacity - 1) / carry_capacity = 8 :=
by
  intro carry_capacity table1 table2 table3 table4
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_ned_trips_l688_68846


namespace NUMINAMATH_GPT_circle_line_intersect_property_l688_68876
open Real

theorem circle_line_intersect_property :
  let ρ := fun θ : ℝ => 4 * sqrt 2 * sin (3 * π / 4 - θ)
  let cartesian_eq := fun x y : ℝ => (x - 2) ^ 2 + (y - 2) ^ 2 = 8
  let slope := sqrt 3
  let line_param := fun t : ℝ => (1/2 * t, 2 + sqrt 3 / 2 * t)
  let t_roots := {t | ∃ t1 t2 : ℝ, t1 + t2 = 2 ∧ t1 * t2 = -4 ∧ (t = t1 ∨ t = t2)}
  
  (∀ t ∈ t_roots, 
    let (x, y) := line_param t
    cartesian_eq x y)
  → abs ((1 : ℝ) / abs 1 - (1 : ℝ) / abs 2) = 1 / 2 :=
by
  intro ρ cartesian_eq slope line_param t_roots h
  sorry

end NUMINAMATH_GPT_circle_line_intersect_property_l688_68876


namespace NUMINAMATH_GPT_polynomial_base5_representation_l688_68836

-- Define the polynomials P and Q
def P(x : ℕ) : ℕ := 3 * 5^6 + 0 * 5^5 + 0 * 5^4 + 1 * 5^3 + 2 * 5^2 + 4 * 5 + 1
def Q(x : ℕ) : ℕ := 4 * 5^2 + 3 * 5 + 2

-- Define the representation of these polynomials in base-5
def base5_P : ℕ := 3001241
def base5_Q : ℕ := 432

-- Define the expected interpretation of the base-5 representation in decimal
def decimal_P : ℕ := P 0
def decimal_Q : ℕ := Q 0

-- The proof statement
theorem polynomial_base5_representation :
  decimal_P = base5_P ∧ decimal_Q = base5_Q :=
sorry

end NUMINAMATH_GPT_polynomial_base5_representation_l688_68836


namespace NUMINAMATH_GPT_alcohol_percentage_second_vessel_l688_68848

theorem alcohol_percentage_second_vessel:
  ∃ x : ℝ, 
  let alcohol_in_first := 0.25 * 2
  let alcohol_in_second := 0.01 * x * 6
  let total_alcohol := 0.29 * 8
  alcohol_in_first + alcohol_in_second = total_alcohol → 
  x = 30.333333333333332 :=
by
  sorry

end NUMINAMATH_GPT_alcohol_percentage_second_vessel_l688_68848


namespace NUMINAMATH_GPT_complement_in_U_l688_68800

def A : Set ℝ := { x : ℝ | |x - 1| > 3 }
def U : Set ℝ := Set.univ

theorem complement_in_U :
  (U \ A) = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
by
  sorry

end NUMINAMATH_GPT_complement_in_U_l688_68800


namespace NUMINAMATH_GPT_m_mul_m_add_1_not_power_of_integer_l688_68894

theorem m_mul_m_add_1_not_power_of_integer (m n k : ℕ) : m * (m + 1) ≠ n^k :=
by
  sorry

end NUMINAMATH_GPT_m_mul_m_add_1_not_power_of_integer_l688_68894


namespace NUMINAMATH_GPT_solve_equation_l688_68868

noncomputable def fourthRoot (x : ℝ) := Real.sqrt (Real.sqrt x)

theorem solve_equation (x : ℝ) (hx : x ≥ 0) :
  fourthRoot x = 18 / (9 - fourthRoot x) ↔ x = 81 ∨ x = 1296 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l688_68868


namespace NUMINAMATH_GPT_jennifer_book_spending_l688_68895

variable (initial_total : ℕ)
variable (spent_sandwich : ℚ)
variable (spent_museum : ℚ)
variable (money_left : ℕ)

theorem jennifer_book_spending :
  initial_total = 90 → 
  spent_sandwich = 1/5 * 90 → 
  spent_museum = 1/6 * 90 → 
  money_left = 12 →
  (initial_total - money_left - (spent_sandwich + spent_museum)) / initial_total = 1/2 :=
by
  intros h_initial_total h_spent_sandwich h_spent_museum h_money_left
  sorry

end NUMINAMATH_GPT_jennifer_book_spending_l688_68895


namespace NUMINAMATH_GPT_k_valid_iff_l688_68821

open Nat

theorem k_valid_iff (k : ℕ) :
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_k_valid_iff_l688_68821


namespace NUMINAMATH_GPT_direction_vector_arithmetic_sequence_l688_68838

theorem direction_vector_arithmetic_sequence (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) 
    (n : ℕ) 
    (S2_eq_10 : S_n 2 = 10) 
    (S5_eq_55 : S_n 5 = 55)
    (arith_seq_sum : ∀ n, S_n n = (n * (2 * a_n 1 + (n - 1) * (a_n 2 - a_n 1))) / 2): 
    (a_n (n + 2) - a_n n) / (n + 2 - n) = 4 :=
by
  sorry

end NUMINAMATH_GPT_direction_vector_arithmetic_sequence_l688_68838


namespace NUMINAMATH_GPT_area_ratio_of_squares_l688_68856

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 1 / 2 * (4 * b)) : (b^2 / a^2) = 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_area_ratio_of_squares_l688_68856


namespace NUMINAMATH_GPT_range_of_values_for_a_l688_68810

theorem range_of_values_for_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 < 0 ∧ x1^2 + a * x1 + a^2 - 1 = 0 ∧ x2^2 + a * x2 + a^2 - 1 = 0) → (-1 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_values_for_a_l688_68810


namespace NUMINAMATH_GPT_math_problem_l688_68824

variables (a b c d m : ℝ)

theorem math_problem 
  (h1 : a = -b)            -- condition 1: a and b are opposite numbers
  (h2 : c * d = 1)         -- condition 2: c and d are reciprocal numbers
  (h3 : |m| = 1) :         -- condition 3: absolute value of m is 1
  (a + b) * c * d - 2009 * m = -2009 ∨ (a + b) * c * d - 2009 * m = 2009 :=
sorry

end NUMINAMATH_GPT_math_problem_l688_68824


namespace NUMINAMATH_GPT_xy_inequality_l688_68803

theorem xy_inequality (x y θ : ℝ) 
    (h : (x * Real.cos θ + y * Real.sin θ)^2 + x * Real.sin θ - y * Real.cos θ = 1) : 
    x^2 + y^2 ≥ 3/4 :=
sorry

end NUMINAMATH_GPT_xy_inequality_l688_68803


namespace NUMINAMATH_GPT_find_a_for_positive_root_l688_68853

theorem find_a_for_positive_root (h : ∃ x > 0, (1 - x) / (x - 2) = a / (2 - x) - 2) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_for_positive_root_l688_68853


namespace NUMINAMATH_GPT_contemporaries_probability_l688_68812

theorem contemporaries_probability:
  (∀ (x y : ℝ),
    0 ≤ x ∧ x ≤ 400 ∧
    0 ≤ y ∧ y ≤ 400 ∧
    (x < y + 80) ∧ (y < x + 80)) →
    (∃ p : ℝ, p = 9 / 25) :=
by sorry

end NUMINAMATH_GPT_contemporaries_probability_l688_68812


namespace NUMINAMATH_GPT_find_second_speed_l688_68801

theorem find_second_speed (d t_b : ℝ) (v1 : ℝ) (t_m t_a : ℤ): 
  d = 13.5 ∧ v1 = 5 ∧ t_m = 12 ∧ t_a = 15 →
  (t_b = (d / v1) - (t_m / 60)) →
  (t2 = t_b - (t_a / 60)) →
  v = d / t2 →
  v = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_second_speed_l688_68801


namespace NUMINAMATH_GPT_abc_solution_l688_68892

theorem abc_solution (a b c : ℕ) (h1 : a + b = c - 1) (h2 : a^3 + b^3 = c^2 - 1) : 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ (a = 3 ∧ b = 2 ∧ c = 6) :=
sorry

end NUMINAMATH_GPT_abc_solution_l688_68892


namespace NUMINAMATH_GPT_smallest_number_l688_68817

/-
  Let's declare each number in its base form as variables,
  convert them to their decimal equivalents, and assert that the decimal
  value of $(31)_4$ is the smallest among the given numbers.

  Note: We're not providing the proof steps, just the statement.
-/

noncomputable def A_base7_to_dec : ℕ := 2 * 7^1 + 0 * 7^0
noncomputable def B_base5_to_dec : ℕ := 3 * 5^1 + 0 * 5^0
noncomputable def C_base6_to_dec : ℕ := 2 * 6^1 + 3 * 6^0
noncomputable def D_base4_to_dec : ℕ := 3 * 4^1 + 1 * 4^0

theorem smallest_number : D_base4_to_dec < A_base7_to_dec ∧ D_base4_to_dec < B_base5_to_dec ∧ D_base4_to_dec < C_base6_to_dec := by
  sorry

end NUMINAMATH_GPT_smallest_number_l688_68817


namespace NUMINAMATH_GPT_mike_reaches_office_time_l688_68831

-- Define the given conditions
def dave_steps_per_minute : ℕ := 80
def dave_step_length_cm : ℕ := 85
def dave_time_min : ℕ := 20

def mike_steps_per_minute : ℕ := 95
def mike_step_length_cm : ℕ := 70

-- Define Dave's walking speed
def dave_speed_cm_per_min : ℕ := dave_steps_per_minute * dave_step_length_cm

-- Define the total distance to the office
def distance_to_office_cm : ℕ := dave_speed_cm_per_min * dave_time_min

-- Define Mike's walking speed
def mike_speed_cm_per_min : ℕ := mike_steps_per_minute * mike_step_length_cm

-- Define the time it takes Mike to walk to the office
noncomputable def mike_time_to_office_min : ℚ := distance_to_office_cm / mike_speed_cm_per_min

-- State the theorem to prove
theorem mike_reaches_office_time :
  mike_time_to_office_min = 20.45 :=
sorry

end NUMINAMATH_GPT_mike_reaches_office_time_l688_68831


namespace NUMINAMATH_GPT_find_f_x_l688_68866

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem find_f_x (x : ℝ) : (f (x+1)) = x^2 - 3*x + 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_x_l688_68866


namespace NUMINAMATH_GPT_race_distance_l688_68845

theorem race_distance
  (x y z d : ℝ) 
  (h1 : d / x = (d - 25) / y)
  (h2 : d / y = (d - 15) / z)
  (h3 : d / x = (d - 35) / z) : 
  d = 75 := 
sorry

end NUMINAMATH_GPT_race_distance_l688_68845


namespace NUMINAMATH_GPT_max_triangle_area_l688_68862

-- Definitions for the conditions
def Point := (ℝ × ℝ)

def point_A : Point := (0, 0)
def point_B : Point := (17, 0)
def point_C : Point := (23, 0)

def slope_ell_A : ℝ := 2
def slope_ell_C : ℝ := -2

axiom rotating_clockwise_with_same_angular_velocity (A B C : Point) : Prop

-- Question transcribed as proving a statement about the maximum area
theorem max_triangle_area (A B C : Point)
  (hA : A = point_A)
  (hB : B = point_B)
  (hC : C = point_C)
  (h_slopeA : ∀ p: Point, slope_ell_A = 2)
  (h_slopeC : ∀ p: Point, slope_ell_C = -2)
  (h_rotation : rotating_clockwise_with_same_angular_velocity A B C) :
  ∃ area_max : ℝ, area_max = 264.5 :=
sorry

end NUMINAMATH_GPT_max_triangle_area_l688_68862


namespace NUMINAMATH_GPT_ellipse_condition_range_k_l688_68825

theorem ellipse_condition_range_k (k : ℝ) : 
  (2 - k > 0) ∧ (3 + k > 0) ∧ (2 - k ≠ 3 + k) → -3 < k ∧ k < 2 := 
by 
  sorry

end NUMINAMATH_GPT_ellipse_condition_range_k_l688_68825


namespace NUMINAMATH_GPT_range_of_m_l688_68818

noncomputable def f (x m : ℝ) : ℝ := -x^2 + m * x

theorem range_of_m {m : ℝ} : (∀ x y : ℝ, x ≤ y → x ≤ 1 → y ≤ 1 → f x m ≤ f y m) ↔ 2 ≤ m := 
sorry

end NUMINAMATH_GPT_range_of_m_l688_68818


namespace NUMINAMATH_GPT_number_of_persons_l688_68834

theorem number_of_persons (n : ℕ) (h : n * (n - 1) / 2 = 78) : n = 13 :=
sorry

end NUMINAMATH_GPT_number_of_persons_l688_68834


namespace NUMINAMATH_GPT_intersection_M_N_l688_68832

open Set

variable (x y : ℝ)

theorem intersection_M_N :
  let M := {x | x < 1}
  let N := {y | ∃ x, x < 1 ∧ y = 1 - 2 * x}
  M ∩ N = ∅ := sorry

end NUMINAMATH_GPT_intersection_M_N_l688_68832


namespace NUMINAMATH_GPT_problem1_problem2_l688_68891

-- Problem 1
theorem problem1 (a : ℝ) (h : a = Real.sqrt 3 - 1) : (a^2 + a) * (a + 1) / a = 3 := 
sorry

-- Problem 2
theorem problem2 (a : ℝ) (h : a = 1 / 2) : (a + 1) / (a^2 - 1) - (a + 1) / (1 - a) = -5 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l688_68891


namespace NUMINAMATH_GPT_incorrect_option_B_l688_68854

noncomputable def Sn : ℕ → ℝ := sorry
-- S_n is the sum of the first n terms of the arithmetic sequence

axiom S5_S6 : Sn 5 < Sn 6
axiom S6_eq_S_gt_S8 : Sn 6 = Sn 7 ∧ Sn 7 > Sn 8

theorem incorrect_option_B : ¬ (Sn 9 < Sn 5) := sorry

end NUMINAMATH_GPT_incorrect_option_B_l688_68854


namespace NUMINAMATH_GPT_diff_sum_even_odd_l688_68839

theorem diff_sum_even_odd (n : ℕ) (hn : n = 1500) :
  let sum_odd := n * (2 * n - 1)
  let sum_even := n * (2 * n + 1)
  sum_even - sum_odd = 1500 :=
by
  sorry

end NUMINAMATH_GPT_diff_sum_even_odd_l688_68839


namespace NUMINAMATH_GPT_monotonic_interval_a_l688_68844

theorem monotonic_interval_a (a : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → (2 * x - 2 * a) * (2 * 2 - 2 * a) ≥ 0 ∧ (2 * x - 2 * a) * (2 * 3 - 2 * a) ≥ 0) →
  a ≤ 2 ∨ a ≥ 3 := sorry

end NUMINAMATH_GPT_monotonic_interval_a_l688_68844


namespace NUMINAMATH_GPT_negation_of_proposition_l688_68806

open Real

theorem negation_of_proposition (P : ∀ x : ℝ, sin x ≥ 1) :
  ∃ x : ℝ, sin x < 1 :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l688_68806


namespace NUMINAMATH_GPT_general_formula_correct_sequence_T_max_term_l688_68884

open Classical

noncomputable def geometric_sequence_term (n : ℕ) : ℝ :=
  if h : n > 0 then (-1)^(n-1) * (3 / 2^n)
  else 0

noncomputable def geometric_sequence_sum (n : ℕ) : ℝ :=
  if h : n > 0 then 1 - (-1 / 2)^n
  else 0

noncomputable def sequence_T (n : ℕ) : ℝ :=
  geometric_sequence_sum n + 1 / geometric_sequence_sum n

theorem general_formula_correct :
  ∀ n : ℕ, n > 0 → geometric_sequence_term n = (-1)^(n-1) * (3 / 2^n) :=
sorry

theorem sequence_T_max_term :
  ∀ n : ℕ, n > 0 → sequence_T n ≤ sequence_T 1 ∧ sequence_T 1 = 13 / 6 :=
sorry

end NUMINAMATH_GPT_general_formula_correct_sequence_T_max_term_l688_68884


namespace NUMINAMATH_GPT_functional_equation_solution_l688_68822

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 * y) = f (x * y) + y * f (f x + y)) →
  (∀ y : ℝ, f y = 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l688_68822


namespace NUMINAMATH_GPT_divisibility_criterion_l688_68898

theorem divisibility_criterion (n : ℕ) : 
  (20^n - 13^n - 7^n) % 309 = 0 ↔ 
  ∃ k : ℕ, n = 1 + 6 * k ∨ n = 5 + 6 * k := 
  sorry

end NUMINAMATH_GPT_divisibility_criterion_l688_68898


namespace NUMINAMATH_GPT_solve_equations_l688_68840

theorem solve_equations :
  (∀ x : ℝ, x^2 - 2 * x - 15 = 0 ↔ x = 5 ∨ x = -3) ∧
  (∀ x : ℝ, 2 * x^2 + 3 * x - 1 = 0 ↔ x = (-3 + Real.sqrt 17) / 4 ∨ x = (-3 - Real.sqrt 17) / 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_equations_l688_68840


namespace NUMINAMATH_GPT_value_of_a_l688_68842

theorem value_of_a (a : ℝ) (A : Set ℝ) (hA : A = {a^2, 1}) (h : 3 ∈ A) : 
  a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l688_68842


namespace NUMINAMATH_GPT_third_class_males_eq_nineteen_l688_68837

def first_class_males : ℕ := 17
def first_class_females : ℕ := 13
def second_class_males : ℕ := 14
def second_class_females : ℕ := 18
def third_class_females : ℕ := 17
def students_unable_to_partner : ℕ := 2
def total_males_from_first_two_classes : ℕ := first_class_males + second_class_males
def total_females_from_first_two_classes : ℕ := first_class_females + second_class_females
def total_females : ℕ := total_females_from_first_two_classes + third_class_females

theorem third_class_males_eq_nineteen (M : ℕ) : 
  total_males_from_first_two_classes + M - (total_females + students_unable_to_partner) = 0 → M = 19 :=
by
  sorry

end NUMINAMATH_GPT_third_class_males_eq_nineteen_l688_68837


namespace NUMINAMATH_GPT_find_income_l688_68828

def income_and_savings (x : ℕ) : ℕ := 10 * x
def expenditure (x : ℕ) : ℕ := 4 * x
def savings (x : ℕ) : ℕ := income_and_savings x - expenditure x

theorem find_income (savings_eq : 6 * 1900 = 11400) : income_and_savings 1900 = 19000 :=
by
  sorry

end NUMINAMATH_GPT_find_income_l688_68828


namespace NUMINAMATH_GPT_field_area_l688_68802

theorem field_area (L W : ℝ) (h1: L = 20) (h2 : 2 * W + L = 41) : L * W = 210 :=
by
  sorry

end NUMINAMATH_GPT_field_area_l688_68802


namespace NUMINAMATH_GPT_calculate_y_l688_68847

theorem calculate_y (w x y : ℝ) (h1 : (7 / w) + (7 / x) = 7 / y) (h2 : w * x = y) (h3 : (w + x) / 2 = 0.5) : y = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_calculate_y_l688_68847


namespace NUMINAMATH_GPT_abc_relationship_l688_68861

variable (x y : ℝ)

def parabola (x : ℝ) : ℝ :=
  x^2 + x + 2

def a := parabola 2
def b := parabola (-1)
def c := parabola 3

theorem abc_relationship : c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_abc_relationship_l688_68861


namespace NUMINAMATH_GPT_find_a_plus_b_l688_68896

theorem find_a_plus_b (a b : ℝ) (x y : ℝ) 
  (h1 : x = 2) 
  (h2 : y = -1) 
  (h3 : a * x - 2 * y = 4) 
  (h4 : 3 * x + b * y = -7) : a + b = 14 := 
by 
  -- Begin the proof
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l688_68896


namespace NUMINAMATH_GPT_problem_statement_l688_68815

open Set

variable (U P Q : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5, 6}) (hP : P = {1, 2, 3, 4}) (hQ : Q = {3, 4, 5})

theorem problem_statement : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l688_68815


namespace NUMINAMATH_GPT_cylinder_side_surface_area_l688_68843

-- Define the given conditions
def base_circumference : ℝ := 4
def height_of_cylinder : ℝ := 4

-- Define the relation we need to prove
theorem cylinder_side_surface_area : 
  base_circumference * height_of_cylinder = 16 := 
by
  sorry

end NUMINAMATH_GPT_cylinder_side_surface_area_l688_68843


namespace NUMINAMATH_GPT_pencils_total_l688_68869

def pencils_remaining (Jeff_pencils_initial : ℕ) (Jeff_donation_percent : ℕ) 
                      (Vicki_factor : ℕ) (Vicki_donation_fraction_num : ℕ) 
                      (Vicki_donation_fraction_den : ℕ) : ℕ :=
  let Jeff_donated := Jeff_pencils_initial * Jeff_donation_percent / 100
  let Jeff_remaining := Jeff_pencils_initial - Jeff_donated
  let Vicki_pencils_initial := Vicki_factor * Jeff_pencils_initial
  let Vicki_donated := Vicki_pencils_initial * Vicki_donation_fraction_num / Vicki_donation_fraction_den
  let Vicki_remaining := Vicki_pencils_initial - Vicki_donated
  Jeff_remaining + Vicki_remaining

theorem pencils_total :
  pencils_remaining 300 30 2 3 4 = 360 :=
by
  -- The proof should be inserted here
  sorry

end NUMINAMATH_GPT_pencils_total_l688_68869


namespace NUMINAMATH_GPT_simple_interest_correct_l688_68883

def principal : ℝ := 400
def rate : ℝ := 0.20
def time : ℝ := 2

def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem simple_interest_correct :
  simple_interest principal rate time = 160 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_correct_l688_68883


namespace NUMINAMATH_GPT_reduction_in_jury_running_time_l688_68888

def week1_miles : ℕ := 2
def week2_miles : ℕ := 2 * week1_miles + 3
def week3_miles : ℕ := (9 * week2_miles) / 7
def week4_miles : ℕ := 4

theorem reduction_in_jury_running_time : week3_miles - week4_miles = 5 :=
by
  -- sorry specifies the proof is skipped
  sorry

end NUMINAMATH_GPT_reduction_in_jury_running_time_l688_68888


namespace NUMINAMATH_GPT_polynomial_unique_l688_68819

noncomputable def p (x : ℝ) : ℝ := x^2 + 1

theorem polynomial_unique (p : ℝ → ℝ) 
  (h1 : p 2 = 5) 
  (h2 : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) : 
  ∀ x : ℝ, p x = x^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_unique_l688_68819


namespace NUMINAMATH_GPT_smallest_multiple_of_2019_of_form_abcabcabc_l688_68880

def is_digit (n : ℕ) : Prop := n < 10

theorem smallest_multiple_of_2019_of_form_abcabcabc
    (a b c : ℕ)
    (h_a : is_digit a)
    (h_b : is_digit b)
    (h_c : is_digit c)
    (k : ℕ)
    (form : Nat)
    (rep: ℕ) : 
  (form = (a * 100 + b * 10 + c) * rep) →
  (∃ n : ℕ, form = 2019 * n) →
  form >= 673673673 :=
sorry

end NUMINAMATH_GPT_smallest_multiple_of_2019_of_form_abcabcabc_l688_68880


namespace NUMINAMATH_GPT_circumcircle_excircle_distance_squared_l688_68885

variable (R r_A d_A : ℝ)

theorem circumcircle_excircle_distance_squared 
  (h : R ≥ 0)
  (h1 : r_A ≥ 0)
  (h2 : d_A^2 = R^2 + 2 * R * r_A) : d_A^2 = R^2 + 2 * R * r_A := 
by
  sorry

end NUMINAMATH_GPT_circumcircle_excircle_distance_squared_l688_68885


namespace NUMINAMATH_GPT_inequality_transformation_l688_68897

theorem inequality_transformation (x : ℝ) :
  x - 2 > 1 → x > 3 :=
by
  intro h
  linarith

end NUMINAMATH_GPT_inequality_transformation_l688_68897


namespace NUMINAMATH_GPT_second_interest_rate_exists_l688_68899

theorem second_interest_rate_exists (X Y : ℝ) (H : 0 < X ∧ X ≤ 10000) : ∃ Y, 8 * X + Y * (10000 - X) = 85000 :=
by
  sorry

end NUMINAMATH_GPT_second_interest_rate_exists_l688_68899


namespace NUMINAMATH_GPT_inequality_proof_l688_68820

theorem inequality_proof (a b c : ℝ) (h1 : b < c) (h2 : 1 < a) (h3 : a < b + c) (h4 : b + c < a + 1) : b < a :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l688_68820


namespace NUMINAMATH_GPT_Loisa_saves_70_l688_68813

-- Define the conditions
def tablet_cost_cash := 450
def down_payment := 100
def payment_first_4_months := 40 * 4
def payment_next_4_months := 35 * 4
def payment_last_4_months := 30 * 4

-- Define the total installment payment
def total_installment_payment := down_payment + payment_first_4_months + payment_next_4_months + payment_last_4_months

-- Define the amount saved by paying cash instead of on installment
def amount_saved := total_installment_payment - tablet_cost_cash

-- The theorem to prove the savings amount
theorem Loisa_saves_70 : amount_saved = 70 := by
  -- Direct calculation or further proof steps here
  sorry

end NUMINAMATH_GPT_Loisa_saves_70_l688_68813


namespace NUMINAMATH_GPT_faye_candy_count_l688_68833

theorem faye_candy_count :
  let initial_candy := 47
  let candy_ate := 25
  let candy_given := 40
  initial_candy - candy_ate + candy_given = 62 :=
by
  let initial_candy := 47
  let candy_ate := 25
  let candy_given := 40
  sorry

end NUMINAMATH_GPT_faye_candy_count_l688_68833


namespace NUMINAMATH_GPT_f_monotonically_increasing_intervals_f_max_min_in_range_f_max_at_pi_over_3_f_min_at_neg_pi_over_12_l688_68872

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi / 6) + 1

theorem f_monotonically_increasing_intervals:
  ∀ (k : ℤ), ∀ x y, (-Real.pi / 6 + k * Real.pi) ≤ x ∧ x ≤ y ∧ y ≤ (k * Real.pi + Real.pi / 3) → f x ≤ f y :=
sorry

theorem f_max_min_in_range:
  ∀ x, (-Real.pi / 12) ≤ x ∧ x ≤ (5 * Real.pi / 12) → 
  (f x ≤ 2 ∧ f x ≥ -Real.sqrt 3) :=
sorry

theorem f_max_at_pi_over_3:
  f (Real.pi / 3) = 2 :=
sorry

theorem f_min_at_neg_pi_over_12:
  f (-Real.pi / 12) = -Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_f_monotonically_increasing_intervals_f_max_min_in_range_f_max_at_pi_over_3_f_min_at_neg_pi_over_12_l688_68872


namespace NUMINAMATH_GPT_incircle_hexagon_area_ratio_l688_68851

noncomputable def area_hexagon (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def radius_incircle (s : ℝ) : ℝ :=
  (s * Real.sqrt 3) / 2

noncomputable def area_incircle (r : ℝ) : ℝ :=
  Real.pi * r^2

noncomputable def area_ratio (s : ℝ) : ℝ :=
  let A_hexagon := area_hexagon s
  let r := radius_incircle s
  let A_incircle := area_incircle r
  A_incircle / A_hexagon

theorem incircle_hexagon_area_ratio (s : ℝ) (h : s = 1) :
  area_ratio s = (Real.pi * Real.sqrt 3) / 6 :=
by
  sorry

end NUMINAMATH_GPT_incircle_hexagon_area_ratio_l688_68851


namespace NUMINAMATH_GPT_johns_cloth_cost_per_metre_l688_68863

noncomputable def calculate_cost_per_metre (total_cost : ℝ) (total_metres : ℝ) : ℝ :=
  total_cost / total_metres

def johns_cloth_purchasing_data : Prop :=
  calculate_cost_per_metre 444 9.25 = 48

theorem johns_cloth_cost_per_metre : johns_cloth_purchasing_data :=
  sorry

end NUMINAMATH_GPT_johns_cloth_cost_per_metre_l688_68863


namespace NUMINAMATH_GPT_find_a_l688_68889

-- Definitions for the hyperbola and its eccentricity
def hyperbola_eq (a : ℝ) : Prop := a > 0 ∧ ∃ b : ℝ, b^2 = 3 ∧ ∃ e : ℝ, e = 2 ∧ 
  e = Real.sqrt (1 + b^2 / a^2)

-- The main theorem stating the value of 'a' given the conditions
theorem find_a (a : ℝ) (h : hyperbola_eq a) : a = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_a_l688_68889


namespace NUMINAMATH_GPT_correct_quotient_l688_68860

def original_number : ℕ :=
  8 * 156 + 2

theorem correct_quotient :
  (8 * 156 + 2) / 5 = 250 :=
sorry

end NUMINAMATH_GPT_correct_quotient_l688_68860


namespace NUMINAMATH_GPT_units_digit_of_result_l688_68858

theorem units_digit_of_result (a b c : ℕ) (h1 : a = c + 3) : 
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  let result := original - reversed
  result % 10 = 7 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_result_l688_68858


namespace NUMINAMATH_GPT_nesbitt_inequality_l688_68816

variable (a b c d : ℝ)

-- Assume a, b, c, d are positive real numbers
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom pos_d : 0 < d

theorem nesbitt_inequality :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by
  sorry

end NUMINAMATH_GPT_nesbitt_inequality_l688_68816


namespace NUMINAMATH_GPT_range_of_a_l688_68867

theorem range_of_a 
  (x1 x2 a : ℝ) 
  (h1 : x1 + x2 = 4) 
  (h2 : x1 * x2 = a) 
  (h3 : x1 > 1) 
  (h4 : x2 > 1) : 
  3 < a ∧ a ≤ 4 := 
sorry

end NUMINAMATH_GPT_range_of_a_l688_68867


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l688_68805

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
    ∃ d, ∀ n, a (n+1) = a n + d

-- The conditions
def condition_1 (a : ℕ → ℝ) : Prop :=
    (a 1 + a 2 + a 3 = 6)

def condition_2 (a : ℕ → ℝ) : Prop :=
    (a 10 + a 11 + a 12 = 9)

-- The Theorem statement
theorem sum_arithmetic_sequence :
    is_arithmetic_sequence a →
    condition_1 a →
    condition_2 a →
    (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 = 30) :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_l688_68805


namespace NUMINAMATH_GPT_percentage_cut_is_50_l688_68878

-- Conditions
def yearly_subscription_cost : ℝ := 940.0
def reduction_amount : ℝ := 470.0

-- Assertion to be proved
theorem percentage_cut_is_50 :
  (reduction_amount / yearly_subscription_cost) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_cut_is_50_l688_68878


namespace NUMINAMATH_GPT_lunch_break_duration_l688_68882

theorem lunch_break_duration :
  ∃ (L : ℝ), 
    (∃ (p a : ℝ),
      (6 - L) * (p + a) = 0.4 ∧
      (4 - L) * a = 0.15 ∧
      (10 - L) * p = 0.45) ∧
    291 = L * 60 := 
by
  sorry

end NUMINAMATH_GPT_lunch_break_duration_l688_68882


namespace NUMINAMATH_GPT_inscribed_circle_radius_l688_68809

theorem inscribed_circle_radius (a r : ℝ) (unit_square : a = 1)
  (touches_arc_AC : ∀ (x : ℝ × ℝ), x.1^2 + x.2^2 = (a - r)^2)
  (touches_arc_BD : ∀ (y : ℝ × ℝ), y.1^2 + y.2^2 = (a - r)^2)
  (touches_side_AB : ∀ (z : ℝ × ℝ), z.1 = r ∨ z.2 = r) :
  r = 3 / 8 := by sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l688_68809


namespace NUMINAMATH_GPT_length_of_train_l688_68814

variable (L : ℕ)

def speed_tree (L : ℕ) : ℚ := L / 120

def speed_platform (L : ℕ) : ℚ := (L + 500) / 160

theorem length_of_train
    (h1 : speed_tree L = speed_platform L)
    : L = 1500 :=
sorry

end NUMINAMATH_GPT_length_of_train_l688_68814


namespace NUMINAMATH_GPT_total_gift_amount_l688_68874

-- Definitions based on conditions
def workers_per_block := 200
def number_of_blocks := 15
def worth_of_each_gift := 2

-- The statement we need to prove
theorem total_gift_amount : workers_per_block * number_of_blocks * worth_of_each_gift = 6000 := by
  sorry

end NUMINAMATH_GPT_total_gift_amount_l688_68874


namespace NUMINAMATH_GPT_sally_pokemon_cards_count_l688_68870

-- Defining the initial conditions
def initial_cards : ℕ := 27
def cards_given_by_dan : ℕ := 41
def cards_bought_by_sally : ℕ := 20

-- Statement of the problem to be proved
theorem sally_pokemon_cards_count :
  initial_cards + cards_given_by_dan + cards_bought_by_sally = 88 := by
  sorry

end NUMINAMATH_GPT_sally_pokemon_cards_count_l688_68870


namespace NUMINAMATH_GPT_remainder_8_pow_310_mod_9_l688_68886

theorem remainder_8_pow_310_mod_9 : (8 ^ 310) % 9 = 8 := 
by
  sorry

end NUMINAMATH_GPT_remainder_8_pow_310_mod_9_l688_68886


namespace NUMINAMATH_GPT_same_color_eye_proportion_l688_68841

theorem same_color_eye_proportion :
  ∀ (a b c d e f : ℝ),
  a + b + c = 0.30 →
  a + d + e = 0.40 →
  b + d + f = 0.50 →
  a + b + c + d + e + f = 1 →
  c + e + f = 0.80 :=
by
  intros a b c d e f h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_same_color_eye_proportion_l688_68841


namespace NUMINAMATH_GPT_new_bottles_from_recycling_l688_68826

theorem new_bottles_from_recycling (initial_bottles : ℕ) (required_bottles : ℕ) (h : initial_bottles = 125) (r : required_bottles = 5) : 
∃ new_bottles : ℕ, new_bottles = (initial_bottles / required_bottles ^ 2 + initial_bottles / (required_bottles * required_bottles / required_bottles) + initial_bottles / (required_bottles * required_bottles * required_bottles / required_bottles * required_bottles * required_bottles)) :=
  sorry

end NUMINAMATH_GPT_new_bottles_from_recycling_l688_68826


namespace NUMINAMATH_GPT_blue_face_area_greater_than_red_face_area_l688_68881

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end NUMINAMATH_GPT_blue_face_area_greater_than_red_face_area_l688_68881


namespace NUMINAMATH_GPT_largest_decimal_of_4bit_binary_l688_68829

-- Define the maximum 4-bit binary number and its interpretation in base 10
def max_4bit_binary_value : ℕ := 2^4 - 1

-- The theorem to prove the statement
theorem largest_decimal_of_4bit_binary : max_4bit_binary_value = 15 :=
by
  -- Lean tactics or explicitly writing out the solution steps can be used here.
  -- Skipping proof as instructed.
  sorry

end NUMINAMATH_GPT_largest_decimal_of_4bit_binary_l688_68829


namespace NUMINAMATH_GPT_fgh_supermarkets_l688_68855

theorem fgh_supermarkets (U C : ℕ) 
  (h1 : U + C = 70) 
  (h2 : U = C + 14) : U = 42 :=
by
  sorry

end NUMINAMATH_GPT_fgh_supermarkets_l688_68855


namespace NUMINAMATH_GPT_erick_total_revenue_l688_68893

def lemon_price_increase := 4
def grape_price_increase := lemon_price_increase / 2
def original_lemon_price := 8
def original_grape_price := 7
def lemons_sold := 80
def grapes_sold := 140

def new_lemon_price := original_lemon_price + lemon_price_increase -- $12 per lemon
def new_grape_price := original_grape_price + grape_price_increase -- $9 per grape

def revenue_from_lemons := lemons_sold * new_lemon_price -- $960
def revenue_from_grapes := grapes_sold * new_grape_price -- $1260

def total_revenue := revenue_from_lemons + revenue_from_grapes

theorem erick_total_revenue : total_revenue = 2220 := by
  -- Skipping proof with sorry
  sorry

end NUMINAMATH_GPT_erick_total_revenue_l688_68893


namespace NUMINAMATH_GPT_simplify_fraction_l688_68857

theorem simplify_fraction :
  ( (2^1010)^2 - (2^1008)^2 ) / ( (2^1009)^2 - (2^1007)^2 ) = 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l688_68857


namespace NUMINAMATH_GPT_exinscribed_sphere_inequality_l688_68811

variable (r r_A r_B r_C r_D : ℝ)

theorem exinscribed_sphere_inequality 
  (hr : 0 < r) 
  (hrA : 0 < r_A) 
  (hrB : 0 < r_B) 
  (hrC : 0 < r_C) 
  (hrD : 0 < r_D) :
  1 / Real.sqrt (r_A^2 - r_A * r_B + r_B^2) +
  1 / Real.sqrt (r_B^2 - r_B * r_C + r_C^2) +
  1 / Real.sqrt (r_C^2 - r_C * r_D + r_D^2) +
  1 / Real.sqrt (r_D^2 - r_D * r_A + r_A^2) ≤
  2 / r := by
  sorry

end NUMINAMATH_GPT_exinscribed_sphere_inequality_l688_68811


namespace NUMINAMATH_GPT_food_cost_max_l688_68852

theorem food_cost_max (x : ℝ) (total_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (max_total : ℝ) (food_cost_max : ℝ) :
  total_cost = x * (1 + tax_rate + tip_rate) →
  tax_rate = 0.07 →
  tip_rate = 0.15 →
  max_total = 50 →
  total_cost ≤ max_total →
  food_cost_max = 50 / 1.22 →
  x ≤ food_cost_max :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_food_cost_max_l688_68852


namespace NUMINAMATH_GPT_house_number_units_digit_is_five_l688_68890

/-- Define the house number as a two-digit number -/
def is_two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- Define the properties for the statements -/
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_power_of_prime (n : ℕ) : Prop := ∃ p : ℕ, Nat.Prime p ∧ p ^ Nat.log p n = n
def is_divisible_by_five (n : ℕ) : Prop := n % 5 = 0
def has_digit_seven (n : ℕ) : Prop := (n / 10 = 7 ∨ n % 10 = 7)

/-- The theorem stating that the units digit of the house number is 5 -/
theorem house_number_units_digit_is_five (n : ℕ) 
  (h1 : is_two_digit_number n)
  (h2 : (is_prime n ∧ is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (¬is_prime n ∧ is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ ¬is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ is_power_of_prime n ∧ ¬is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ is_power_of_prime n ∧ is_divisible_by_five n ∧ ¬has_digit_seven n) ∨ 
        (¬is_prime n ∧ ¬is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (¬is_prime n ∧ is_power_of_prime n ∧ ¬is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ ¬is_power_of_prime n ∧ is_divisible_by_five n ∧ ¬has_digit_seven n))
  : n % 10 = 5 := 
sorry

end NUMINAMATH_GPT_house_number_units_digit_is_five_l688_68890


namespace NUMINAMATH_GPT_great_wall_scientific_notation_l688_68887

theorem great_wall_scientific_notation : 
  (21200000 : ℝ) = 2.12 * 10^7 :=
by
  sorry

end NUMINAMATH_GPT_great_wall_scientific_notation_l688_68887


namespace NUMINAMATH_GPT_other_coin_denomination_l688_68850

theorem other_coin_denomination :
  ∀ (total_coins : ℕ) (value_rs : ℕ) (paise_per_rs : ℕ) (num_20_paise_coins : ℕ) (total_value_paise : ℕ),
  total_coins = 324 →
  value_rs = 71 →
  paise_per_rs = 100 →
  num_20_paise_coins = 200 →
  total_value_paise = value_rs * paise_per_rs →
  (∃ (denom_other_coin : ℕ),
    total_value_paise - num_20_paise_coins * 20 = (total_coins - num_20_paise_coins) * denom_other_coin
    → denom_other_coin = 25) :=
by
  sorry

end NUMINAMATH_GPT_other_coin_denomination_l688_68850


namespace NUMINAMATH_GPT_find_a2_plus_b2_l688_68830

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a * b = -1) (h2 : a - b = 2) : a^2 + b^2 = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_a2_plus_b2_l688_68830


namespace NUMINAMATH_GPT_share_ratio_l688_68873

theorem share_ratio (A B C : ℝ) (x : ℝ) (h1 : A + B + C = 500) (h2 : A = 200) (h3 : A = x * (B + C)) (h4 : B = (6/9) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_share_ratio_l688_68873


namespace NUMINAMATH_GPT_total_week_cost_proof_l688_68865

-- Defining variables for costs and consumption
def cost_brand_a_biscuit : ℝ := 0.25
def cost_brand_b_biscuit : ℝ := 0.35
def cost_small_rawhide : ℝ := 1
def cost_large_rawhide : ℝ := 1.50

def odd_days_biscuits_brand_a : ℕ := 3
def odd_days_biscuits_brand_b : ℕ := 2
def odd_days_small_rawhide : ℕ := 1
def odd_days_large_rawhide : ℕ := 1

def even_days_biscuits_brand_a : ℕ := 4
def even_days_small_rawhide : ℕ := 2

def odd_day_cost : ℝ :=
  odd_days_biscuits_brand_a * cost_brand_a_biscuit +
  odd_days_biscuits_brand_b * cost_brand_b_biscuit +
  odd_days_small_rawhide * cost_small_rawhide +
  odd_days_large_rawhide * cost_large_rawhide

def even_day_cost : ℝ :=
  even_days_biscuits_brand_a * cost_brand_a_biscuit +
  even_days_small_rawhide * cost_small_rawhide

def total_cost_per_week : ℝ :=
  4 * odd_day_cost + 3 * even_day_cost

theorem total_week_cost_proof :
  total_cost_per_week = 24.80 :=
  by
    unfold total_cost_per_week
    unfold odd_day_cost
    unfold even_day_cost
    norm_num
    sorry

end NUMINAMATH_GPT_total_week_cost_proof_l688_68865


namespace NUMINAMATH_GPT_common_difference_in_arithmetic_sequence_l688_68864

theorem common_difference_in_arithmetic_sequence
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 2 = 3)
  (h2 : a 5 = 12) :
  d = 3 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_in_arithmetic_sequence_l688_68864


namespace NUMINAMATH_GPT_mean_equality_l688_68807

-- Define the mean calculation
def mean (a b c : ℕ) : ℚ := (a + b + c) / 3

-- The given conditions
theorem mean_equality (z : ℕ) (y : ℕ) (hz : z = 24) :
  mean 8 15 21 = mean 16 z y → y = 4 :=
by
  sorry

end NUMINAMATH_GPT_mean_equality_l688_68807


namespace NUMINAMATH_GPT_smallest_k_l688_68835

theorem smallest_k (a b c : ℤ) (k : ℤ) (h1 : a < b) (h2 : b < c) 
  (h3 : 2 * b = a + c) (h4 : (k * c) ^ 2 = a * b) (h5 : k > 1) : 
  c > 0 → k = 2 := 
sorry

end NUMINAMATH_GPT_smallest_k_l688_68835


namespace NUMINAMATH_GPT_dogs_eat_times_per_day_l688_68827

theorem dogs_eat_times_per_day (dogs : ℕ) (food_per_dog_per_meal : ℚ) (total_food : ℚ) 
                                (food_left : ℚ) (days : ℕ) 
                                (dogs_eat_times_per_day : ℚ)
                                (h_dogs : dogs = 3)
                                (h_food_per_dog_per_meal : food_per_dog_per_meal = 1 / 2)
                                (h_total_food : total_food = 30)
                                (h_food_left : food_left = 9)
                                (h_days : days = 7) :
                                dogs_eat_times_per_day = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_dogs_eat_times_per_day_l688_68827


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l688_68859

def upstream_speed := 34 -- in kmph
def downstream_speed := 48 -- in kmph

def speed_in_still_water := (upstream_speed + downstream_speed) / 2

theorem speed_of_man_in_still_water :
  speed_in_still_water = 41 := by
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l688_68859


namespace NUMINAMATH_GPT_count_solutions_sin_equation_l688_68849

theorem count_solutions_sin_equation : 
  ∃ S : Finset ℝ, (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 3 * (Real.sin x)^4 - 7 * (Real.sin x)^3 + 5 * (Real.sin x)^2 - Real.sin x = 0) ∧ S.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_count_solutions_sin_equation_l688_68849


namespace NUMINAMATH_GPT_jason_initial_cards_l688_68804

theorem jason_initial_cards (cards_sold : Nat) (cards_after_selling : Nat) (initial_cards : Nat) 
  (h1 : cards_sold = 224) 
  (h2 : cards_after_selling = 452) 
  (h3 : initial_cards = cards_after_selling + cards_sold) : 
  initial_cards = 676 := 
sorry

end NUMINAMATH_GPT_jason_initial_cards_l688_68804


namespace NUMINAMATH_GPT_rodney_lifting_capacity_l688_68879

theorem rodney_lifting_capacity 
  (R O N : ℕ)
  (h1 : R + O + N = 239)
  (h2 : R = 2 * O)
  (h3 : O = 4 * N - 7) : 
  R = 146 := 
by
  sorry

end NUMINAMATH_GPT_rodney_lifting_capacity_l688_68879


namespace NUMINAMATH_GPT_system_equations_sum_14_l688_68808

theorem system_equations_sum_14 (a b c d : ℝ) 
  (h1 : a + c = 4) 
  (h2 : a * d + b * c = 5) 
  (h3 : a * c + b + d = 8) 
  (h4 : b * d = 1) :
  a + b + c + d = 7 ∨ a + b + c + d = 7 → (a + b + c + d) * 2 = 14 := 
by {
  sorry
}

end NUMINAMATH_GPT_system_equations_sum_14_l688_68808


namespace NUMINAMATH_GPT_num_pens_multiple_of_16_l688_68823

theorem num_pens_multiple_of_16 (Pencils Students : ℕ) (h1 : Pencils = 928) (h2 : Students = 16)
  (h3 : ∃ (Pn : ℕ), Pencils = Pn * Students) :
  ∃ (k : ℕ), ∃ (Pens : ℕ), Pens = 16 * k :=
by
  sorry

end NUMINAMATH_GPT_num_pens_multiple_of_16_l688_68823


namespace NUMINAMATH_GPT_initial_numbers_conditions_l688_68877

theorem initial_numbers_conditions (a b c : ℤ)
    (h : ∀ (x y z : ℤ), (x, y, z) = (17, 1967, 1983) → 
      x = y + z - 1 ∨ y = x + z - 1 ∨ z = x + y - 1) :
  (a = 2 ∧ b = 2 ∧ c = 2) → false ∧ 
  (a = 3 ∧ b = 3 ∧ c = 3) → true := 
sorry

end NUMINAMATH_GPT_initial_numbers_conditions_l688_68877


namespace NUMINAMATH_GPT_problem1_l688_68871

theorem problem1 (a b c : ℝ) (h : a * c + b * c + c^2 < 0) : b^2 > 4 * a * c := sorry

end NUMINAMATH_GPT_problem1_l688_68871


namespace NUMINAMATH_GPT_unique_solution_k_l688_68875

theorem unique_solution_k (k : ℕ) (f : ℕ → ℕ) :
  (∀ n : ℕ, (Nat.iterate f n n) = n + k) → k = 0 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_k_l688_68875
