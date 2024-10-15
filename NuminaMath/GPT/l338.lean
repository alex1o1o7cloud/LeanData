import Mathlib

namespace NUMINAMATH_GPT_coursework_materials_spending_l338_33822

def budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

theorem coursework_materials_spending : 
    budget - (budget * food_percentage + budget * accommodation_percentage + budget * entertainment_percentage) = 300 := 
by 
  -- steps you would use to prove this
  sorry

end NUMINAMATH_GPT_coursework_materials_spending_l338_33822


namespace NUMINAMATH_GPT_circle_center_radius_l338_33839

open Real

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 6*x = 0 ↔ (x - 3)^2 + y^2 = 9 :=
by sorry

end NUMINAMATH_GPT_circle_center_radius_l338_33839


namespace NUMINAMATH_GPT_estimate_larger_than_difference_l338_33808

theorem estimate_larger_than_difference (x y z : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : z > 0) :
    (x + z) - (y - z) > x - y :=
    sorry

end NUMINAMATH_GPT_estimate_larger_than_difference_l338_33808


namespace NUMINAMATH_GPT_smaller_circle_radius_l338_33812

-- Given conditions
def larger_circle_radius : ℝ := 10
def number_of_smaller_circles : ℕ := 7

-- The goal
theorem smaller_circle_radius :
  ∃ r : ℝ, (∃ D : ℝ, D = 2 * larger_circle_radius ∧ D = 4 * r) ∧ r = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_smaller_circle_radius_l338_33812


namespace NUMINAMATH_GPT_sum_digits_500_l338_33815

noncomputable def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_500 (k : ℕ) (h : k = 55) :
  sum_digits (63 * 10^k - 64) = 500 :=
by
  sorry

end NUMINAMATH_GPT_sum_digits_500_l338_33815


namespace NUMINAMATH_GPT_find_number_l338_33832

theorem find_number (x : ℝ) : (30 / 100) * x = (60 / 100) * 150 + 120 ↔ x = 700 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l338_33832


namespace NUMINAMATH_GPT_complement_intersection_l338_33861

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection :
  (U \ A) ∩ B = {0} :=
  by
    sorry

end NUMINAMATH_GPT_complement_intersection_l338_33861


namespace NUMINAMATH_GPT_trigonometric_identity_l338_33895

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l338_33895


namespace NUMINAMATH_GPT_total_consultation_time_l338_33801

-- Define the times in which each chief finishes a pipe
def chief1_time := 10
def chief2_time := 30
def chief3_time := 60

theorem total_consultation_time : 
  ∃ (t : ℕ), (∃ x, ((x / chief1_time) + (x / chief2_time) + (x / chief3_time) = 1) ∧ t = 3 * x) ∧ t = 20 :=
sorry

end NUMINAMATH_GPT_total_consultation_time_l338_33801


namespace NUMINAMATH_GPT_kristin_annual_income_l338_33891

theorem kristin_annual_income (p : ℝ) :
  ∃ A : ℝ, 
  (0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = (0.01 * (p + 0.25) * A)) ∧
  A = 32000 :=
by
  sorry

end NUMINAMATH_GPT_kristin_annual_income_l338_33891


namespace NUMINAMATH_GPT_fraction_zero_iff_numerator_zero_l338_33896

variable (x : ℝ)

def numerator (x : ℝ) : ℝ := x - 5
def denominator (x : ℝ) : ℝ := 6 * x + 12

theorem fraction_zero_iff_numerator_zero (h_denominator_nonzero : denominator 5 ≠ 0) : 
  numerator x / denominator x = 0 ↔ x = 5 :=
by sorry

end NUMINAMATH_GPT_fraction_zero_iff_numerator_zero_l338_33896


namespace NUMINAMATH_GPT_common_ratio_geometric_progression_l338_33858

theorem common_ratio_geometric_progression {x y z r : ℝ} (h_diff1 : x ≠ y) (h_diff2 : y ≠ z) (h_diff3 : z ≠ x)
  (hx_nonzero : x ≠ 0) (hy_nonzero : y ≠ 0) (hz_nonzero : z ≠ 0)
  (h_gm_progression : ∃ r : ℝ, x * (y - z) = x * (y - z) * r ∧ z * (x - y) = (y * (z - x)) * r) : r^2 + r + 1 = 0 :=
sorry

end NUMINAMATH_GPT_common_ratio_geometric_progression_l338_33858


namespace NUMINAMATH_GPT_total_education_duration_l338_33821

-- Definitions from the conditions
def high_school_duration : ℕ := 4 - 1
def tertiary_education_duration : ℕ := 3 * high_school_duration

-- The theorem statement
theorem total_education_duration : high_school_duration + tertiary_education_duration = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_education_duration_l338_33821


namespace NUMINAMATH_GPT_stops_away_pinedale_mall_from_yahya_house_l338_33876

-- Definitions based on problem conditions
def bus_speed_kmh : ℕ := 60
def stop_interval_minutes : ℕ := 5
def distance_to_mall_km : ℕ := 40

-- Definition of how many stops away is Pinedale mall from Yahya's house
def stops_to_mall : ℕ := distance_to_mall_km / (bus_speed_kmh / 60 * stop_interval_minutes)

-- Lean statement to prove the given conditions lead to the correct number of stops
theorem stops_away_pinedale_mall_from_yahya_house :
  stops_to_mall = 8 :=
by 
  -- This is a placeholder for the proof. 
  -- Actual proof steps would convert units and calculate as described in the problem.
  sorry

end NUMINAMATH_GPT_stops_away_pinedale_mall_from_yahya_house_l338_33876


namespace NUMINAMATH_GPT_find_a_l338_33800

noncomputable def has_exactly_one_solution_in_x (a : ℝ) : Prop :=
  ∀ x : ℝ, |x^2 + 2*a*x + a + 5| = 3 → x = -a

theorem find_a (a : ℝ) : has_exactly_one_solution_in_x a ↔ (a = 4 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l338_33800


namespace NUMINAMATH_GPT_rancher_monetary_loss_l338_33804

def rancher_head_of_cattle := 500
def market_rate_per_head := 700
def sick_cattle := 350
def additional_cost_per_sick_animal := 80
def reduced_price_per_head := 450

def expected_revenue := rancher_head_of_cattle * market_rate_per_head
def loss_from_death := sick_cattle * market_rate_per_head
def additional_sick_cost := sick_cattle * additional_cost_per_sick_animal
def remaining_cattle := rancher_head_of_cattle - sick_cattle
def revenue_from_remaining_cattle := remaining_cattle * reduced_price_per_head

def total_loss := (expected_revenue - revenue_from_remaining_cattle) + additional_sick_cost

theorem rancher_monetary_loss : total_loss = 310500 := by
  sorry

end NUMINAMATH_GPT_rancher_monetary_loss_l338_33804


namespace NUMINAMATH_GPT_unit_prices_purchase_plans_exchange_methods_l338_33883

theorem unit_prices (x r : ℝ) (hx : r = 2 * x) 
  (h_eq : (40/(2*r)) + 4 = 30/x) : 
  x = 2.5 ∧ r = 5 := sorry

theorem purchase_plans (x r : ℝ) (a b : ℕ)
  (hx : x = 2.5) (hr : r = 5) (h_eq : x * a + r * b = 200)
  (h_ge_20 : 20 ≤ a ∧ 20 ≤ b) (h_mult_10 : a % 10 = 0) :
  (a, b) = (20, 30) ∨ (a, b) = (30, 25) ∨ (a, b) = (40, 20) := sorry

theorem exchange_methods (a b t m : ℕ) 
  (hx : x = 2.5) (hr : r = 5) 
  (h_leq : 1 < m ∧ m < 10) 
  (h_eq : a + 2 * t = b + (m - t))
  (h_planA : (a = 20 ∧ b = 30) ∨ (a = 30 ∧ b = 25) ∨ (a = 40 ∧ b = 20)) :
  (m = 5 ∧ t = 5 ∧ b = 30) ∨
  (m = 8 ∧ t = 6 ∧ b = 25) ∨
  (m = 5 ∧ t = 0 ∧ b = 25) ∨
  (m = 8 ∧ t = 1 ∧ b = 20) := sorry

end NUMINAMATH_GPT_unit_prices_purchase_plans_exchange_methods_l338_33883


namespace NUMINAMATH_GPT_good_numbers_correct_l338_33886

noncomputable def good_numbers (n : ℕ) : ℝ :=
  1 / 2 * (8^n + 10^n) - 1

theorem good_numbers_correct (n : ℕ) : good_numbers n = 
  1 / 2 * (8^n + 10^n) - 1 := 
sorry

end NUMINAMATH_GPT_good_numbers_correct_l338_33886


namespace NUMINAMATH_GPT_perimeter_of_rectangle_l338_33879

theorem perimeter_of_rectangle (L W : ℝ) (h1 : L / W = 5 / 2) (h2 : L * W = 4000) : 2 * L + 2 * W = 280 :=
sorry

end NUMINAMATH_GPT_perimeter_of_rectangle_l338_33879


namespace NUMINAMATH_GPT_sqrt_ax3_eq_negx_sqrt_ax_l338_33871

variable (a x : ℝ)
variable (ha : a < 0) (hx : x < 0)

theorem sqrt_ax3_eq_negx_sqrt_ax : Real.sqrt (a * x^3) = -x * Real.sqrt (a * x) := by
  sorry

end NUMINAMATH_GPT_sqrt_ax3_eq_negx_sqrt_ax_l338_33871


namespace NUMINAMATH_GPT_find_A_l338_33850

-- Given a three-digit number AB2 such that AB2 - 41 = 591
def valid_number (A B : ℕ) : Prop :=
  (A * 100) + (B * 10) + 2 - 41 = 591

-- We aim to prove that A = 6 given B = 2
theorem find_A (A : ℕ) (B : ℕ) (hB : B = 2) : A = 6 :=
  by
  have h : valid_number A B := by sorry
  sorry

end NUMINAMATH_GPT_find_A_l338_33850


namespace NUMINAMATH_GPT_find_function_l338_33889

theorem find_function (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (x + y) = f x + f y - 2023) →
  ∃ c : ℤ, ∀ x : ℤ, f x = c * x + 2023 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_function_l338_33889


namespace NUMINAMATH_GPT_cassie_has_8_parrots_l338_33870

-- Define the conditions
def num_dogs : ℕ := 4
def nails_per_foot : ℕ := 4
def feet_per_dog : ℕ := 4
def nails_per_dog := nails_per_foot * feet_per_dog

def nails_total_dogs : ℕ := num_dogs * nails_per_dog

def claws_per_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def normal_claws_per_parrot := claws_per_leg * legs_per_parrot

def extra_toe_parrot_claws : ℕ := normal_claws_per_parrot + 1

def total_nails : ℕ := 113

-- Establishing the proof problem
theorem cassie_has_8_parrots : 
  ∃ (P : ℕ), (6 * (P - 1) + 7 = 49) ∧ P = 8 := by
  sorry

end NUMINAMATH_GPT_cassie_has_8_parrots_l338_33870


namespace NUMINAMATH_GPT_solution_set_l338_33854

noncomputable def satisfies_equations (x y : ℝ) : Prop :=
  (x^2 + 3 * x * y = 12) ∧ (x * y = 16 + y^2 - x * y - x^2)

theorem solution_set :
  {p : ℝ × ℝ | satisfies_equations p.1 p.2} = {(4, 1), (-4, -1), (-4, 1), (4, -1)} :=
by sorry

end NUMINAMATH_GPT_solution_set_l338_33854


namespace NUMINAMATH_GPT_rahul_share_of_payment_l338_33875

-- Definitions
def rahulWorkDays : ℕ := 3
def rajeshWorkDays : ℕ := 2
def totalPayment : ℤ := 355

-- Theorem statement
theorem rahul_share_of_payment :
  let rahulWorkRate := 1 / (rahulWorkDays : ℝ)
  let rajeshWorkRate := 1 / (rajeshWorkDays : ℝ)
  let combinedWorkRate := rahulWorkRate + rajeshWorkRate
  let rahulShareRatio := rahulWorkRate / combinedWorkRate
  let rahulShare := (totalPayment : ℝ) * rahulShareRatio
  rahulShare = 142 :=
by
  sorry

end NUMINAMATH_GPT_rahul_share_of_payment_l338_33875


namespace NUMINAMATH_GPT_arithmetic_sum_l338_33840

theorem arithmetic_sum :
  ∀ (a : ℕ → ℝ),
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →
  (∃ x : ℝ, ∃ y : ℝ, x^2 - 6 * x - 1 = 0 ∧ y^2 - 6 * y - 1 = 0 ∧ x = a 3 ∧ y = a 15) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by
  intros a h_arith_seq h_roots
  sorry

end NUMINAMATH_GPT_arithmetic_sum_l338_33840


namespace NUMINAMATH_GPT_find_three_tuple_solutions_l338_33864

open Real

theorem find_three_tuple_solutions :
  (x y z : ℝ) → (x^2 + y^2 + 25 * z^2 = 6 * x * z + 8 * y * z)
  → (3 * x^2 + 2 * y^2 + z^2 = 240)
  → (x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2) :=
by
  intro x y z
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_three_tuple_solutions_l338_33864


namespace NUMINAMATH_GPT_find_s_for_g_l338_33863

def g (x : ℝ) (s : ℝ) : ℝ := 3*x^4 - 2*x^3 + 2*x^2 + x + s

theorem find_s_for_g (s : ℝ) : g (-1) s = 0 ↔ s = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_s_for_g_l338_33863


namespace NUMINAMATH_GPT_fixed_monthly_fee_l338_33848

theorem fixed_monthly_fee (x y : ℝ) 
  (h1 : x + 20 * y = 15.20) 
  (h2 : x + 40 * y = 25.20) : 
  x = 5.20 := 
sorry

end NUMINAMATH_GPT_fixed_monthly_fee_l338_33848


namespace NUMINAMATH_GPT_intersection_points_lie_on_circle_l338_33897

variables (u x y : ℝ)

theorem intersection_points_lie_on_circle :
  (∃ u : ℝ, 3 * u - 4 * y + 2 = 0 ∧ 2 * x - 3 * u * y - 4 = 0) →
  ∃ r : ℝ, (x^2 + y^2 = r^2) :=
by 
  sorry

end NUMINAMATH_GPT_intersection_points_lie_on_circle_l338_33897


namespace NUMINAMATH_GPT_gear_angular_speed_proportion_l338_33892

theorem gear_angular_speed_proportion :
  ∀ (ω_A ω_B ω_C ω_D k: ℝ),
    30 * ω_A = k →
    45 * ω_B = k →
    50 * ω_C = k →
    60 * ω_D = k →
    ω_A / ω_B = 1 ∧
    ω_B / ω_C = 45 / 50 ∧
    ω_C / ω_D = 50 / 60 ∧
    ω_A / ω_D = 10 / 7.5 :=
  by
    -- proof goes here
    sorry

end NUMINAMATH_GPT_gear_angular_speed_proportion_l338_33892


namespace NUMINAMATH_GPT_prop_A_l338_33803

theorem prop_A (x : ℝ) (h : x > 1) : (x + (1 / (x - 1)) >= 3) :=
sorry

end NUMINAMATH_GPT_prop_A_l338_33803


namespace NUMINAMATH_GPT_problem1_problem2_l338_33894

-- Problem 1
theorem problem1 : ∀ x : ℝ, 4 * x - 3 * (20 - x) + 4 = 0 → x = 8 :=
by
  intro x
  intro h
  sorry

-- Problem 2
theorem problem2 : ∀ x : ℝ, (2 * x + 1) / 3 = 1 - (x - 1) / 5 → x = 1 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_problem1_problem2_l338_33894


namespace NUMINAMATH_GPT_length_of_other_train_is_correct_l338_33838

noncomputable def length_of_second_train 
  (length_first_train : ℝ) 
  (speed_first_train : ℝ) 
  (speed_second_train : ℝ) 
  (time_to_cross : ℝ) 
  : ℝ := 
  let speed_first_train_m_s := speed_first_train * (1000 / 3600)
  let speed_second_train_m_s := speed_second_train * (1000 / 3600)
  let relative_speed := speed_first_train_m_s + speed_second_train_m_s
  let total_distance := relative_speed * time_to_cross
  total_distance - length_first_train

theorem length_of_other_train_is_correct :
  length_of_second_train 250 120 80 9 = 249.95 :=
by
  unfold length_of_second_train
  simp
  sorry

end NUMINAMATH_GPT_length_of_other_train_is_correct_l338_33838


namespace NUMINAMATH_GPT_vertical_bisecting_line_of_circles_l338_33881

theorem vertical_bisecting_line_of_circles :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 2 * x + 6 * y + 2 = 0 ∨ x^2 + y^2 + 4 * x - 2 * y - 4 = 0) →
  (4 * x + 3 * y + 5 = 0) :=
sorry

end NUMINAMATH_GPT_vertical_bisecting_line_of_circles_l338_33881


namespace NUMINAMATH_GPT_sum_of_series_is_correct_l338_33853

noncomputable def geometric_series_sum_5_terms : ℚ :=
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  a * (1 - r^n) / (1 - r)

theorem sum_of_series_is_correct :
  geometric_series_sum_5_terms = 1023 / 3072 := by
  sorry

end NUMINAMATH_GPT_sum_of_series_is_correct_l338_33853


namespace NUMINAMATH_GPT_calc_expression_l338_33865

theorem calc_expression : (3^500 + 4^501)^2 - (3^500 - 4^501)^2 = 16 * 10^500 :=
  sorry

end NUMINAMATH_GPT_calc_expression_l338_33865


namespace NUMINAMATH_GPT_convert_polar_to_rectangular_example_l338_33873

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular_example :
  polar_to_rectangular 6 (5 * Real.pi / 2) = (0, 6) := by
  sorry

end NUMINAMATH_GPT_convert_polar_to_rectangular_example_l338_33873


namespace NUMINAMATH_GPT_average_percentage_for_all_students_l338_33807

-- Definitions of the variables
def students1 : Nat := 15
def average1 : Nat := 75
def students2 : Nat := 10
def average2 : Nat := 90
def total_students : Nat := students1 + students2
def total_percentage1 : Nat := students1 * average1
def total_percentage2 : Nat := students2 * average2
def total_percentage : Nat := total_percentage1 + total_percentage2

-- Main theorem stating the average percentage for all students.
theorem average_percentage_for_all_students :
  total_percentage / total_students = 81 := by
  sorry

end NUMINAMATH_GPT_average_percentage_for_all_students_l338_33807


namespace NUMINAMATH_GPT_solution_set_l338_33867

theorem solution_set (x y : ℝ) : 
  x^5 - 10 * x^3 * y^2 + 5 * x * y^4 = 0 ↔ 
  x = 0 
  ∨ y = x / Real.sqrt (5 + 2 * Real.sqrt 5) 
  ∨ y = x / Real.sqrt (5 - 2 * Real.sqrt 5) 
  ∨ y = -x / Real.sqrt (5 + 2 * Real.sqrt 5) 
  ∨ y = -x / Real.sqrt (5 - 2 * Real.sqrt 5) := 
by
  sorry

end NUMINAMATH_GPT_solution_set_l338_33867


namespace NUMINAMATH_GPT_max_trees_cut_l338_33826

theorem max_trees_cut (n : ℕ) (h : n = 2001) :
  (∃ m : ℕ, m = n * n ∧ ∀ (x y : ℕ), x < n ∧ y < n → (x % 2 = 0 ∧ y % 2 = 0 → m = 1001001)) := sorry

end NUMINAMATH_GPT_max_trees_cut_l338_33826


namespace NUMINAMATH_GPT_general_term_seq_l338_33805

universe u

-- Define the sequence
def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) = 2 * a n + 1)

-- State the theorem
theorem general_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n, a n = 2^n - 1 :=
by
  sorry

end NUMINAMATH_GPT_general_term_seq_l338_33805


namespace NUMINAMATH_GPT_tree_difference_l338_33855

-- Given constants
def Hassans_apple_trees : Nat := 1
def Hassans_orange_trees : Nat := 2

def Ahmeds_orange_trees : Nat := 8
def Ahmeds_apple_trees : Nat := 4 * Hassans_apple_trees

-- Total trees computations
def Ahmeds_total_trees : Nat := Ahmeds_apple_trees + Ahmeds_orange_trees
def Hassans_total_trees : Nat := Hassans_apple_trees + Hassans_orange_trees

-- Theorem to prove the difference in total trees
theorem tree_difference : Ahmeds_total_trees - Hassans_total_trees = 9 := by
  sorry

end NUMINAMATH_GPT_tree_difference_l338_33855


namespace NUMINAMATH_GPT_four_digit_numbers_sum_30_l338_33810

-- Definitions of the variables and constraints
def valid_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

-- The main statement we aim to prove
theorem four_digit_numbers_sum_30 : 
  ∃ (count : ℕ), 
  count = 20 ∧ 
  ∃ (a b c d : ℕ), 
  (1 ≤ a ∧ valid_digit a) ∧ 
  (valid_digit b) ∧ 
  (valid_digit c) ∧ 
  (valid_digit d) ∧ 
  a + b + c + d = 30 := sorry

end NUMINAMATH_GPT_four_digit_numbers_sum_30_l338_33810


namespace NUMINAMATH_GPT_problem1_l338_33809

theorem problem1 (A B C : Prop) : (A ∨ (B ∧ C)) ↔ ((A ∨ B) ∧ (A ∨ C)) :=
sorry 

end NUMINAMATH_GPT_problem1_l338_33809


namespace NUMINAMATH_GPT_factor_polynomial_l338_33846

theorem factor_polynomial : ∀ x : ℝ, 
  x^8 - 4 * x^6 + 6 * x^4 - 4 * x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factor_polynomial_l338_33846


namespace NUMINAMATH_GPT_find_z_l338_33868

theorem find_z (x y z : ℝ) (h : 1 / (x + 1) + 1 / (y + 1) = 1 / z) :
  z = (x + 1) * (y + 1) / (x + y + 2) :=
sorry

end NUMINAMATH_GPT_find_z_l338_33868


namespace NUMINAMATH_GPT_smallest_possible_sum_l338_33860

theorem smallest_possible_sum (E F G H : ℕ) (h1 : F > 0) (h2 : E + F + G = 3 * F) (h3 : F * G = 4 * F * F / 3) :
  E = 6 ∧ F = 9 ∧ G = 12 ∧ H = 16 ∧ E + F + G + H = 43 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_possible_sum_l338_33860


namespace NUMINAMATH_GPT_tetrahedron_edge_length_l338_33825

-- Define the problem as a Lean theorem statement
theorem tetrahedron_edge_length (r : ℝ) (a : ℝ) (h : r = 1) :
  a = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_tetrahedron_edge_length_l338_33825


namespace NUMINAMATH_GPT_relationship_among_abc_l338_33843

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := (2 / 3) ^ (2 / 5)
noncomputable def c : ℝ := Real.log (1 / 5) / Real.log (1 / 3)

theorem relationship_among_abc : c > b ∧ b > a :=
by
  have h1 : a = (1 / 3) ^ (2 / 5) := rfl
  have h2 : b = (2 / 3) ^ (2 / 5) := rfl
  have h3 : c = Real.log (1 / 5) / Real.log (1 / 3) := rfl
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l338_33843


namespace NUMINAMATH_GPT_mixed_oil_rate_l338_33828

theorem mixed_oil_rate :
  let v₁ := 10
  let p₁ := 50
  let v₂ := 5
  let p₂ := 68
  let v₃ := 8
  let p₃ := 42
  let v₄ := 7
  let p₄ := 62
  let v₅ := 12
  let p₅ := 55
  let v₆ := 6
  let p₆ := 75
  let total_cost := v₁ * p₁ + v₂ * p₂ + v₃ * p₃ + v₄ * p₄ + v₅ * p₅ + v₆ * p₆
  let total_volume := v₁ + v₂ + v₃ + v₄ + v₅ + v₆
  let rate := total_cost / total_volume
  rate = 56.67 :=
by
  sorry

end NUMINAMATH_GPT_mixed_oil_rate_l338_33828


namespace NUMINAMATH_GPT_sum_original_and_correct_value_l338_33802

theorem sum_original_and_correct_value (x : ℕ) (h : x + 14 = 68) :
  x + (x + 41) = 149 := by
  sorry

end NUMINAMATH_GPT_sum_original_and_correct_value_l338_33802


namespace NUMINAMATH_GPT_two_point_four_times_eight_point_two_l338_33834

theorem two_point_four_times_eight_point_two (x y z : ℝ) (hx : x = 2.4) (hy : y = 8.2) (hz : z = 4.8 + 5.2) :
  x * y * z = 2.4 * 8.2 * 10 ∧ abs (x * y * z - 200) < abs (x * y * z - 150) ∧
  abs (x * y * z - 200) < abs (x * y * z - 250) ∧
  abs (x * y * z - 200) < abs (x * y * z - 300) ∧
  abs (x * y * z - 200) < abs (x * y * z - 350) := by
  sorry

end NUMINAMATH_GPT_two_point_four_times_eight_point_two_l338_33834


namespace NUMINAMATH_GPT_min_perimeter_l338_33856

theorem min_perimeter (a b : ℕ) (h1 : b = 3 * a) (h2 : 3 * a + 8 * a = 11) (h3 : 2 * a + 12 * a = 14)
  : 2 * (15 + 11) = 52 := 
sorry

end NUMINAMATH_GPT_min_perimeter_l338_33856


namespace NUMINAMATH_GPT_at_least_one_tails_up_l338_33824

-- Define propositions p and q
variable (p q : Prop)

-- The theorem statement
theorem at_least_one_tails_up : (¬p ∨ ¬q) ↔ ¬(p ∧ q) := by
  sorry

end NUMINAMATH_GPT_at_least_one_tails_up_l338_33824


namespace NUMINAMATH_GPT_minimize_J_l338_33888

noncomputable def H (p q : ℝ) : ℝ := -3 * p * q + 4 * p * (1 - q) + 2 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ := max (H p 0) (H p 1)

theorem minimize_J : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ (∀ p' : ℝ, 0 ≤ p' ∧ p' ≤ 1 → J p ≤ J p') ∧ p = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimize_J_l338_33888


namespace NUMINAMATH_GPT_minimum_value_of_u_l338_33851

noncomputable def minimum_value_lemma (x y : ℝ) (hx : Real.sin x + Real.sin y = 1 / 3) : Prop :=
  ∃ m : ℝ, m = -1/9 ∧ ∀ z, z = Real.sin x + Real.cos x ^ 2 → z ≥ m

theorem minimum_value_of_u
  (x y : ℝ)
  (hx : Real.sin x + Real.sin y = 1 / 3) :
  ∃ m : ℝ, m = -1/9 ∧ ∀ z, z = Real.sin x + Real.cos x ^ 2 → z ≥ m :=
sorry

end NUMINAMATH_GPT_minimum_value_of_u_l338_33851


namespace NUMINAMATH_GPT_incorrect_conclusion_l338_33844

theorem incorrect_conclusion (b x : ℂ) (h : x^2 - b * x + 1 = 0) : x = 1 ∨ x = -1
  ↔ (b = 2 ∨ b = -2) :=
by sorry

end NUMINAMATH_GPT_incorrect_conclusion_l338_33844


namespace NUMINAMATH_GPT_complement_A_l338_33829

-- Definitions for the conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 1}

-- Proof statement
theorem complement_A : (U \ A) = {x | x ≥ 1} := by
  sorry

end NUMINAMATH_GPT_complement_A_l338_33829


namespace NUMINAMATH_GPT_no_point_satisfies_both_systems_l338_33872

theorem no_point_satisfies_both_systems (x y : ℝ) :
  (y < 3 ∧ x - y < 3 ∧ x + y < 4) ∧
  ((y - 3) * (x - y - 3) ≥ 0 ∧ (y - 3) * (x + y - 4) ≤ 0 ∧ (x - y - 3) * (x + y - 4) ≤ 0)
  → false :=
sorry

end NUMINAMATH_GPT_no_point_satisfies_both_systems_l338_33872


namespace NUMINAMATH_GPT_find_abc_l338_33899

open Real

noncomputable def abc_value (a b c : ℝ) : ℝ := a * b * c

theorem find_abc (a b c : ℝ)
  (h₁ : a - b = 3)
  (h₂ : a^2 + b^2 = 39)
  (h₃ : a + b + c = 10) :
  abc_value a b c = -150 + 15 * Real.sqrt 69 :=
by
  sorry

end NUMINAMATH_GPT_find_abc_l338_33899


namespace NUMINAMATH_GPT_sum_of_coordinates_l338_33820

-- Define the conditions for m and n
def m : ℤ := -3
def n : ℤ := 2

-- State the proposition based on the conditions
theorem sum_of_coordinates : m + n = -1 := 
by 
  -- Provide an incomplete proof skeleton with "sorry" to skip the proof
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_l338_33820


namespace NUMINAMATH_GPT_license_plate_combinations_l338_33893

def num_choices_two_repeat_letters : ℕ :=
  (Nat.choose 26 2) * (Nat.choose 4 2) * (5 * 4)

theorem license_plate_combinations : num_choices_two_repeat_letters = 39000 := by
  sorry

end NUMINAMATH_GPT_license_plate_combinations_l338_33893


namespace NUMINAMATH_GPT_distance_A_to_C_through_B_l338_33836

-- Define the distances on the map
def Distance_AB_map : ℝ := 20
def Distance_BC_map : ℝ := 10

-- Define the scale of the map
def scale : ℝ := 5

-- Define the actual distances
def Distance_AB := Distance_AB_map * scale
def Distance_BC := Distance_BC_map * scale

-- Define the total distance from A to C through B
def Distance_AC_through_B := Distance_AB + Distance_BC

-- Theorem to be proved
theorem distance_A_to_C_through_B : Distance_AC_through_B = 150 := by
  sorry

end NUMINAMATH_GPT_distance_A_to_C_through_B_l338_33836


namespace NUMINAMATH_GPT_revenue_increase_l338_33882

theorem revenue_increase (n : ℕ) (C P : ℝ) 
  (h1 : n * P = 1.20 * C) : 
  (0.95 * n * P) = 1.14 * C :=
by
  sorry

end NUMINAMATH_GPT_revenue_increase_l338_33882


namespace NUMINAMATH_GPT_helga_ratio_l338_33816

variable (a b c d : ℕ)

def helga_shopping (a b c d total_shoes pairs_first_three : ℕ) : Prop :=
  a = 7 ∧
  b = a + 2 ∧
  c = 0 ∧
  a + b + c + d = total_shoes ∧
  pairs_first_three = a + b + c ∧
  total_shoes = 48 ∧
  (d : ℚ) / (pairs_first_three : ℚ) = 2

theorem helga_ratio : helga_shopping 7 9 0 32 48 16 := by
  sorry

end NUMINAMATH_GPT_helga_ratio_l338_33816


namespace NUMINAMATH_GPT_range_of_a_l338_33849

-- Definitions for propositions
def p (a : ℝ) : Prop :=
  (1 - 4 * (a^2 - 6 * a) > 0) ∧ (a^2 - 6 * a < 0)

def q (a : ℝ) : Prop :=
  (a - 3)^2 - 4 ≥ 0

-- Proof statement
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ (a ≤ 0 ∨ 1 < a ∧ a < 5 ∨ a ≥ 6) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l338_33849


namespace NUMINAMATH_GPT_no_real_solutions_l338_33818

theorem no_real_solutions :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 4 → ¬(3 * x^2 - 15 * x) / (x^2 - 4 * x) = x - 2) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_l338_33818


namespace NUMINAMATH_GPT_cannot_move_reach_goal_l338_33847

structure Point :=
(x : ℤ)
(y : ℤ)

def area (p1 p2 p3 : Point) : ℚ :=
  (1 / 2 : ℚ) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

noncomputable def isTriangleAreaPreserved (initPos finalPos : Point) (helper1Init helper1Final helper2Init helper2Final : Point) : Prop :=
  area initPos helper1Init helper2Init = area finalPos helper1Final helper2Final

theorem cannot_move_reach_goal :
  ¬ ∃ (r₀ r₁ : Point) (a₀ a₁ : Point) (s₀ s₁ : Point),
    r₀ = ⟨0, 0⟩ ∧ r₁ = ⟨2, 2⟩ ∧
    a₀ = ⟨0, 1⟩ ∧ a₁ = ⟨0, 1⟩ ∧
    s₀ = ⟨1, 0⟩ ∧ s₁ = ⟨1, 0⟩ ∧
    isTriangleAreaPreserved r₀ r₁ a₀ a₁ s₀ s₁ :=
by sorry

end NUMINAMATH_GPT_cannot_move_reach_goal_l338_33847


namespace NUMINAMATH_GPT_gcf_54_81_l338_33823

theorem gcf_54_81 : Nat.gcd 54 81 = 27 :=
by sorry

end NUMINAMATH_GPT_gcf_54_81_l338_33823


namespace NUMINAMATH_GPT_scientific_notation_14nm_l338_33866

theorem scientific_notation_14nm :
  0.000000014 = 1.4 * 10^(-8) := 
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_14nm_l338_33866


namespace NUMINAMATH_GPT_k_values_for_perpendicular_lines_l338_33837

-- Definition of perpendicular condition for lines
def perpendicular_lines (k : ℝ) : Prop :=
  k * (k - 1) + (1 - k) * (2 * k + 3) = 0

-- Lean 4 statement representing the math proof problem
theorem k_values_for_perpendicular_lines (k : ℝ) :
  perpendicular_lines k ↔ k = -3 ∨ k = 1 :=
by
  sorry

end NUMINAMATH_GPT_k_values_for_perpendicular_lines_l338_33837


namespace NUMINAMATH_GPT_james_weekly_pistachio_cost_l338_33811

def cost_per_can : ℕ := 10
def ounces_per_can : ℕ := 5
def consumption_per_5_days : ℕ := 30
def days_per_week : ℕ := 7

theorem james_weekly_pistachio_cost : (days_per_week / 5 * consumption_per_5_days) / ounces_per_can * cost_per_can = 90 := 
by
  sorry

end NUMINAMATH_GPT_james_weekly_pistachio_cost_l338_33811


namespace NUMINAMATH_GPT_jen_total_birds_l338_33890

theorem jen_total_birds (C D G : ℕ) (h1 : D = 150) (h2 : D = 4 * C + 10) (h3 : G = (D + C) / 2) :
  D + C + G = 277 := sorry

end NUMINAMATH_GPT_jen_total_birds_l338_33890


namespace NUMINAMATH_GPT_central_angle_of_sector_l338_33845

theorem central_angle_of_sector (P : ℝ) (x : ℝ) (h : P = 1 / 8) : x = 45 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l338_33845


namespace NUMINAMATH_GPT_volume_of_box_l338_33884

noncomputable def volume_expression (y : ℝ) : ℝ :=
  (15 - 2 * y) * (12 - 2 * y) * y

theorem volume_of_box (y : ℝ) :
  volume_expression y = 4 * y^3 - 54 * y^2 + 180 * y :=
by
  sorry

end NUMINAMATH_GPT_volume_of_box_l338_33884


namespace NUMINAMATH_GPT_total_apples_bought_l338_33859

def apples_bought_by_Junhyeok := 7 * 16
def apples_bought_by_Jihyun := 6 * 25

theorem total_apples_bought : apples_bought_by_Junhyeok + apples_bought_by_Jihyun = 262 := by
  sorry

end NUMINAMATH_GPT_total_apples_bought_l338_33859


namespace NUMINAMATH_GPT_loss_percentage_initial_selling_l338_33880

theorem loss_percentage_initial_selling (CP SP' : ℝ) 
  (hCP : CP = 1250) 
  (hSP' : SP' = CP * 1.15) 
  (h_diff : SP' - 500 = 937.5) : 
  (CP - 937.5) / CP * 100 = 25 := 
by 
  sorry

end NUMINAMATH_GPT_loss_percentage_initial_selling_l338_33880


namespace NUMINAMATH_GPT_max_earnings_l338_33835

section MaryEarnings

def regular_rate : ℝ := 10
def first_period_hours : ℕ := 40
def second_period_hours : ℕ := 10
def third_period_hours : ℕ := 10
def weekend_days : ℕ := 2
def weekend_bonus_per_day : ℝ := 50
def bonus_threshold_hours : ℕ := 55
def overtime_multiplier_second_period : ℝ := 0.25
def overtime_multiplier_third_period : ℝ := 0.5
def milestone_bonus : ℝ := 100

def regular_pay := regular_rate * first_period_hours
def second_period_pay := (regular_rate * (1 + overtime_multiplier_second_period)) * second_period_hours
def third_period_pay := (regular_rate * (1 + overtime_multiplier_third_period)) * third_period_hours
def weekend_bonus := weekend_days * weekend_bonus_per_day
def milestone_pay := milestone_bonus

def total_earnings := regular_pay + second_period_pay + third_period_pay + weekend_bonus + milestone_pay

theorem max_earnings : total_earnings = 875 := by
  sorry

end MaryEarnings

end NUMINAMATH_GPT_max_earnings_l338_33835


namespace NUMINAMATH_GPT_student_A_recruit_as_pilot_exactly_one_student_pass_l338_33862

noncomputable def student_A_recruit_prob : ℝ :=
  1 * 0.5 * 0.6 * 1

theorem student_A_recruit_as_pilot :
  student_A_recruit_prob = 0.3 :=
by
  sorry

noncomputable def one_student_pass_reinspection : ℝ :=
  0.5 * (1 - 0.6) * (1 - 0.75) +
  (1 - 0.5) * 0.6 * (1 - 0.75) +
  (1 - 0.5) * (1 - 0.6) * 0.75

theorem exactly_one_student_pass :
  one_student_pass_reinspection = 0.275 :=
by
  sorry

end NUMINAMATH_GPT_student_A_recruit_as_pilot_exactly_one_student_pass_l338_33862


namespace NUMINAMATH_GPT_length_of_one_side_of_regular_pentagon_l338_33819

-- Define the conditions
def is_regular_pentagon (P : ℝ) (n : ℕ) : Prop := n = 5 ∧ P = 23.4

-- State the theorem
theorem length_of_one_side_of_regular_pentagon (P : ℝ) (n : ℕ) 
  (h : is_regular_pentagon P n) : P / n = 4.68 :=
by
  sorry

end NUMINAMATH_GPT_length_of_one_side_of_regular_pentagon_l338_33819


namespace NUMINAMATH_GPT_number_of_unsold_items_l338_33817

theorem number_of_unsold_items (v k : ℕ) (hv : v ≤ 53) (havg_int : ∃ n : ℕ, k = n * v)
  (hk_eq : k = 130*v - 1595) 
  (hnew_avg : (k + 2505) / (v + 7) = 130) :
  60 - (v + 7) = 24 :=
by
  sorry

end NUMINAMATH_GPT_number_of_unsold_items_l338_33817


namespace NUMINAMATH_GPT_symmetric_points_sum_l338_33885

theorem symmetric_points_sum (n m : ℤ) 
  (h₁ : (3 : ℤ) = m)
  (h₂ : n = (-5 : ℤ)) : 
  m + n = (-2 : ℤ) := 
by 
  sorry

end NUMINAMATH_GPT_symmetric_points_sum_l338_33885


namespace NUMINAMATH_GPT_anna_has_4_twenty_cent_coins_l338_33852

theorem anna_has_4_twenty_cent_coins (x y : ℕ) (h1 : x + y = 15) (h2 : 59 - 3 * x = 24) : y = 4 :=
by {
  -- evidence based on the established conditions would be derived here
  sorry
}

end NUMINAMATH_GPT_anna_has_4_twenty_cent_coins_l338_33852


namespace NUMINAMATH_GPT_find_particular_number_l338_33898

theorem find_particular_number (x : ℤ) (h : ((x / 23) - 67) * 2 = 102) : x = 2714 := 
by 
  sorry

end NUMINAMATH_GPT_find_particular_number_l338_33898


namespace NUMINAMATH_GPT_concave_number_probability_l338_33813

/-- Definition of a concave number -/
def is_concave_number (a b c : ℕ) : Prop :=
  a > b ∧ c > b

/-- Set of possible digits -/
def digits : Finset ℕ := {4, 5, 6, 7, 8}

 /-- Total number of distinct three-digit combinations -/
def total_combinations : ℕ := 60

 /-- Number of concave numbers -/
def concave_numbers : ℕ := 20

 /-- Probability that a randomly chosen three-digit number is a concave number -/
def probability_concave : ℚ := concave_numbers / total_combinations

theorem concave_number_probability :
  probability_concave = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_concave_number_probability_l338_33813


namespace NUMINAMATH_GPT_imaginary_part_of_complex_l338_33827

theorem imaginary_part_of_complex :
  let i := Complex.I
  let z := 10 * i / (3 + i)
  z.im = 3 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_complex_l338_33827


namespace NUMINAMATH_GPT_angle_between_adjacent_triangles_l338_33842

-- Define the setup of the problem
def five_nonoverlapping_equilateral_triangles (angles : Fin 5 → ℝ) :=
  ∀ i, angles i = 60

def angles_between_adjacent_triangles (angles : Fin 5 → ℝ) :=
  ∀ i j, i ≠ j → angles i = angles j

-- State the main theorem
theorem angle_between_adjacent_triangles :
  ∀ (angles : Fin 5 → ℝ),
    five_nonoverlapping_equilateral_triangles angles →
    angles_between_adjacent_triangles angles →
    ((360 - 5 * 60) / 5) = 12 :=
by
  intros angles h1 h2
  sorry

end NUMINAMATH_GPT_angle_between_adjacent_triangles_l338_33842


namespace NUMINAMATH_GPT_total_carrots_grown_l338_33869

theorem total_carrots_grown
  (Sandy_carrots : ℕ) (Sam_carrots : ℕ) (Sophie_carrots : ℕ) (Sara_carrots : ℕ)
  (h1 : Sandy_carrots = 6)
  (h2 : Sam_carrots = 3)
  (h3 : Sophie_carrots = 2 * Sam_carrots)
  (h4 : Sara_carrots = (Sandy_carrots + Sam_carrots + Sophie_carrots) - 5) :
  Sandy_carrots + Sam_carrots + Sophie_carrots + Sara_carrots = 25 :=
by sorry

end NUMINAMATH_GPT_total_carrots_grown_l338_33869


namespace NUMINAMATH_GPT_range_of_a_l338_33814

theorem range_of_a {a : ℝ} :
  (∃ (x y : ℝ), (x - a)^2 + (y - a)^2 = 4 ∧ x^2 + y^2 = 4) ↔ (-2*Real.sqrt 2 < a ∧ a < 2*Real.sqrt 2 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_GPT_range_of_a_l338_33814


namespace NUMINAMATH_GPT_Cindy_initial_marbles_l338_33830

theorem Cindy_initial_marbles (M : ℕ) 
  (h1 : 4 * (M - 320) = 720) : M = 500 :=
by
  sorry

end NUMINAMATH_GPT_Cindy_initial_marbles_l338_33830


namespace NUMINAMATH_GPT_solution_to_diameter_area_problem_l338_33857

def diameter_area_problem : Prop :=
  let radius := 4
  let area_of_shaded_region := 16 + 8 * Real.pi
  -- Definitions derived directly from conditions
  let circle_radius := radius
  let diameter1_perpendicular_to_diameter2 := True
  -- Conclusively prove the area of the shaded region
  ∀ (PQ RS : ℝ) (h1 : PQ = 2 * circle_radius) (h2 : RS = 2 * circle_radius) (h3 : diameter1_perpendicular_to_diameter2),
  ∃ (area : ℝ), area = area_of_shaded_region

-- This is just the statement, the proof part is omitted.
theorem solution_to_diameter_area_problem : diameter_area_problem :=
  sorry

end NUMINAMATH_GPT_solution_to_diameter_area_problem_l338_33857


namespace NUMINAMATH_GPT_length_of_second_train_l338_33841

theorem length_of_second_train 
  (length_first_train : ℝ) 
  (speed_first_train_kmph : ℝ) 
  (speed_second_train_kmph : ℝ) 
  (time_to_cross : ℝ) 
  (h1 : length_first_train = 400)
  (h2 : speed_first_train_kmph = 72)
  (h3 : speed_second_train_kmph = 36)
  (h4 : time_to_cross = 69.99440044796417) :
  let speed_first_train := speed_first_train_kmph * (1000 / 3600)
  let speed_second_train := speed_second_train_kmph * (1000 / 3600)
  let relative_speed := speed_first_train - speed_second_train
  let distance := relative_speed * time_to_cross
  let length_second_train := distance - length_first_train
  length_second_train = 299.9440044796417 :=
  by
    sorry

end NUMINAMATH_GPT_length_of_second_train_l338_33841


namespace NUMINAMATH_GPT_Mehki_is_10_years_older_than_Jordyn_l338_33831

def Zrinka_age : Nat := 6
def Mehki_age : Nat := 22
def Jordyn_age : Nat := 2 * Zrinka_age

theorem Mehki_is_10_years_older_than_Jordyn : Mehki_age - Jordyn_age = 10 :=
by
  sorry

end NUMINAMATH_GPT_Mehki_is_10_years_older_than_Jordyn_l338_33831


namespace NUMINAMATH_GPT_find_given_number_l338_33878

theorem find_given_number (x : ℕ) : 10 * x + 2 = 3 * (x + 200000) → x = 85714 :=
by
  sorry

end NUMINAMATH_GPT_find_given_number_l338_33878


namespace NUMINAMATH_GPT_remainder_is_4_l338_33874

-- Definitions based on the given conditions
def dividend := 132
def divisor := 16
def quotient := 8

-- The theorem we aim to prove, stating the remainder
theorem remainder_is_4 : dividend = divisor * quotient + 4 := sorry

end NUMINAMATH_GPT_remainder_is_4_l338_33874


namespace NUMINAMATH_GPT_pyramid_volume_l338_33877

noncomputable def volume_of_pyramid 
  (ABCD : Type) 
  (rectangle : ABCD) 
  (DM_perpendicular : Prop) 
  (MA MC MB : ℕ) 
  (lengths : MA = 11 ∧ MC = 13 ∧ MB = 15 ∧ DM = 5) : ℝ :=
  80 * Real.sqrt 6

theorem pyramid_volume (ABCD : Type) 
    (rectangle : ABCD) 
    (DM_perpendicular : Prop) 
    (MA MC MB DM : ℕ)
    (lengths : MA = 11 ∧ MC = 13 ∧ MB = 15 ∧ DM = 5) 
  : volume_of_pyramid ABCD rectangle DM_perpendicular MA MC MB lengths = 80 * Real.sqrt 6 :=
  by {
    sorry
  }

end NUMINAMATH_GPT_pyramid_volume_l338_33877


namespace NUMINAMATH_GPT_work_problem_l338_33833

theorem work_problem 
  (A_work_time : ℤ) 
  (B_work_time : ℤ) 
  (x : ℤ)
  (A_work_rate : ℚ := 1 / 15 )
  (work_left : ℚ := 0.18333333333333335)
  (worked_together_for : ℚ := 7)
  (work_done : ℚ := 1 - work_left) :
  (7 * (1 / 15 + 1 / x) = work_done) → x = 20 :=
by sorry

end NUMINAMATH_GPT_work_problem_l338_33833


namespace NUMINAMATH_GPT_smallest_positive_integer_x_l338_33806

theorem smallest_positive_integer_x (x : ℕ) (h900 : ∃ a b c : ℕ, 900 = (2^a) * (3^b) * (5^c) ∧ a = 2 ∧ b = 2 ∧ c = 2) (h1152 : ∃ a b : ℕ, 1152 = (2^a) * (3^b) ∧ a = 7 ∧ b = 2) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_x_l338_33806


namespace NUMINAMATH_GPT_equation_solution_l338_33887

theorem equation_solution (x : ℝ) (h : x + 1/x = 2.5) : x^2 + 1/x^2 = 4.25 := 
by sorry

end NUMINAMATH_GPT_equation_solution_l338_33887
