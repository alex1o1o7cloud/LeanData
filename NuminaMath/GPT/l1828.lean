import Mathlib

namespace NUMINAMATH_GPT_abs_eq_solutions_l1828_182851

theorem abs_eq_solutions (x : ℝ) (hx : |x - 5| = 3 * x + 6) :
  x = -11 / 2 ∨ x = -1 / 4 :=
sorry

end NUMINAMATH_GPT_abs_eq_solutions_l1828_182851


namespace NUMINAMATH_GPT_marble_weight_l1828_182833

-- Define the weights of marbles and waffle irons
variables (m w : ℝ)

-- Given conditions
def condition1 : Prop := 9 * m = 4 * w
def condition2 : Prop := 3 * w = 75 

-- The theorem we want to prove
theorem marble_weight (h1 : condition1 m w) (h2 : condition2 w) : m = 100 / 9 :=
by
  sorry

end NUMINAMATH_GPT_marble_weight_l1828_182833


namespace NUMINAMATH_GPT_manny_marbles_l1828_182825

theorem manny_marbles (total_marbles : ℕ) (ratio_m : ℕ) (ratio_n : ℕ) (manny_gives : ℕ) 
  (h_total : total_marbles = 36) (h_ratio_m : ratio_m = 4) (h_ratio_n : ratio_n = 5) (h_manny_gives : manny_gives = 2) : 
  (total_marbles * ratio_n / (ratio_m + ratio_n)) - manny_gives = 18 :=
by
  sorry

end NUMINAMATH_GPT_manny_marbles_l1828_182825


namespace NUMINAMATH_GPT_A_share_of_gain_l1828_182830

-- Given problem conditions
def investment_A (x : ℝ) : ℝ := x * 12
def investment_B (x : ℝ) : ℝ := 2 * x * 6
def investment_C (x : ℝ) : ℝ := 3 * x * 4
def total_investment (x : ℝ) : ℝ := investment_A x + investment_B x + investment_C x
def total_gain : ℝ := 21000

-- Mathematically equivalent proof problem statement
theorem A_share_of_gain (x : ℝ) : (investment_A x) / (total_investment x) * total_gain = 7000 :=
by
  sorry

end NUMINAMATH_GPT_A_share_of_gain_l1828_182830


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1828_182890

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 5 * x = 4) ↔ (∃ x : ℝ, x^2 + 5 * x ≠ 4) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1828_182890


namespace NUMINAMATH_GPT_double_inequality_l1828_182870

variable (a b c : ℝ)

theorem double_inequality 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b ≤ 1) (hbc : b + c ≤ 1) (hca : c + a ≤ 1) :
  a^2 + b^2 + c^2 ≤ a + b + c - a * b - b * c - c * a ∧ 
  a + b + c - a * b - b * c - c * a ≤ 1 / 2 * (1 + a^2 + b^2 + c^2) := 
sorry

end NUMINAMATH_GPT_double_inequality_l1828_182870


namespace NUMINAMATH_GPT_problem_statement_l1828_182803

theorem problem_statement (x y : ℝ) : (x * y < 18) → (x < 2 ∨ y < 9) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1828_182803


namespace NUMINAMATH_GPT_inscribed_circle_radius_l1828_182850

theorem inscribed_circle_radius (AB BC CD DA: ℝ) (hAB: AB = 13) (hBC: BC = 10) (hCD: CD = 8) (hDA: DA = 11) :
  ∃ r, r = 2 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l1828_182850


namespace NUMINAMATH_GPT_monotonic_m_range_l1828_182824

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x - 12

-- Prove the range of m where f(x) is monotonic on [m, m+4]
theorem monotonic_m_range {m : ℝ} :
  (∀ x y : ℝ, m ≤ x ∧ x ≤ m + 4 ∧ m ≤ y ∧ y ≤ m + 4 → (x ≤ y → f x ≤ f y ∨ f x ≥ f y))
  ↔ (m ≤ -5 ∨ m ≥ 2) :=
sorry

end NUMINAMATH_GPT_monotonic_m_range_l1828_182824


namespace NUMINAMATH_GPT_abs_neg_three_l1828_182802

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end NUMINAMATH_GPT_abs_neg_three_l1828_182802


namespace NUMINAMATH_GPT_find_m_value_l1828_182856

noncomputable def fx (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

theorem find_m_value (m : ℝ) : (∀ x > 0, fx m x > fx m 0) → m = 2 := by
  sorry

end NUMINAMATH_GPT_find_m_value_l1828_182856


namespace NUMINAMATH_GPT_power_inequality_l1828_182813

theorem power_inequality (n : ℕ) (x : ℝ) (h1 : 0 < n) (h2 : x > -1) : (1 + x)^n ≥ 1 + n * x :=
sorry

end NUMINAMATH_GPT_power_inequality_l1828_182813


namespace NUMINAMATH_GPT_inverse_function_value_l1828_182893

theorem inverse_function_value (f : ℝ → ℝ) (h : ∀ y : ℝ, f (3^y) = y) : f 3 = 1 :=
sorry

end NUMINAMATH_GPT_inverse_function_value_l1828_182893


namespace NUMINAMATH_GPT_rain_at_least_once_l1828_182826

theorem rain_at_least_once (p : ℚ) (h : p = 3/4) : 
    (1 - (1 - p)^4) = 255/256 :=
by
  sorry

end NUMINAMATH_GPT_rain_at_least_once_l1828_182826


namespace NUMINAMATH_GPT_find_x_l1828_182808

theorem find_x (a b x : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < x)
    (h4 : (4 * a)^(4 * b) = a^b * x^(2 * b)) : 
    x = 16 * a^(3 / 2) :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l1828_182808


namespace NUMINAMATH_GPT_total_distance_walked_l1828_182809

-- Condition 1: Distance in feet
def distance_feet : ℝ := 30

-- Condition 2: Conversion factor from feet to meters
def feet_to_meters : ℝ := 0.3048

-- Condition 3: Number of trips
def trips : ℝ := 4

-- Question: Total distance walked in meters
theorem total_distance_walked :
  distance_feet * feet_to_meters * trips = 36.576 :=
sorry

end NUMINAMATH_GPT_total_distance_walked_l1828_182809


namespace NUMINAMATH_GPT_expand_expression_l1828_182828

theorem expand_expression :
  (3 * t^2 - 2 * t + 3) * (-2 * t^2 + 3 * t - 4) = -6 * t^4 + 13 * t^3 - 24 * t^2 + 17 * t - 12 :=
by sorry

end NUMINAMATH_GPT_expand_expression_l1828_182828


namespace NUMINAMATH_GPT_John_height_in_feet_after_growth_spurt_l1828_182836

def John_initial_height : ℕ := 66
def growth_rate_per_month : ℕ := 2
def number_of_months : ℕ := 3
def inches_per_foot : ℕ := 12

theorem John_height_in_feet_after_growth_spurt :
  (John_initial_height + growth_rate_per_month * number_of_months) / inches_per_foot = 6 := by
  sorry

end NUMINAMATH_GPT_John_height_in_feet_after_growth_spurt_l1828_182836


namespace NUMINAMATH_GPT_arithmetic_sequence_y_solution_l1828_182876

theorem arithmetic_sequence_y_solution : 
  ∃ y : ℚ, (y + 2 - - (1 / 3)) = (4 * y - (y + 2)) ∧ y = 13 / 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_y_solution_l1828_182876


namespace NUMINAMATH_GPT_inscribed_circle_probability_l1828_182853

theorem inscribed_circle_probability (r : ℝ) (h : r > 0) : 
  let square_area := 4 * r^2
  let circle_area := π * r^2
  (circle_area / square_area) = π / 4 := by
  sorry

end NUMINAMATH_GPT_inscribed_circle_probability_l1828_182853


namespace NUMINAMATH_GPT_find_x_floor_mult_eq_45_l1828_182869

theorem find_x_floor_mult_eq_45 (x : ℝ) (h1 : 0 < x) (h2 : ⌊x⌋ * x = 45) : x = 7.5 :=
sorry

end NUMINAMATH_GPT_find_x_floor_mult_eq_45_l1828_182869


namespace NUMINAMATH_GPT_remainder_123456789012_mod_252_l1828_182839

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end NUMINAMATH_GPT_remainder_123456789012_mod_252_l1828_182839


namespace NUMINAMATH_GPT_floor_width_l1828_182817

theorem floor_width
  (widthX lengthX : ℝ) (widthY lengthY : ℝ)
  (hX : widthX = 10) (lX : lengthX = 18) (lY : lengthY = 20)
  (h : lengthX * widthX = lengthY * widthY) :
  widthY = 9 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_floor_width_l1828_182817


namespace NUMINAMATH_GPT_average_paychecks_l1828_182827

def first_paychecks : Nat := 6
def remaining_paychecks : Nat := 20
def total_paychecks : Nat := 26
def amount_first : Nat := 750
def amount_remaining : Nat := 770

theorem average_paychecks : 
  (first_paychecks * amount_first + remaining_paychecks * amount_remaining) / total_paychecks = 765 :=
by
  sorry

end NUMINAMATH_GPT_average_paychecks_l1828_182827


namespace NUMINAMATH_GPT_women_per_table_l1828_182835

theorem women_per_table 
  (total_tables : ℕ)
  (men_per_table : ℕ)
  (total_customers : ℕ) 
  (h_total_tables : total_tables = 6)
  (h_men_per_table : men_per_table = 5)
  (h_total_customers : total_customers = 48) :
  (total_customers - (men_per_table * total_tables)) / total_tables = 3 :=
by
  subst h_total_tables
  subst h_men_per_table
  subst h_total_customers
  sorry

end NUMINAMATH_GPT_women_per_table_l1828_182835


namespace NUMINAMATH_GPT_profit_percentage_l1828_182883

theorem profit_percentage (CP SP : ℝ) (h₁ : CP = 400) (h₂ : SP = 560) : 
  ((SP - CP) / CP) * 100 = 40 := by 
  sorry

end NUMINAMATH_GPT_profit_percentage_l1828_182883


namespace NUMINAMATH_GPT_fraction_compare_l1828_182878

theorem fraction_compare : 
  let a := (1 : ℝ) / 4
  let b := 250000025 / (10^9)
  let diff := a - b
  diff = (1 : ℝ) / (4 * 10^7) :=
by
  sorry

end NUMINAMATH_GPT_fraction_compare_l1828_182878


namespace NUMINAMATH_GPT_min_score_needed_l1828_182855

theorem min_score_needed 
  (s1 s2 s3 s4 s5 : ℕ)
  (next_test_goal_increment : ℕ)
  (current_scores_sum : ℕ)
  (desired_average : ℕ)
  (total_tests : ℕ)
  (required_total_sum : ℕ)
  (required_next_score : ℕ)
  (current_scores : s1 = 88 ∧ s2 = 92 ∧ s3 = 75 ∧ s4 = 85 ∧ s5 = 80)
  (increment_eq : next_test_goal_increment = 5)
  (current_sum_eq : current_scores_sum = s1 + s2 + s3 + s4 + s5)
  (desired_average_eq : desired_average = (current_scores_sum / 5) + next_test_goal_increment)
  (total_tests_eq : total_tests = 6)
  (required_total_sum_eq : required_total_sum = desired_average * total_tests)
  (required_next_score_eq : required_next_score = required_total_sum - current_scores_sum) :
  required_next_score = 114 := by
    sorry

end NUMINAMATH_GPT_min_score_needed_l1828_182855


namespace NUMINAMATH_GPT_solve_for_x_l1828_182842

theorem solve_for_x 
  (x : ℝ) 
  (h : (2/7) * (1/4) * x = 8) : 
  x = 112 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1828_182842


namespace NUMINAMATH_GPT_james_painted_area_l1828_182875

-- Define the dimensions of the wall and windows
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window1_height : ℕ := 3
def window1_length : ℕ := 5
def window2_height : ℕ := 2
def window2_length : ℕ := 6

-- Calculate the areas
def wall_area : ℕ := wall_height * wall_length
def window1_area : ℕ := window1_height * window1_length
def window2_area : ℕ := window2_height * window2_length
def total_window_area : ℕ := window1_area + window2_area
def painted_area : ℕ := wall_area - total_window_area

theorem james_painted_area : painted_area = 123 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_james_painted_area_l1828_182875


namespace NUMINAMATH_GPT_gcd_18_30_45_l1828_182837

-- Define the conditions
def a := 18
def b := 30
def c := 45

-- Prove that the gcd of a, b, and c is 3
theorem gcd_18_30_45 : Nat.gcd (Nat.gcd a b) c = 3 :=
by
  -- Skip the proof itself
  sorry

end NUMINAMATH_GPT_gcd_18_30_45_l1828_182837


namespace NUMINAMATH_GPT_find_line_AB_l1828_182877

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 16

-- Define the line AB
def lineAB (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- Proof statement: Line AB is the correct line through the intersection points of the two circles
theorem find_line_AB :
  (∃ x y, circle1 x y ∧ circle2 x y) →
  (∀ x y, (circle1 x y ∧ circle2 x y) ↔ lineAB x y) :=
by
  sorry

end NUMINAMATH_GPT_find_line_AB_l1828_182877


namespace NUMINAMATH_GPT_prescription_duration_l1828_182898

theorem prescription_duration (D : ℕ) (h1 : (2 * D) * (1 / 5) = 12) : D = 30 :=
by
  sorry

end NUMINAMATH_GPT_prescription_duration_l1828_182898


namespace NUMINAMATH_GPT_number_of_roots_l1828_182841

def S : Set ℚ := { x : ℚ | 0 < x ∧ x < (5 : ℚ)/8 }

def f (x : ℚ) : ℚ := 
  match x.num, x.den with
  | num, den => num / den + 1

theorem number_of_roots (h : ∀ q p, (p, q) = 1 → (q : ℚ) / p ∈ S → ((q + 1 : ℚ) / p = (2 : ℚ) / 3)) :
  ∃ n : ℕ, n = 7 :=
sorry

end NUMINAMATH_GPT_number_of_roots_l1828_182841


namespace NUMINAMATH_GPT_cube_surface_area_correct_l1828_182888

def edge_length : ℝ := 11

def cube_surface_area (e : ℝ) : ℝ := 6 * e^2

theorem cube_surface_area_correct : cube_surface_area edge_length = 726 := by
  sorry

end NUMINAMATH_GPT_cube_surface_area_correct_l1828_182888


namespace NUMINAMATH_GPT_total_volume_of_all_cubes_l1828_182882

def volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume_of_cubes (num_cubes : ℕ) (side_length : ℕ) : ℕ :=
  num_cubes * volume side_length

theorem total_volume_of_all_cubes :
  total_volume_of_cubes 3 3 + total_volume_of_cubes 4 4 = 337 :=
by
  sorry

end NUMINAMATH_GPT_total_volume_of_all_cubes_l1828_182882


namespace NUMINAMATH_GPT_animal_sale_money_l1828_182831

theorem animal_sale_money (G S : ℕ) (h1 : G + S = 360) (h2 : 5 * S = 7 * G) : 
  (1/2 * G * 40) + (2/3 * S * 30) = 7200 := 
by
  sorry

end NUMINAMATH_GPT_animal_sale_money_l1828_182831


namespace NUMINAMATH_GPT_division_problem_l1828_182863

theorem division_problem :
  (0.25 / 0.005) / 0.1 = 500 := by
  sorry

end NUMINAMATH_GPT_division_problem_l1828_182863


namespace NUMINAMATH_GPT_Kindergarten_Students_l1828_182814

theorem Kindergarten_Students (X : ℕ) (h1 : 40 * X + 40 * 10 + 40 * 11 = 1200) : X = 9 :=
by
  sorry

end NUMINAMATH_GPT_Kindergarten_Students_l1828_182814


namespace NUMINAMATH_GPT_polynomial_solution_l1828_182892

theorem polynomial_solution (f : ℝ → ℝ) (x : ℝ) (h : f (x^2 + 2) = x^4 + 6 * x^2 + 4) : 
  f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l1828_182892


namespace NUMINAMATH_GPT_Phil_earns_per_hour_l1828_182897

-- Definitions based on the conditions in the problem
def Mike_hourly_rate : ℝ := 14
def Phil_hourly_rate : ℝ := Mike_hourly_rate - (0.5 * Mike_hourly_rate)

-- Mathematical assertion to prove
theorem Phil_earns_per_hour : Phil_hourly_rate = 7 :=
by 
  sorry

end NUMINAMATH_GPT_Phil_earns_per_hour_l1828_182897


namespace NUMINAMATH_GPT_inverse_function_point_l1828_182849

theorem inverse_function_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ (A : ℝ × ℝ), A = (2, 3) ∧ ∀ y, (∀ x, y = a^(x-3) + 1) → (2, 3) ∈ {(y, x) | y = a^(x-3) + 1} :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_point_l1828_182849


namespace NUMINAMATH_GPT_simultaneous_equations_solution_l1828_182843

theorem simultaneous_equations_solution (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ y = (2 * m - 1) * x + 4) ↔ m ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_simultaneous_equations_solution_l1828_182843


namespace NUMINAMATH_GPT_intersecting_lines_l1828_182819

theorem intersecting_lines (m : ℝ) :
  (∃ (x y : ℝ), y = 2 * x ∧ x + y = 3 ∧ m * x + 2 * y + 5 = 0) ↔ (m = -9) :=
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_l1828_182819


namespace NUMINAMATH_GPT_minimize_base_side_length_l1828_182879

theorem minimize_base_side_length (V : ℝ) (a h : ℝ) 
  (volume_eq : V = a ^ 2 * h) (V_given : V = 256) (h_eq : h = 256 / (a ^ 2)) :
  a = 8 :=
by
  -- Recognize that for a given volume, making it a cube minimizes the surface area.
  -- As the volume of the cube a^3 = 256, solving for a gives 8.
  -- a := (256:ℝ) ^ (1/3:ℝ)
  sorry

end NUMINAMATH_GPT_minimize_base_side_length_l1828_182879


namespace NUMINAMATH_GPT_log4_80_cannot_be_found_without_additional_values_l1828_182822

-- Conditions provided in the problem
def log4_16 : Real := 2
def log4_32 : Real := 2.5

-- Lean statement of the proof problem
theorem log4_80_cannot_be_found_without_additional_values :
  ¬(∃ (log4_80 : Real), log4_80 = log4_16 + log4_5) :=
sorry

end NUMINAMATH_GPT_log4_80_cannot_be_found_without_additional_values_l1828_182822


namespace NUMINAMATH_GPT_evaluate_expression_l1828_182891

-- Define the terms a and b
def a : ℕ := 2023
def b : ℕ := 2024

-- The given expression
def expression : ℤ := (a^3 - 2 * a^2 * b + 3 * a * b^2 - b^3 + 1) / (a * b)

-- The theorem to prove
theorem evaluate_expression : expression = ↑a := 
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1828_182891


namespace NUMINAMATH_GPT_series_sum_l1828_182820

theorem series_sum :
  ∑' n : ℕ,  n ≠ 0 → (6 * n + 2) / ((6 * n - 1)^2 * (6 * n + 5)^2) = 1 / 600 := by
  sorry

end NUMINAMATH_GPT_series_sum_l1828_182820


namespace NUMINAMATH_GPT_intersection_point_l1828_182896

theorem intersection_point : ∃ (x y : ℝ), y = 3 - x ∧ y = 3 * x - 5 ∧ x = 2 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_l1828_182896


namespace NUMINAMATH_GPT_find_f_inv_486_l1828_182865

-- Assuming function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Given conditions
axiom f_cond1 : f 4 = 2
axiom f_cond2 : ∀ x : ℝ, f (3 * x) = 3 * f x

-- Proof problem: Prove that f⁻¹(486) = 972
theorem find_f_inv_486 : (∃ x : ℝ, f x = 486 ∧ x = 972) :=
sorry

end NUMINAMATH_GPT_find_f_inv_486_l1828_182865


namespace NUMINAMATH_GPT_find_calories_per_slice_l1828_182861

/-- Defining the number of slices and their respective calories. -/
def slices_in_cake : ℕ := 8
def calories_per_brownie : ℕ := 375
def brownies_in_pan : ℕ := 6
def extra_calories_in_cake : ℕ := 526

/-- Defining the total calories in cake and brownies -/
def total_calories_in_brownies : ℕ := brownies_in_pan * calories_per_brownie
def total_calories_in_cake (c : ℕ) : ℕ := slices_in_cake * c

/-- The equation from the given problem -/
theorem find_calories_per_slice (c : ℕ) :
  total_calories_in_cake c = total_calories_in_brownies + extra_calories_in_cake → c = 347 :=
by
  sorry

end NUMINAMATH_GPT_find_calories_per_slice_l1828_182861


namespace NUMINAMATH_GPT_find_other_number_l1828_182864

theorem find_other_number (B : ℕ) (hcf_cond : Nat.gcd 36 B = 14) (lcm_cond : Nat.lcm 36 B = 396) : B = 66 :=
sorry

end NUMINAMATH_GPT_find_other_number_l1828_182864


namespace NUMINAMATH_GPT_xiao_ming_speed_difference_l1828_182821

noncomputable def distance_school : ℝ := 9.3
noncomputable def time_cycling : ℝ := 0.6
noncomputable def distance_park : ℝ := 0.9
noncomputable def time_walking : ℝ := 0.2

noncomputable def cycling_speed : ℝ := distance_school / time_cycling
noncomputable def walking_speed : ℝ := distance_park / time_walking
noncomputable def speed_difference : ℝ := cycling_speed - walking_speed

theorem xiao_ming_speed_difference : speed_difference = 11 := by
  sorry

end NUMINAMATH_GPT_xiao_ming_speed_difference_l1828_182821


namespace NUMINAMATH_GPT_novel_cost_l1828_182807

-- Given conditions
variable (N : ℕ) -- cost of the novel
variable (lunch_cost : ℕ) -- cost of lunch

-- Conditions
axiom gift_amount : N + lunch_cost + 29 = 50
axiom lunch_cost_eq : lunch_cost = 2 * N

-- Question and answer tuple as a theorem
theorem novel_cost : N = 7 := 
by
  sorry -- Proof estaps are to be filled in.

end NUMINAMATH_GPT_novel_cost_l1828_182807


namespace NUMINAMATH_GPT_inequality_proof_l1828_182846

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1828_182846


namespace NUMINAMATH_GPT_find_some_number_l1828_182815

def simplify_expr (x : ℚ) : Prop :=
  1 / 2 + ((2 / 3 * (3 / 8)) + x) - (8 / 16) = 4.25

theorem find_some_number :
  ∃ x : ℚ, simplify_expr x ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_some_number_l1828_182815


namespace NUMINAMATH_GPT_peanuts_added_l1828_182823

theorem peanuts_added (a b x : ℕ) (h1 : a = 4) (h2 : b = 6) (h3 : a + x = b) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_peanuts_added_l1828_182823


namespace NUMINAMATH_GPT_negation_exists_equation_l1828_182889

theorem negation_exists_equation (P : ℝ → Prop) :
  (∃ x > 0, x^2 + 3 * x - 5 = 0) → ¬ (∃ x > 0, x^2 + 3 * x - 5 = 0) = ∀ x > 0, x^2 + 3 * x - 5 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_negation_exists_equation_l1828_182889


namespace NUMINAMATH_GPT_left_seats_equals_15_l1828_182873

variable (L : ℕ)

noncomputable def num_seats_left (L : ℕ) : Prop :=
  ∃ L, 3 * L + 3 * (L - 3) + 8 = 89

theorem left_seats_equals_15 : num_seats_left L → L = 15 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_left_seats_equals_15_l1828_182873


namespace NUMINAMATH_GPT_find_width_of_rect_box_l1828_182860

-- Define the dimensions of the wooden box in meters
def wooden_box_length_m : ℕ := 8
def wooden_box_width_m : ℕ := 7
def wooden_box_height_m : ℕ := 6

-- Define the dimensions of the rectangular boxes in centimeters (with unknown width W)
def rect_box_length_cm : ℕ := 8
def rect_box_height_cm : ℕ := 6

-- Define the maximum number of rectangular boxes
def max_boxes : ℕ := 1000000

-- Define the constraint that the total volume of the boxes should not exceed the volume of the wooden box
theorem find_width_of_rect_box (W : ℕ) (wooden_box_volume : ℕ := (wooden_box_length_m * 100) * (wooden_box_width_m * 100) * (wooden_box_height_m * 100)) : 
  (rect_box_length_cm * W * rect_box_height_cm) * max_boxes = wooden_box_volume → W = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_width_of_rect_box_l1828_182860


namespace NUMINAMATH_GPT_time_to_cross_platform_l1828_182838

variable (l t p : ℝ) -- Define relevant variables

-- Conditions as definitions in Lean 4
def length_of_train := l
def time_to_pass_man := t
def length_of_platform := p

-- Assume given values in the problem
def cond1 : length_of_train = 186 := by sorry
def cond2 : time_to_pass_man = 8 := by sorry
def cond3 : length_of_platform = 279 := by sorry

-- Statement that represents the target theorem to be proved
theorem time_to_cross_platform (h₁ : length_of_train = 186) (h₂ : time_to_pass_man = 8) (h₃ : length_of_platform = 279) : 
  let speed := length_of_train / time_to_pass_man
  let total_distance := length_of_train + length_of_platform
  let time_to_cross := total_distance / speed
  time_to_cross = 20 :=
by sorry

end NUMINAMATH_GPT_time_to_cross_platform_l1828_182838


namespace NUMINAMATH_GPT_calculate_total_cost_l1828_182810

def total_cost (num_boxes : ℕ) (packs_per_box : ℕ) (tissues_per_pack : ℕ) (cost_per_tissue : ℝ) : ℝ :=
  num_boxes * packs_per_box * tissues_per_pack * cost_per_tissue

theorem calculate_total_cost :
  total_cost 10 20 100 0.05 = 1000 := 
by
  sorry

end NUMINAMATH_GPT_calculate_total_cost_l1828_182810


namespace NUMINAMATH_GPT_solve_for_x_l1828_182844

theorem solve_for_x (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (4 * x + 28)) : 
  x = -17 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1828_182844


namespace NUMINAMATH_GPT_mike_daily_work_hours_l1828_182866

def total_hours_worked : ℕ := 15
def number_of_days_worked : ℕ := 5

theorem mike_daily_work_hours : total_hours_worked / number_of_days_worked = 3 :=
by
  sorry

end NUMINAMATH_GPT_mike_daily_work_hours_l1828_182866


namespace NUMINAMATH_GPT_third_number_in_first_set_is_42_l1828_182805

theorem third_number_in_first_set_is_42 (x y : ℕ) :
  (28 + x + y + 78 + 104) / 5 = 90 →
  (128 + 255 + 511 + 1023 + x) / 5 = 423 →
  y = 42 :=
by { sorry }

end NUMINAMATH_GPT_third_number_in_first_set_is_42_l1828_182805


namespace NUMINAMATH_GPT_find_sticker_price_l1828_182859

variable (x : ℝ)

def price_at_store_A (x : ℝ) : ℝ := 0.80 * x - 120
def price_at_store_B (x : ℝ) : ℝ := 0.70 * x
def savings (x : ℝ) : ℝ := price_at_store_B x - price_at_store_A x

theorem find_sticker_price (h : savings x = 30) : x = 900 :=
by
  -- proof can be filled in here
  sorry

end NUMINAMATH_GPT_find_sticker_price_l1828_182859


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1828_182871

-- Define arithmetic sequence and sum of first n terms
def arithmetic_seq (a d : ℕ → ℕ) :=
  ∀ n, a (n + 1) = a n + d 1

def arithmetic_sum (a d : ℕ → ℕ) (n : ℕ) :=
  (n * (a 1 + a n)) / 2

-- Conditions from the problem
variables {a : ℕ → ℕ} {d : ℕ}

axiom condition : a 3 + a 7 + a 11 = 6

-- Definition of a_7 as derived in the solution
def a_7 : ℕ := 2

-- Proof problem equivalent statement
theorem arithmetic_sequence_sum : arithmetic_sum a d 13 = 26 :=
by
  -- These steps would involve setting up and proving the calculation details
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1828_182871


namespace NUMINAMATH_GPT_stratified_sampling_example_l1828_182886

theorem stratified_sampling_example 
  (N : ℕ) (S : ℕ) (D : ℕ) 
  (hN : N = 1000) (hS : S = 50) (hD : D = 200) : 
  D * (S : ℝ) / (N : ℝ) = 10 := 
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_example_l1828_182886


namespace NUMINAMATH_GPT_laila_utility_l1828_182845

theorem laila_utility (u : ℝ) :
  (2 * u * (10 - 2 * u) = 2 * (4 - 2 * u) * (2 * u + 4)) → u = 4 := 
by 
  sorry

end NUMINAMATH_GPT_laila_utility_l1828_182845


namespace NUMINAMATH_GPT_sqrt_expression_evaluation_l1828_182894

theorem sqrt_expression_evaluation : 
  (Real.sqrt (4 + 2 * Real.sqrt 3) + Real.sqrt (4 - 2 * Real.sqrt 3) = 2) := 
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_evaluation_l1828_182894


namespace NUMINAMATH_GPT_total_pieces_of_pizza_l1828_182848

def pieces_per_pizza : ℕ := 6
def pizzas_per_fourthgrader : ℕ := 20
def fourthgraders_count : ℕ := 10

theorem total_pieces_of_pizza :
  fourthgraders_count * (pieces_per_pizza * pizzas_per_fourthgrader) = 1200 :=
by
  /-
  We have:
  - pieces_per_pizza = 6
  - pizzas_per_fourthgrader = 20
  - fourthgraders_count = 10

  Therefore,
  10 * (6 * 20) = 1200
  -/
  sorry

end NUMINAMATH_GPT_total_pieces_of_pizza_l1828_182848


namespace NUMINAMATH_GPT_quadratic_vertex_a_l1828_182852

theorem quadratic_vertex_a
  (a b c : ℝ)
  (h1 : ∀ x, (a * x^2 + b * x + c = a * (x - 2)^2 + 5))
  (h2 : a * 0^2 + b * 0 + c = 0) :
  a = -5/4 :=
by
  -- Use the given conditions to outline the proof (proof not provided here as per instruction)
  sorry

end NUMINAMATH_GPT_quadratic_vertex_a_l1828_182852


namespace NUMINAMATH_GPT_car_speed_l1828_182818

variable (Distance : ℕ) (Time : ℕ)
variable (h1 : Distance = 495)
variable (h2 : Time = 5)

theorem car_speed (Distance Time : ℕ) (h1 : Distance = 495) (h2 : Time = 5) : 
  Distance / Time = 99 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_l1828_182818


namespace NUMINAMATH_GPT_lemons_and_oranges_for_100_gallons_l1828_182829

-- Given conditions
def lemons_per_gallon := 30 / 40
def oranges_per_gallon := 20 / 40

-- Theorem to be proven
theorem lemons_and_oranges_for_100_gallons : 
  lemons_per_gallon * 100 = 75 ∧ oranges_per_gallon * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_lemons_and_oranges_for_100_gallons_l1828_182829


namespace NUMINAMATH_GPT_reciprocal_neg_one_over_2011_l1828_182834

theorem reciprocal_neg_one_over_2011 : 1 / (- (1 / 2011)) = -2011 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_neg_one_over_2011_l1828_182834


namespace NUMINAMATH_GPT_geometric_sequence_b_value_l1828_182857

theorem geometric_sequence_b_value 
  (b : ℝ)
  (h1 : b > 0)
  (h2 : ∃ r : ℝ, 160 * r = b ∧ b * r = 1)
  : b = 4 * Real.sqrt 10 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_b_value_l1828_182857


namespace NUMINAMATH_GPT_proportional_relationships_l1828_182847

-- Let l, v, t be real numbers indicating distance, velocity, and time respectively.
variables (l v t : ℝ)

-- Define the relationships according to the given formulas
def distance_formula := l = v * t
def velocity_formula := v = l / t
def time_formula := t = l / v

-- Definitions of proportionality
def directly_proportional (x y : ℝ) := ∃ k : ℝ, x = k * y
def inversely_proportional (x y : ℝ) := ∃ k : ℝ, x * y = k

-- The main theorem
theorem proportional_relationships (const_t const_v const_l : ℝ) :
  (distance_formula l v const_t → directly_proportional l v) ∧
  (distance_formula l const_v t → directly_proportional l t) ∧
  (velocity_formula const_l v t → inversely_proportional v t) :=
by
  sorry

end NUMINAMATH_GPT_proportional_relationships_l1828_182847


namespace NUMINAMATH_GPT_intersection_is_4_l1828_182832

-- Definitions of the sets
def U : Set Int := {0, 1, 2, 4, 6, 8}
def M : Set Int := {0, 4, 6}
def N : Set Int := {0, 1, 6}

-- Definition of the complement
def complement_U_N : Set Int := U \ N

-- Definition of the intersection
def intersection_M_complement_U_N : Set Int := M ∩ complement_U_N

-- Statement of the theorem
theorem intersection_is_4 : intersection_M_complement_U_N = {4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_is_4_l1828_182832


namespace NUMINAMATH_GPT_simplify_fraction_l1828_182868

theorem simplify_fraction (a b c : ℕ) (h1 : a = 222) (h2 : b = 8888) (h3 : c = 44) : 
  (a : ℚ) / b * c = 111 / 101 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1828_182868


namespace NUMINAMATH_GPT_spotted_mushrooms_ratio_l1828_182858

theorem spotted_mushrooms_ratio 
  (total_mushrooms : ℕ) 
  (gilled_mushrooms : ℕ) 
  (spotted_mushrooms : ℕ) 
  (total_mushrooms_eq : total_mushrooms = 30) 
  (gilled_mushrooms_eq : gilled_mushrooms = 3) 
  (spots_and_gills_exclusive : ∀ x, x = spotted_mushrooms ∨ x = gilled_mushrooms) : 
  spotted_mushrooms / gilled_mushrooms = 9 := 
by
  sorry

end NUMINAMATH_GPT_spotted_mushrooms_ratio_l1828_182858


namespace NUMINAMATH_GPT_ratio_of_pages_given_l1828_182862

variable (Lana_initial_pages : ℕ) (Duane_initial_pages : ℕ) (Lana_final_pages : ℕ)

theorem ratio_of_pages_given
  (h1 : Lana_initial_pages = 8)
  (h2 : Duane_initial_pages = 42)
  (h3 : Lana_final_pages = 29) :
  (Lana_final_pages - Lana_initial_pages) / Duane_initial_pages = 1 / 2 :=
  by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_ratio_of_pages_given_l1828_182862


namespace NUMINAMATH_GPT_find_certain_number_l1828_182872

theorem find_certain_number (n : ℕ) (h : 9823 + n = 13200) : n = 3377 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l1828_182872


namespace NUMINAMATH_GPT_general_term_a_n_sum_b_n_terms_l1828_182804

-- Given definitions based on the conditions
def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := (2^(2*n-1))^2

def b_sum (n : ℕ) : (ℕ → ℕ) := 
  (fun b : ℕ => match b with 
                | 1 => 4 
                | 2 => 64 
                | _ => (4^(2*(b - 2 + 1) - 1)))

def T (n : ℕ) : ℕ := (4 / 15) * (16^n - 1)

-- First part: Proving the general term of {a_n} is 2^(n-1)
theorem general_term_a_n (n : ℕ) : a n = 2^(n-1) := by
  sorry

-- Second part: Proving the sum of the first n terms of {b_n} is (4/15)*(16^n - 1)
theorem sum_b_n_terms (n : ℕ) : T n = (4 / 15) * (16^n - 1) := by 
  sorry

end NUMINAMATH_GPT_general_term_a_n_sum_b_n_terms_l1828_182804


namespace NUMINAMATH_GPT_line_ellipse_intersect_l1828_182800

theorem line_ellipse_intersect (m k : ℝ) (h₀ : ∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1) : m ≥ 1 ∧ m ≠ 5 :=
sorry

end NUMINAMATH_GPT_line_ellipse_intersect_l1828_182800


namespace NUMINAMATH_GPT_largest_constant_c_l1828_182880

theorem largest_constant_c :
  ∃ c : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x^2 + y^2 = 1 → x^6 + y^6 ≥ c * x * y) ∧ c = 1 / 2 :=
sorry

end NUMINAMATH_GPT_largest_constant_c_l1828_182880


namespace NUMINAMATH_GPT_sum_remainder_div_9_l1828_182887

theorem sum_remainder_div_9 : 
  let S := (20 / 2) * (1 + 20)
  S % 9 = 3 := 
by
  -- use let S to simplify the proof
  let S := (20 / 2) * (1 + 20)
  -- sum of first 20 natural numbers
  have H1 : S = 210 := by sorry
  -- division and remainder result
  have H2 : 210 % 9 = 3 := by sorry
  -- combine both results to conclude 
  exact H2

end NUMINAMATH_GPT_sum_remainder_div_9_l1828_182887


namespace NUMINAMATH_GPT_initial_number_306_l1828_182895

theorem initial_number_306 (x : ℝ) : 
  (x / 34) * 15 + 270 = 405 → x = 306 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_number_306_l1828_182895


namespace NUMINAMATH_GPT_product_eq_one_of_abs_log_eq_l1828_182806

theorem product_eq_one_of_abs_log_eq (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |Real.log a| = |Real.log b|) : a * b = 1 := 
sorry

end NUMINAMATH_GPT_product_eq_one_of_abs_log_eq_l1828_182806


namespace NUMINAMATH_GPT_bananas_more_than_pears_l1828_182899

theorem bananas_more_than_pears (A P B : ℕ) (h1 : P = A + 2) (h2 : A + P + B = 19) (h3 : B = 9) : B - P = 3 :=
  by
  sorry

end NUMINAMATH_GPT_bananas_more_than_pears_l1828_182899


namespace NUMINAMATH_GPT_triangle_BPC_area_l1828_182812

universe u

variables {T : Type u} [LinearOrderedField T]

-- Define the points
variables (A B C E F P : T)
variables (area : T → T → T → T) -- A function to compute the area of a triangle

-- Hypotheses
def conditions :=
  E ∈ [A, B] ∧
  F ∈ [A, C] ∧
  (∃ P, P ∈ [B, F] ∧ P ∈ [C, E]) ∧
  area A E P + area E P F + area P F A = 4 ∧ -- AEPF
  area B E P = 4 ∧ -- BEP
  area C F P = 4   -- CFP

-- The theorem to prove
theorem triangle_BPC_area (h : conditions A B C E F P area) : area B P C = 12 :=
sorry

end NUMINAMATH_GPT_triangle_BPC_area_l1828_182812


namespace NUMINAMATH_GPT_avg_xy_l1828_182885

theorem avg_xy (x y : ℝ) (h : (4 + 6.5 + 8 + x + y) / 5 = 18) : (x + y) / 2 = 35.75 :=
by
  sorry

end NUMINAMATH_GPT_avg_xy_l1828_182885


namespace NUMINAMATH_GPT_parallelogram_side_problem_l1828_182867

theorem parallelogram_side_problem (y z : ℝ) (h1 : 4 * z + 1 = 15) (h2 : 3 * y - 2 = 15) :
  y + z = 55 / 6 :=
sorry

end NUMINAMATH_GPT_parallelogram_side_problem_l1828_182867


namespace NUMINAMATH_GPT_ineq_10_3_minus_9_5_l1828_182816

variable {a b c : ℝ}

/-- Given \(a, b, c\) are positive real numbers and \(a + b + c = 1\), prove \(10(a^3 + b^3 + c^3) - 9(a^5 + b^5 + c^5) \geq 1\). -/
theorem ineq_10_3_minus_9_5 (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := 
sorry

end NUMINAMATH_GPT_ineq_10_3_minus_9_5_l1828_182816


namespace NUMINAMATH_GPT_find_a_l1828_182811

theorem find_a 
  (x : ℤ) 
  (a : ℤ) 
  (h1 : x = 2) 
  (h2 : y = a) 
  (h3 : 2 * x - 3 * y = 5) : a = -1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l1828_182811


namespace NUMINAMATH_GPT_find_q_l1828_182854

variable (p q : ℝ)
variable (h1 : 1 < p)
variable (h2 : p < q)
variable (h3 : 1 / p + 1 / q = 1)
variable (h4 : p * q = 8)

theorem find_q : q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l1828_182854


namespace NUMINAMATH_GPT_top_black_second_red_probability_l1828_182874

-- Define the problem conditions in Lean
def num_standard_cards : ℕ := 52
def num_jokers : ℕ := 2
def num_total_cards : ℕ := num_standard_cards + num_jokers

def num_black_cards : ℕ := 26
def num_red_cards : ℕ := 26

-- Lean statement
theorem top_black_second_red_probability :
  (num_black_cards / num_total_cards * num_red_cards / (num_total_cards - 1)) = 338 / 1431 := by
  sorry

end NUMINAMATH_GPT_top_black_second_red_probability_l1828_182874


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1828_182884

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h_arithmetic : ∀ n, a (n+1) = a n + d)
  (h_pos_diff : d > 0)
  (h_sum_3 : a 0 + a 1 + a 2 = 15)
  (h_prod_3 : a 0 * a 1 * a 2 = 80) :
  a 10 + a 11 + a 12 = 105 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1828_182884


namespace NUMINAMATH_GPT_smallest_integer_divisible_l1828_182840

theorem smallest_integer_divisible:
  ∃ n : ℕ, n > 1 ∧ (n % 4 = 1) ∧ (n % 5 = 1) ∧ (n % 6 = 1) ∧ n = 61 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_divisible_l1828_182840


namespace NUMINAMATH_GPT_smallest_three_digit_number_with_property_l1828_182881

theorem smallest_three_digit_number_with_property :
  ∃ (a : ℕ), 100 ≤ a ∧ a ≤ 999 ∧ (∃ (n : ℕ), 317 ≤ n ∧ n ≤ 999 ∧ 1001 * a + 1 = n^2) ∧ a = 183 :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_number_with_property_l1828_182881


namespace NUMINAMATH_GPT_fractional_identity_l1828_182801

theorem fractional_identity (m n r t : ℚ) 
  (h₁ : m / n = 5 / 2) 
  (h₂ : r / t = 8 / 5) : 
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -5 / 11 :=
by 
  sorry

end NUMINAMATH_GPT_fractional_identity_l1828_182801
