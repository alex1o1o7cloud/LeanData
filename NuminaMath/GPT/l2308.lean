import Mathlib

namespace NUMINAMATH_GPT_manager_salary_proof_l2308_230836

noncomputable def manager_salary 
    (avg_salary_without_manager : ℝ) 
    (num_employees_without_manager : ℕ) 
    (increase_in_avg_salary : ℝ) 
    (new_total_salary : ℝ) : ℝ :=
    new_total_salary - (num_employees_without_manager * avg_salary_without_manager)

theorem manager_salary_proof :
    manager_salary 3500 100 800 (101 * (3500 + 800)) = 84300 :=
by
    sorry

end NUMINAMATH_GPT_manager_salary_proof_l2308_230836


namespace NUMINAMATH_GPT_total_pencils_l2308_230893

theorem total_pencils (initial_additional1 initial_additional2 : ℕ) (h₁ : initial_additional1 = 37) (h₂ : initial_additional2 = 17) : (initial_additional1 + initial_additional2) = 54 :=
by sorry

end NUMINAMATH_GPT_total_pencils_l2308_230893


namespace NUMINAMATH_GPT_range_of_m_l2308_230879

def f (a b x : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

def f_prime (a b x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + b

def has_local_extremum_at (a b x : ℝ) : Prop :=
  f_prime a b x = 0 ∧ f a b x = 0

def h (a b m x : ℝ) : ℝ := f a b x - m + 1

theorem range_of_m (a b m : ℝ) :
  (has_local_extremum_at 2 9 (-1) ∧
   ∀ x, f 2 9 x = x^3 + 6 * x^2 + 9 * x + 4) →
  (∀ x, (x^3 + 6 * x^2 + 9 * x + 4 - m + 1 = 0) → 
  1 < m ∧ m < 5) := 
sorry

end NUMINAMATH_GPT_range_of_m_l2308_230879


namespace NUMINAMATH_GPT_sin_double_angle_subst_l2308_230854

open Real

theorem sin_double_angle_subst 
  (α : ℝ)
  (h : sin (α + π / 6) = -1 / 3) :
  sin (2 * α - π / 6) = -7 / 9 := 
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_subst_l2308_230854


namespace NUMINAMATH_GPT_operation_1_and_2004_l2308_230822

def operation (m n : ℕ) : ℕ :=
  if m = 1 ∧ n = 1 then 2
  else if m = 1 ∧ n > 1 then 2 + 3 * (n - 1)
  else 0 -- handle other cases generically, although specifics are not given

theorem operation_1_and_2004 : operation 1 2004 = 6011 :=
by
  unfold operation
  sorry

end NUMINAMATH_GPT_operation_1_and_2004_l2308_230822


namespace NUMINAMATH_GPT_prob_ending_game_after_five_distribution_and_expectation_l2308_230841

-- Define the conditions
def shooting_accuracy_rate : ℚ := 2 / 3
def game_clear_coupon : ℕ := 9
def game_fail_coupon : ℕ := 3
def game_no_clear_no_fail_coupon : ℕ := 6

-- Define the probabilities for ending the game after 5 shots
def ending_game_after_five : ℚ := (shooting_accuracy_rate^2 * (1 - shooting_accuracy_rate)^3 * 2) + (shooting_accuracy_rate^4 * (1 - shooting_accuracy_rate))

-- Define the distribution table
def P_clear : ℚ := (shooting_accuracy_rate^3) + (shooting_accuracy_rate^3 * (1 - shooting_accuracy_rate)) + (shooting_accuracy_rate^4 * (1 - shooting_accuracy_rate) * 2)
def P_fail : ℚ := ((1 - shooting_accuracy_rate)^2) + ((1 - shooting_accuracy_rate)^2 * shooting_accuracy_rate * 2) + ((1 - shooting_accuracy_rate)^3 * shooting_accuracy_rate^2 * 3) + ((1 - shooting_accuracy_rate)^3 * shooting_accuracy_rate^3)
def P_neither : ℚ := 1 - P_clear - P_fail

-- Expected value calculation
def expectation : ℚ := (P_fail * game_fail_coupon) + (P_neither * game_no_clear_no_fail_coupon) + (P_clear * game_clear_coupon)

-- The Part I proof statement
theorem prob_ending_game_after_five : ending_game_after_five = 8 / 81 :=
by
  sorry

-- The Part II proof statement
theorem distribution_and_expectation (X : ℕ → ℚ) :
  (X game_fail_coupon = 233 / 729) ∧
  (X game_no_clear_no_fail_coupon = 112 / 729) ∧
  (X game_clear_coupon = 128 / 243) ∧
  (expectation = 1609 / 243) :=
by
  sorry

end NUMINAMATH_GPT_prob_ending_game_after_five_distribution_and_expectation_l2308_230841


namespace NUMINAMATH_GPT_girls_attending_event_l2308_230858

theorem girls_attending_event (g b : ℕ) 
  (h1 : g + b = 1500)
  (h2 : 3 / 4 * g + 2 / 5 * b = 900) :
  3 / 4 * g = 643 := 
by
  sorry

end NUMINAMATH_GPT_girls_attending_event_l2308_230858


namespace NUMINAMATH_GPT_bruce_three_times_son_in_six_years_l2308_230883

-- Define the current ages of Bruce and his son
def bruce_age : ℕ := 36
def son_age : ℕ := 8

-- Define the statement to be proved
theorem bruce_three_times_son_in_six_years :
  ∃ (x : ℕ), x = 6 ∧ ∀ t, (t = x) → (bruce_age + t = 3 * (son_age + t)) :=
by
  sorry

end NUMINAMATH_GPT_bruce_three_times_son_in_six_years_l2308_230883


namespace NUMINAMATH_GPT_fraction_computation_l2308_230801

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_computation_l2308_230801


namespace NUMINAMATH_GPT_average_weight_of_class_l2308_230818

variable (SectionA_students : ℕ := 26)
variable (SectionB_students : ℕ := 34)
variable (SectionA_avg_weight : ℝ := 50)
variable (SectionB_avg_weight : ℝ := 30)

theorem average_weight_of_class :
  (SectionA_students * SectionA_avg_weight + SectionB_students * SectionB_avg_weight) / (SectionA_students + SectionB_students) = 38.67 := by
  sorry

end NUMINAMATH_GPT_average_weight_of_class_l2308_230818


namespace NUMINAMATH_GPT_minimize_sum_areas_l2308_230880

theorem minimize_sum_areas (x : ℝ) (h_wire_length : 0 < x ∧ x < 1) :
    let side_length := x / 4
    let square_area := (side_length ^ 2)
    let circle_radius := (1 - x) / (2 * Real.pi)
    let circle_area := Real.pi * (circle_radius ^ 2)
    let total_area := square_area + circle_area
    total_area = (x^2 / 16 + (1 - x)^2 / (4 * Real.pi)) -> 
    x = Real.pi / (Real.pi + 4) :=
by
  sorry

end NUMINAMATH_GPT_minimize_sum_areas_l2308_230880


namespace NUMINAMATH_GPT_min_rubles_reaching_50_points_l2308_230814

-- Define conditions and prove the required rubles amount
def min_rubles_needed : ℕ := 11

theorem min_rubles_reaching_50_points (points : ℕ) (rubles : ℕ) : points = 50 ∧ rubles = min_rubles_needed → rubles = 11 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_min_rubles_reaching_50_points_l2308_230814


namespace NUMINAMATH_GPT_intersection_eq_l2308_230821

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_eq : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l2308_230821


namespace NUMINAMATH_GPT_base8_357_plus_base13_4CD_eq_1084_l2308_230869

def C := 12
def D := 13

def base8_357 := 3 * (8^2) + 5 * (8^1) + 7 * (8^0)
def base13_4CD := 4 * (13^2) + C * (13^1) + D * (13^0)

theorem base8_357_plus_base13_4CD_eq_1084 :
  base8_357 + base13_4CD = 1084 :=
by
  sorry

end NUMINAMATH_GPT_base8_357_plus_base13_4CD_eq_1084_l2308_230869


namespace NUMINAMATH_GPT_ethanol_solution_exists_l2308_230870

noncomputable def ethanol_problem : Prop :=
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 204 ∧ 0.12 * x + 0.16 * (204 - x) = 30

theorem ethanol_solution_exists : ethanol_problem :=
sorry

end NUMINAMATH_GPT_ethanol_solution_exists_l2308_230870


namespace NUMINAMATH_GPT_triangle_inequality_check_triangle_sets_l2308_230897

theorem triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem check_triangle_sets :
  ¬triangle_inequality 1 2 3 ∧
  triangle_inequality 2 2 2 ∧
  ¬triangle_inequality 2 2 4 ∧
  ¬triangle_inequality 1 3 5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_check_triangle_sets_l2308_230897


namespace NUMINAMATH_GPT_unique_function_satisfying_conditions_l2308_230834

theorem unique_function_satisfying_conditions :
  ∀ (f : ℝ → ℝ), 
    (∀ x : ℝ, f x ≥ 0) → 
    (∀ x : ℝ, f (x^2) = f x ^ 2 - 2 * x * f x) →
    (∀ x : ℝ, f (-x) = f (x - 1)) → 
    (∀ x y : ℝ, 1 < x → x < y → f x < f y) →
    (∀ x : ℝ, f x = x^2 + x + 1) :=
by
  -- formal proof would go here
  sorry

end NUMINAMATH_GPT_unique_function_satisfying_conditions_l2308_230834


namespace NUMINAMATH_GPT_smallest_of_five_consecutive_l2308_230855

theorem smallest_of_five_consecutive (n : ℤ) (h : (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 2015) : n - 2 = 401 :=
by sorry

end NUMINAMATH_GPT_smallest_of_five_consecutive_l2308_230855


namespace NUMINAMATH_GPT_total_distance_l2308_230846

noncomputable def total_distance_covered 
  (radius1 radius2 radius3 : ℝ) 
  (rev1 rev2 rev3 : ℕ) : ℝ :=
  let π := Real.pi
  let circumference r := 2 * π * r
  let distance r rev := circumference r * rev
  distance radius1 rev1 + distance radius2 rev2 + distance radius3 rev3

theorem total_distance
  (h1 : radius1 = 20.4) 
  (h2 : radius2 = 15.3) 
  (h3 : radius3 = 25.6) 
  (h4 : rev1 = 400) 
  (h5 : rev2 = 320) 
  (h6 : rev3 = 500) :
  total_distance_covered 20.4 15.3 25.6 400 320 500 = 162436.6848 := 
sorry

end NUMINAMATH_GPT_total_distance_l2308_230846


namespace NUMINAMATH_GPT_min_value_of_f_solution_set_of_inequality_l2308_230877

-- Define the given function f
def f (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 4)

-- (1) Prove that the minimum value of y = f(x) is 3
theorem min_value_of_f : ∃ x : ℝ, f x = 3 := 
sorry

-- (2) Prove that the solution set of the inequality |f(x) - 6| ≤ 1 is [-10/3, -8/3] ∪ [0, 4/3]
theorem solution_set_of_inequality : 
  {x | |f x - 6| ≤ 1} = {x | -(10/3) ≤ x ∧ x ≤ -(8/3) ∨ 0 ≤ x ∧ x ≤ (4/3)} :=
sorry

end NUMINAMATH_GPT_min_value_of_f_solution_set_of_inequality_l2308_230877


namespace NUMINAMATH_GPT_inequality_abc_l2308_230892

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^3) / (a^2 + a * b + b^2) + (b^3) / (b^2 + b * c + c^2) + (c^3) / (c^2 + c * a + a^2) ≥ (a + b + c) / 3 := 
by
    sorry

end NUMINAMATH_GPT_inequality_abc_l2308_230892


namespace NUMINAMATH_GPT_Ferris_break_length_l2308_230835

noncomputable def Audrey_rate_per_hour := (1:ℝ) / 4
noncomputable def Ferris_rate_per_hour := (1:ℝ) / 3
noncomputable def total_completion_time := (2:ℝ)
noncomputable def number_of_breaks := (6:ℝ)
noncomputable def job_completion_audrey := total_completion_time * Audrey_rate_per_hour
noncomputable def job_completion_ferris := 1 - job_completion_audrey
noncomputable def working_time_ferris := job_completion_ferris / Ferris_rate_per_hour
noncomputable def total_break_time := total_completion_time - working_time_ferris
noncomputable def break_length := total_break_time / number_of_breaks

theorem Ferris_break_length :
  break_length = (5:ℝ) / 60 := 
sorry

end NUMINAMATH_GPT_Ferris_break_length_l2308_230835


namespace NUMINAMATH_GPT_find_k_l2308_230824

theorem find_k (α β k : ℝ) (h₁ : α^2 - α + k - 1 = 0) (h₂ : β^2 - β + k - 1 = 0) (h₃ : α^2 - 2*α - β = 4) :
  k = -4 :=
sorry

end NUMINAMATH_GPT_find_k_l2308_230824


namespace NUMINAMATH_GPT_power_division_simplify_l2308_230889

theorem power_division_simplify :
  ((9^9 / 9^8)^2 * 3^4) / 2^4 = 410 + 1/16 := by
  sorry

end NUMINAMATH_GPT_power_division_simplify_l2308_230889


namespace NUMINAMATH_GPT_limo_gas_price_l2308_230850

theorem limo_gas_price
  (hourly_wage : ℕ := 15)
  (ride_payment : ℕ := 5)
  (review_bonus : ℕ := 20)
  (hours_worked : ℕ := 8)
  (rides_given : ℕ := 3)
  (gallons_gas : ℕ := 17)
  (good_reviews : ℕ := 2)
  (total_owed : ℕ := 226) :
  total_owed = (hours_worked * hourly_wage) + (rides_given * ride_payment) + (good_reviews * review_bonus) + (gallons_gas * 3) :=
by
  sorry

end NUMINAMATH_GPT_limo_gas_price_l2308_230850


namespace NUMINAMATH_GPT_number_of_roots_of_unity_l2308_230865

theorem number_of_roots_of_unity (n : ℕ) (z : ℂ) (c d : ℤ) (h1 : n ≥ 3) (h2 : z^n = 1) (h3 : z^3 + (c : ℂ) * z + (d : ℂ) = 0) : 
  ∃ k : ℕ, k = 4 :=
by sorry

end NUMINAMATH_GPT_number_of_roots_of_unity_l2308_230865


namespace NUMINAMATH_GPT_ax5_plus_by5_l2308_230815

-- Declare real numbers a, b, x, y
variables (a b x y : ℝ)

theorem ax5_plus_by5 (h1 : a * x + b * y = 3)
                     (h2 : a * x^2 + b * y^2 = 7)
                     (h3 : a * x^3 + b * y^3 = 6)
                     (h4 : a * x^4 + b * y^4 = 42) :
                     a * x^5 + b * y^5 = 20 := 
sorry

end NUMINAMATH_GPT_ax5_plus_by5_l2308_230815


namespace NUMINAMATH_GPT_river_depth_in_mid_may_l2308_230823

variable (D : ℕ)
variable (h1 : D + 10 - 5 + 8 = 45)

theorem river_depth_in_mid_may (h1 : D + 13 = 45) : D = 32 := by
  sorry

end NUMINAMATH_GPT_river_depth_in_mid_may_l2308_230823


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l2308_230863

noncomputable def hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (b * c) / (Real.sqrt (a^2 + b^2)) = (Real.sqrt 2 * c) / 3) : ℝ :=
  (3 * Real.sqrt 7) / 7

-- Ensure the function returns the correct eccentricity
theorem eccentricity_of_hyperbola (a b c : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (b * c) / (Real.sqrt (a^2 + b^2)) = (Real.sqrt 2 * c) / 3) : hyperbola_eccentricity a b c ha hb h = (3 * Real.sqrt 7) / 7 :=
sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l2308_230863


namespace NUMINAMATH_GPT_market_value_correct_l2308_230885

noncomputable def market_value : ℝ :=
  let dividend_income (M : ℝ) := 0.12 * M
  let fees (M : ℝ) := 0.01 * M
  let taxes (M : ℝ) := 0.15 * dividend_income M
  have yield_after_fees_and_taxes : ∀ M, 0.08 * M = dividend_income M - fees M - taxes M := 
    by sorry
  86.96

theorem market_value_correct :
  market_value = 86.96 := 
by
  sorry

end NUMINAMATH_GPT_market_value_correct_l2308_230885


namespace NUMINAMATH_GPT_paolo_sevilla_birthday_l2308_230851

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end NUMINAMATH_GPT_paolo_sevilla_birthday_l2308_230851


namespace NUMINAMATH_GPT_profit_percentage_correct_l2308_230828

-- Statement of the problem in Lean
theorem profit_percentage_correct (SP CP : ℝ) (hSP : SP = 400) (hCP : CP = 320) : 
  ((SP - CP) / CP) * 100 = 25 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_profit_percentage_correct_l2308_230828


namespace NUMINAMATH_GPT_largest_perimeter_l2308_230832

noncomputable def interior_angle (n : ℕ) : ℝ :=
  180 * (n - 2) / n

noncomputable def condition (n1 n2 n3 n4 : ℕ) : Prop :=
  2 * interior_angle n1 + interior_angle n2 + interior_angle n3 = 360

theorem largest_perimeter
  {n1 n2 n3 n4 : ℕ}
  (h : n1 = n4)
  (h_condition : condition n1 n2 n3 n4) :
  4 * n1 + 2 * n2 + 2 * n3 - 8 ≤ 22 :=
sorry

end NUMINAMATH_GPT_largest_perimeter_l2308_230832


namespace NUMINAMATH_GPT_mean_of_three_l2308_230812

theorem mean_of_three (a b c : ℝ) (h : (a + b + c + 105) / 4 = 92) : (a + b + c) / 3 = 87.7 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_three_l2308_230812


namespace NUMINAMATH_GPT_smaller_mold_radius_l2308_230853

theorem smaller_mold_radius (R : ℝ) (third_volume_sharing : ℝ) (molds_count : ℝ) (r : ℝ) 
  (hR : R = 3) 
  (h_third_volume_sharing : third_volume_sharing = 1/3) 
  (h_molds_count : molds_count = 9) 
  (h_r : (2/3) * Real.pi * r^3 = (2/3) * Real.pi / molds_count) : 
  r = 1 := 
by
  sorry

end NUMINAMATH_GPT_smaller_mold_radius_l2308_230853


namespace NUMINAMATH_GPT_sqrt_7_minus_a_l2308_230802

theorem sqrt_7_minus_a (a : ℝ) (h : a = -1) : Real.sqrt (7 - a) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_sqrt_7_minus_a_l2308_230802


namespace NUMINAMATH_GPT_base7_perfect_square_xy5z_l2308_230825

theorem base7_perfect_square_xy5z (n : ℕ) (x y z : ℕ) (hx : x ≠ 0) (hn : n = 343 * x + 49 * y + 35 + z) (hsq : ∃ m : ℕ, n = m * m) : z = 1 ∨ z = 6 :=
sorry

end NUMINAMATH_GPT_base7_perfect_square_xy5z_l2308_230825


namespace NUMINAMATH_GPT_ratio_EG_FH_l2308_230876

theorem ratio_EG_FH (EF FG EH : ℝ) (hEF : EF = 3) (hFG : FG = 7) (hEH : EH = 20) :
  (EF + FG) / (EH - EF) = 10 / 17 :=
by
  sorry

end NUMINAMATH_GPT_ratio_EG_FH_l2308_230876


namespace NUMINAMATH_GPT_preimage_of_point_l2308_230882

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

-- Define the statement of the problem
theorem preimage_of_point {x y : ℝ} (h1 : f x y = (3, 1)) : (x = 2 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_preimage_of_point_l2308_230882


namespace NUMINAMATH_GPT_sequence_formula_l2308_230867

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (diff : ∀ n, a (n + 1) - a n = 3^n) :
  ∀ n, a n = (3^n - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l2308_230867


namespace NUMINAMATH_GPT_shelves_needed_number_of_shelves_l2308_230860

-- Define the initial number of books
def initial_books : Float := 46.0

-- Define the number of additional books added by the librarian
def additional_books : Float := 10.0

-- Define the number of books each shelf can hold
def books_per_shelf : Float := 4.0

-- Define the total number of books
def total_books : Float := initial_books + additional_books

-- The mathematical proof statement for the number of shelves needed
theorem shelves_needed : Float := total_books / books_per_shelf

-- The required statement proving that the number of shelves needed is 14.0
theorem number_of_shelves : shelves_needed = 14.0 := by
  sorry

end NUMINAMATH_GPT_shelves_needed_number_of_shelves_l2308_230860


namespace NUMINAMATH_GPT_polar_eq_to_cartesian_l2308_230899

-- Define the conditions
def polar_to_cartesian_eq (ρ : ℝ) : Prop :=
  ρ = 2 → (∃ x y : ℝ, x^2 + y^2 = ρ^2)

-- State the main theorem/proof problem
theorem polar_eq_to_cartesian : polar_to_cartesian_eq 2 :=
by
  -- Proof sketch:
  --   Given ρ = 2
  --   We have ρ^2 = 4
  --   By converting to Cartesian coordinates: x^2 + y^2 = ρ^2
  --   Result: x^2 + y^2 = 4
  sorry

end NUMINAMATH_GPT_polar_eq_to_cartesian_l2308_230899


namespace NUMINAMATH_GPT_reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs_l2308_230807

theorem reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs
  (a b c h : Real)
  (area_legs : ℝ := (1 / 2) * a * b)
  (area_hypotenuse : ℝ := (1 / 2) * c * h)
  (eq_areas : a * b = c * h)
  (height_eq : h = a * b / c)
  (pythagorean_theorem : c ^ 2 = a ^ 2 + b ^ 2) :
  1 / h ^ 2 = 1 / a ^ 2 + 1 / b ^ 2 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs_l2308_230807


namespace NUMINAMATH_GPT_complement_union_of_sets_l2308_230800

variable {U M N : Set ℕ}

theorem complement_union_of_sets (h₁ : M ⊆ N) (h₂ : N ⊆ U) :
  (U \ M) ∪ (U \ N) = U \ M :=
by
  sorry

end NUMINAMATH_GPT_complement_union_of_sets_l2308_230800


namespace NUMINAMATH_GPT_neighbor_packs_l2308_230895

theorem neighbor_packs (n : ℕ) :
  let milly_balloons := 3 * 6 -- Milly and Floretta use 3 packs of their own
  let neighbor_balloons := n * 6 -- some packs of the neighbor's balloons, each contains 6 balloons
  let total_balloons := milly_balloons + neighbor_balloons -- total balloons
  -- They split balloons evenly; Milly takes 7 extra, then Floretta has 8 left
  total_balloons / 2 + 7 = total_balloons - 15
  → n = 2 := sorry

end NUMINAMATH_GPT_neighbor_packs_l2308_230895


namespace NUMINAMATH_GPT_min_value_of_squares_l2308_230806

theorem min_value_of_squares (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_min_value_of_squares_l2308_230806


namespace NUMINAMATH_GPT_circle_S_radius_properties_l2308_230887

theorem circle_S_radius_properties :
  let DE := 120
  let DF := 120
  let EF := 68
  let R_radius := 20
  let S_radius := 52 - 6 * Real.sqrt 35
  let m := 52
  let n := 6
  let k := 35
  m + n * k = 262 := by
  sorry

end NUMINAMATH_GPT_circle_S_radius_properties_l2308_230887


namespace NUMINAMATH_GPT_percentage_salt_in_mixture_l2308_230868

-- Conditions
def volume_pure_water : ℝ := 1
def volume_salt_solution : ℝ := 2
def salt_concentration : ℝ := 0.30
def total_volume : ℝ := volume_pure_water + volume_salt_solution
def amount_of_salt_in_solution : ℝ := salt_concentration * volume_salt_solution

-- Theorem
theorem percentage_salt_in_mixture :
  (amount_of_salt_in_solution / total_volume) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_salt_in_mixture_l2308_230868


namespace NUMINAMATH_GPT_function_even_and_monotonically_increasing_l2308_230817

-- Definition: Even Function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Definition: Monotonically Increasing on (0, ∞)
def is_monotonically_increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y

-- Given Function
def f (x : ℝ) : ℝ := |x| + 1

-- Theorem to prove
theorem function_even_and_monotonically_increasing :
  is_even_function f ∧ is_monotonically_increasing_on_pos f := by
  sorry

end NUMINAMATH_GPT_function_even_and_monotonically_increasing_l2308_230817


namespace NUMINAMATH_GPT_rice_on_8th_day_l2308_230838

variable (a1 : ℕ) (d : ℕ) (n : ℕ)
variable (rice_per_laborer : ℕ)

def is_arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem rice_on_8th_day (ha1 : a1 = 64) (hd : d = 7) (hr : rice_per_laborer = 3) :
  let a8 := is_arithmetic_sequence a1 d 8
  (a8 * rice_per_laborer = 339) :=
by
  sorry

end NUMINAMATH_GPT_rice_on_8th_day_l2308_230838


namespace NUMINAMATH_GPT_quadratic_has_two_real_roots_find_m_for_roots_difference_l2308_230839

theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1^2 + (2 - m) * x1 + (1 - m) = 0 ∧
                 x2^2 + (2 - m) * x2 + (1 - m) = 0 :=
by sorry

theorem find_m_for_roots_difference (m x1 x2 : ℝ) (h1 : x1^2 + (2 - m) * x1 + (1 - m) = 0) 
  (h2 : x2^2 + (2 - m) * x2 + (1 - m) = 0) (hm : m < 0) (hd : x1 - x2 = 3) : 
  m = -3 :=
by sorry

end NUMINAMATH_GPT_quadratic_has_two_real_roots_find_m_for_roots_difference_l2308_230839


namespace NUMINAMATH_GPT_water_consumption_l2308_230810

theorem water_consumption (num_cows num_goats num_pigs num_sheep : ℕ)
  (water_per_cow water_per_goat water_per_pig water_per_sheep daily_total weekly_total : ℕ)
  (h1 : num_cows = 40)
  (h2 : num_goats = 25)
  (h3 : num_pigs = 30)
  (h4 : water_per_cow = 80)
  (h5 : water_per_goat = water_per_cow / 2)
  (h6 : water_per_pig = water_per_cow / 3)
  (h7 : num_sheep = 10 * num_cows)
  (h8 : water_per_sheep = water_per_cow / 4)
  (h9 : daily_total = num_cows * water_per_cow + num_goats * water_per_goat + num_pigs * water_per_pig + num_sheep * water_per_sheep)
  (h10 : weekly_total = daily_total * 7) :
  weekly_total = 91000 := by
  sorry

end NUMINAMATH_GPT_water_consumption_l2308_230810


namespace NUMINAMATH_GPT_Kayla_score_fifth_level_l2308_230857

theorem Kayla_score_fifth_level :
  ∃ (a b c d e f : ℕ),
  a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 8 ∧ f = 17 ∧
  (b - a = 1) ∧ (c - b = 2) ∧ (d - c = 3) ∧ (e - d = 4) ∧ (f - e = 5) ∧ e = 12 :=
sorry

end NUMINAMATH_GPT_Kayla_score_fifth_level_l2308_230857


namespace NUMINAMATH_GPT_Donny_change_l2308_230891

/-- The change Donny will receive after filling up his truck. -/
theorem Donny_change
  (capacity : ℝ)
  (initial_fuel : ℝ)
  (cost_per_liter : ℝ)
  (money_available : ℝ)
  (change : ℝ) :
  capacity = 150 →
  initial_fuel = 38 →
  cost_per_liter = 3 →
  money_available = 350 →
  change = money_available - cost_per_liter * (capacity - initial_fuel) →
  change = 14 :=
by
  intros h_capacity h_initial_fuel h_cost_per_liter h_money_available h_change
  rw [h_capacity, h_initial_fuel, h_cost_per_liter, h_money_available] at h_change
  sorry

end NUMINAMATH_GPT_Donny_change_l2308_230891


namespace NUMINAMATH_GPT_div_1988_form_1989_div_1989_form_1988_l2308_230805

/-- There exists a number of the form 1989...19890... (1989 repeated several times followed by several zeros), which is divisible by 1988. -/
theorem div_1988_form_1989 (k : ℕ) : ∃ n : ℕ, (n = 1989 * 10^(4*k) ∧ n % 1988 = 0) := sorry

/-- There exists a number of the form 1988...1988 (1988 repeated several times), which is divisible by 1989. -/
theorem div_1989_form_1988 (k : ℕ) : ∃ n : ℕ, (n = 1988 * ((10^(4*k)) - 1) ∧ n % 1989 = 0) := sorry

end NUMINAMATH_GPT_div_1988_form_1989_div_1989_form_1988_l2308_230805


namespace NUMINAMATH_GPT_initial_population_l2308_230862

theorem initial_population (P : ℝ) (h1 : 0.76 * P = 3553) : P = 4678 :=
by
  sorry

end NUMINAMATH_GPT_initial_population_l2308_230862


namespace NUMINAMATH_GPT_jellybean_count_l2308_230852

def black_beans : Nat := 8
def green_beans : Nat := black_beans + 2
def orange_beans : Nat := green_beans - 1
def total_jelly_beans : Nat := black_beans + green_beans + orange_beans

theorem jellybean_count : total_jelly_beans = 27 :=
by
  -- proof steps would go here.
  sorry

end NUMINAMATH_GPT_jellybean_count_l2308_230852


namespace NUMINAMATH_GPT_cost_to_paint_cube_l2308_230884

theorem cost_to_paint_cube (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (side_length : ℝ) (total_cost : ℝ) :
  cost_per_kg = 36.50 →
  coverage_per_kg = 16 →
  side_length = 8 →
  total_cost = (6 * side_length^2 / coverage_per_kg) * cost_per_kg →
  total_cost = 876 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cost_to_paint_cube_l2308_230884


namespace NUMINAMATH_GPT_remaining_customers_l2308_230864

theorem remaining_customers (initial: ℕ) (left: ℕ) (remaining: ℕ) 
  (h1: initial = 14) (h2: left = 11) : remaining = initial - left → remaining = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_remaining_customers_l2308_230864


namespace NUMINAMATH_GPT_range_of_k_for_real_roots_l2308_230837

theorem range_of_k_for_real_roots (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 = x2 ∧ x^2 - 2*x + k = 0) ↔ k ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_k_for_real_roots_l2308_230837


namespace NUMINAMATH_GPT_sunil_total_amount_back_l2308_230845

theorem sunil_total_amount_back 
  (CI : ℝ) (P : ℝ) (r : ℝ) (t : ℕ) (total_amount : ℝ) 
  (h1 : CI = 2828.80) 
  (h2 : r = 8) 
  (h3 : t = 2) 
  (h4 : CI = P * ((1 + r / 100) ^ t - 1)) : 
  total_amount = P + CI → 
  total_amount = 19828.80 :=
by
  sorry

end NUMINAMATH_GPT_sunil_total_amount_back_l2308_230845


namespace NUMINAMATH_GPT_equilateral_is_peculiar_rt_triangle_is_peculiar_peculiar_rt_triangle_ratio_l2308_230830

-- Definition of a peculiar triangle.
def is_peculiar_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2

-- Problem 1: Proving an equilateral triangle is a peculiar triangle
theorem equilateral_is_peculiar (a : ℝ) : is_peculiar_triangle a a a :=
sorry

-- Problem 2: Proving the case when b is the hypotenuse in Rt△ABC makes it peculiar
theorem rt_triangle_is_peculiar (a b c : ℝ) (ha : a = 5 * Real.sqrt 2) (hc : c = 10) : 
  is_peculiar_triangle a b c ↔ b = Real.sqrt (c^2 + a^2) :=
sorry

-- Problem 3: Proving the ratio of the sides in a peculiar right triangle is 1 : √2 : √3
theorem peculiar_rt_triangle_ratio (a b c : ℝ) (hc : c^2 = a^2 + b^2) (hpeculiar : is_peculiar_triangle a c b) :
  (b = Real.sqrt 2 * a) ∧ (c = Real.sqrt 3 * a) :=
sorry

end NUMINAMATH_GPT_equilateral_is_peculiar_rt_triangle_is_peculiar_peculiar_rt_triangle_ratio_l2308_230830


namespace NUMINAMATH_GPT_jimmy_more_sheets_than_tommy_l2308_230888

-- Definitions for the conditions
def initial_jimmy_sheets : ℕ := 58
def initial_tommy_sheets : ℕ := initial_jimmy_sheets + 25
def ashton_gives_jimmy : ℕ := 85
def jessica_gives_jimmy : ℕ := 47
def cousin_gives_tommy : ℕ := 30
def aunt_gives_tommy : ℕ := 19

-- Lean 4 statement for the proof problem
theorem jimmy_more_sheets_than_tommy :
  let final_jimmy_sheets := initial_jimmy_sheets + ashton_gives_jimmy + jessica_gives_jimmy;
  let final_tommy_sheets := initial_tommy_sheets + cousin_gives_tommy + aunt_gives_tommy;
  final_jimmy_sheets - final_tommy_sheets = 58 :=
by sorry

end NUMINAMATH_GPT_jimmy_more_sheets_than_tommy_l2308_230888


namespace NUMINAMATH_GPT_cone_prism_volume_ratio_l2308_230804

-- Define the volumes and the ratio proof problem
theorem cone_prism_volume_ratio (r h : ℝ) (h_pos : 0 < r) (h_height : 0 < h) :
    let V_cone := (1 / 12) * π * r^2 * h
    let V_prism := 3 * r^2 * h
    (V_cone / V_prism) = (π / 36) :=
by
    -- Here we define the volumes of the cone and prism as given in the problem
    let V_cone := (1 / 12) * π * r^2 * h
    let V_prism := 3 * r^2 * h
    -- We then assert the ratio condition based on the solution
    sorry

end NUMINAMATH_GPT_cone_prism_volume_ratio_l2308_230804


namespace NUMINAMATH_GPT_remainder_when_divided_l2308_230803

/-- Given integers T, E, N, S, E', N', S'. When T is divided by E, 
the quotient is N and the remainder is S. When N is divided by E', 
the quotient is N' and the remainder is S'. Prove that the remainder 
when T is divided by E + E' is ES' + S. -/
theorem remainder_when_divided (T E N S E' N' S' : ℤ) (h1 : T = N * E + S) (h2 : N = N' * E' + S') :
  (T % (E + E')) = (E * S' + S) :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_l2308_230803


namespace NUMINAMATH_GPT_multiples_of_3_or_5_but_not_6_l2308_230844

theorem multiples_of_3_or_5_but_not_6 (n : ℕ) (h1 : n ≤ 150) :
  (∃ m : ℕ, m ≤ 150 ∧ ((m % 3 = 0 ∨ m % 5 = 0) ∧ m % 6 ≠ 0)) ↔ n = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_multiples_of_3_or_5_but_not_6_l2308_230844


namespace NUMINAMATH_GPT_range_of_a_l2308_230890

theorem range_of_a (a : ℝ) :
  (a > 0 ∧ (∃ x, x^2 - 4 * a * x + 3 * a^2 < 0)) →
  (∃ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2 * x - 8 > 0) →
  (2 < a ∧ a ≤ 2) := sorry

end NUMINAMATH_GPT_range_of_a_l2308_230890


namespace NUMINAMATH_GPT_min_value_a_l2308_230827

noncomputable def equation_has_real_solutions (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, 9 * x1 - (4 + a) * 3 * x1 + 4 = 0 ∧ 9 * x2 - (4 + a) * 3 * x2 + 4 = 0

theorem min_value_a : ∀ a : ℝ, 
  equation_has_real_solutions a → 
  a ≥ 2 :=
sorry

end NUMINAMATH_GPT_min_value_a_l2308_230827


namespace NUMINAMATH_GPT_percentage_increase_in_surface_area_l2308_230811

variable (a : ℝ)

theorem percentage_increase_in_surface_area (ha : a > 0) :
  let original_surface_area := 6 * a^2
  let new_edge_length := 1.5 * a
  let new_surface_area := 6 * (new_edge_length)^2
  let area_increase := new_surface_area - original_surface_area
  let percentage_increase := (area_increase / original_surface_area) * 100
  percentage_increase = 125 := 
by 
  let original_surface_area := 6 * a^2
  let new_edge_length := 1.5 * a
  let new_surface_area := 6 * (new_edge_length)^2
  let area_increase := new_surface_area - original_surface_area
  let percentage_increase := (area_increase / original_surface_area) * 100
  sorry

end NUMINAMATH_GPT_percentage_increase_in_surface_area_l2308_230811


namespace NUMINAMATH_GPT_no_solution_A_eq_B_l2308_230861

theorem no_solution_A_eq_B (a : ℝ) (h1 : a = 2 * a) (h2 : a ≠ 2) : false := by
  sorry

end NUMINAMATH_GPT_no_solution_A_eq_B_l2308_230861


namespace NUMINAMATH_GPT_ratio_child_to_jane_babysit_l2308_230819

-- Definitions of the conditions
def jane_current_age : ℕ := 32
def years_since_jane_stopped_babysitting : ℕ := 10
def oldest_person_current_age : ℕ := 24

-- Derived definitions
def jane_age_when_stopped : ℕ := jane_current_age - years_since_jane_stopped_babysitting
def oldest_person_age_when_jane_stopped : ℕ := oldest_person_current_age - years_since_jane_stopped_babysitting

-- Statement of the problem to be proven in Lean 4
theorem ratio_child_to_jane_babysit :
  (oldest_person_age_when_jane_stopped : ℚ) / (jane_age_when_stopped : ℚ) = 7 / 11 :=
by
  sorry

end NUMINAMATH_GPT_ratio_child_to_jane_babysit_l2308_230819


namespace NUMINAMATH_GPT_sum_of_x_y_z_l2308_230881

theorem sum_of_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 2 * y) : x + y + z = 10 * x := by
  sorry

end NUMINAMATH_GPT_sum_of_x_y_z_l2308_230881


namespace NUMINAMATH_GPT_exists_right_triangle_area_twice_hypotenuse_l2308_230813

theorem exists_right_triangle_area_twice_hypotenuse : 
  ∃ (a : ℝ), a ≠ 0 ∧ (a^2 / 2 = 2 * a * Real.sqrt 2) ∧ (a = 4 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_exists_right_triangle_area_twice_hypotenuse_l2308_230813


namespace NUMINAMATH_GPT_number_of_zeros_g_l2308_230816

variable (f : ℝ → ℝ)
variable (hf_cont : continuous f)
variable (hf_diff : differentiable ℝ f)
variable (h_condition : ∀ x : ℝ, x * (deriv f x) + f x > 0)

theorem number_of_zeros_g (hg : ∀ x : ℝ, x > 0 → x * f x + 1 = 0 → false) : 
    ∀ x : ℝ , x > 0 → ¬ (x * f x + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_number_of_zeros_g_l2308_230816


namespace NUMINAMATH_GPT_perimeter_of_new_rectangle_l2308_230808

-- Definitions based on conditions
def side_of_square : ℕ := 8
def length_of_rectangle : ℕ := 8
def breadth_of_rectangle : ℕ := 4

-- Perimeter calculation
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Formal statement of the problem
theorem perimeter_of_new_rectangle :
  perimeter (side_of_square + length_of_rectangle) side_of_square = 48 :=
  by sorry

end NUMINAMATH_GPT_perimeter_of_new_rectangle_l2308_230808


namespace NUMINAMATH_GPT_rectangle_area_l2308_230856

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2308_230856


namespace NUMINAMATH_GPT_vacation_months_away_l2308_230829

theorem vacation_months_away (total_savings : ℕ) (pay_per_check : ℕ) (checks_per_month : ℕ) :
  total_savings = 3000 → pay_per_check = 100 → checks_per_month = 2 → 
  total_savings / pay_per_check / checks_per_month = 15 :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_vacation_months_away_l2308_230829


namespace NUMINAMATH_GPT_line_through_PQ_l2308_230842

theorem line_through_PQ (x y : ℝ) (P Q : ℝ × ℝ)
  (hP : P = (3, 2)) (hQ : Q = (1, 4))
  (h_line : ∀ t, (x, y) = (1 - t) • P + t • Q):
  y = x - 2 :=
by
  have h1 : P = ((3 : ℝ), (2 : ℝ)) := hP
  have h2 : Q = ((1 : ℝ), (4 : ℝ)) := hQ
  sorry

end NUMINAMATH_GPT_line_through_PQ_l2308_230842


namespace NUMINAMATH_GPT_point_B_third_quadrant_l2308_230833

theorem point_B_third_quadrant (m n : ℝ) (hm : m < 0) (hn : n < 0) :
  (-m * n < 0) ∧ (m < 0) :=
by
  sorry

end NUMINAMATH_GPT_point_B_third_quadrant_l2308_230833


namespace NUMINAMATH_GPT_hyperbola_asymptotes_angle_l2308_230874

theorem hyperbola_asymptotes_angle {a b : ℝ} (h₁ : a > b) 
  (h₂ : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) 
  (h₃ : ∀ θ : ℝ, θ = Real.pi / 4) : a / b = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_angle_l2308_230874


namespace NUMINAMATH_GPT_teams_B_and_C_worked_together_days_l2308_230898

def workload_project_B := 5/4
def time_team_A_project_A := 20
def time_team_B_project_A := 24
def time_team_C_project_A := 30

def equation1 (x y : ℕ) : Prop := 
  3 * x + 5 * y = 60

def equation2 (x y : ℕ) : Prop := 
  9 * x + 5 * y = 150

theorem teams_B_and_C_worked_together_days (x : ℕ) (y : ℕ) :
  equation1 x y ∧ equation2 x y → x = 15 := 
by 
  sorry

end NUMINAMATH_GPT_teams_B_and_C_worked_together_days_l2308_230898


namespace NUMINAMATH_GPT_ticket_price_l2308_230878

theorem ticket_price (Olivia_money : ℕ) (Nigel_money : ℕ) (left_money : ℕ) (total_tickets : ℕ)
  (h1 : Olivia_money = 112)
  (h2 : Nigel_money = 139)
  (h3 : left_money = 83)
  (h4 : total_tickets = 6) :
  (Olivia_money + Nigel_money - left_money) / total_tickets = 28 :=
by
  sorry

end NUMINAMATH_GPT_ticket_price_l2308_230878


namespace NUMINAMATH_GPT_apple_box_weights_l2308_230875

theorem apple_box_weights (a b c d : ℤ) 
  (h1 : a + b + c = 70)
  (h2 : a + b + d = 80)
  (h3 : a + c + d = 73)
  (h4 : b + c + d = 77) : 
  a = 23 ∧ b = 27 ∧ c = 20 ∧ d = 30 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end NUMINAMATH_GPT_apple_box_weights_l2308_230875


namespace NUMINAMATH_GPT_remainder_product_mod_eq_l2308_230840

theorem remainder_product_mod_eq (n : ℤ) :
  ((12 - 2 * n) * (n + 5)) % 11 = (-2 * n^2 + 2 * n + 5) % 11 := by
  sorry

end NUMINAMATH_GPT_remainder_product_mod_eq_l2308_230840


namespace NUMINAMATH_GPT_find_n_from_A_k_l2308_230872

theorem find_n_from_A_k (n : ℕ) (A : ℕ → ℕ) (h1 : A 1 = Int.natAbs (n + 1))
  (h2 : ∀ k : ℕ, k > 0 → A k = Int.natAbs (n + (2 * k - 1)))
  (h3 : A 100 = 2005) : n = 1806 :=
sorry

end NUMINAMATH_GPT_find_n_from_A_k_l2308_230872


namespace NUMINAMATH_GPT_number_of_chicks_is_8_l2308_230896

-- Define the number of total chickens
def total_chickens : ℕ := 15

-- Define the number of hens
def hens : ℕ := 3

-- Define the number of roosters
def roosters : ℕ := total_chickens - hens

-- Define the number of chicks
def chicks : ℕ := roosters - 4

-- State the main theorem
theorem number_of_chicks_is_8 : chicks = 8 := 
by
  -- the solution follows from the given definitions and conditions
  sorry

end NUMINAMATH_GPT_number_of_chicks_is_8_l2308_230896


namespace NUMINAMATH_GPT_phase_shift_of_cosine_l2308_230859

theorem phase_shift_of_cosine (a b c : ℝ) (h : c = -π / 4 ∧ b = 3) :
  (-c / b) = π / 12 :=
by
  sorry

end NUMINAMATH_GPT_phase_shift_of_cosine_l2308_230859


namespace NUMINAMATH_GPT_kate_change_is_correct_l2308_230873

-- Define prices of items
def gum_price : ℝ := 0.89
def chocolate_price : ℝ := 1.25
def chips_price : ℝ := 2.49

-- Define sales tax rate
def tax_rate : ℝ := 0.06

-- Define the total money Kate gave to the clerk
def payment : ℝ := 10.00

-- Define total cost of items before tax
def total_before_tax := gum_price + chocolate_price + chips_price

-- Define the sales tax
def sales_tax := tax_rate * total_before_tax

-- Define the correct answer for total cost
def total_cost := total_before_tax + sales_tax

-- Define the correct amount of change Kate should get back
def change := payment - total_cost

theorem kate_change_is_correct : abs (change - 5.09) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_kate_change_is_correct_l2308_230873


namespace NUMINAMATH_GPT_probability_no_correct_letter_for_7_envelopes_l2308_230847

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * factorial (n - 1)

def derangement (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement (n - 1) + derangement (n - 2))

noncomputable def probability_no_correct_letter (n : ℕ) : ℚ :=
  derangement n / factorial n

theorem probability_no_correct_letter_for_7_envelopes :
  probability_no_correct_letter 7 = 427 / 1160 :=
by sorry

end NUMINAMATH_GPT_probability_no_correct_letter_for_7_envelopes_l2308_230847


namespace NUMINAMATH_GPT_geometric_sequence_n_value_l2308_230820

theorem geometric_sequence_n_value
  (a : ℕ → ℝ) (n : ℕ)
  (h1 : a 1 * a 2 * a 3 = 4)
  (h2 : a 4 * a 5 * a 6 = 12)
  (h3 : a (n-1) * a n * a (n+1) = 324)
  (h_geometric : ∃ r > 0, ∀ i, a (i+1) = a i * r) :
  n = 14 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_n_value_l2308_230820


namespace NUMINAMATH_GPT_lcm_pairs_count_l2308_230871

noncomputable def distinct_pairs_lcm_count : ℕ :=
  sorry

theorem lcm_pairs_count :
  distinct_pairs_lcm_count = 1502 :=
  sorry

end NUMINAMATH_GPT_lcm_pairs_count_l2308_230871


namespace NUMINAMATH_GPT_tan_six_theta_eq_l2308_230826

theorem tan_six_theta_eq (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (6 * θ) = 21 / 8 :=
by
  sorry

end NUMINAMATH_GPT_tan_six_theta_eq_l2308_230826


namespace NUMINAMATH_GPT_total_donation_l2308_230848

theorem total_donation {carwash_proceeds bake_sale_proceeds mowing_lawn_proceeds : ℝ}
    (hc : carwash_proceeds = 100)
    (hb : bake_sale_proceeds = 80)
    (hl : mowing_lawn_proceeds = 50)
    (carwash_donation : ℝ := 0.9 * carwash_proceeds)
    (bake_sale_donation : ℝ := 0.75 * bake_sale_proceeds)
    (mowing_lawn_donation : ℝ := 1.0 * mowing_lawn_proceeds) :
    carwash_donation + bake_sale_donation + mowing_lawn_donation = 200 := by
  sorry

end NUMINAMATH_GPT_total_donation_l2308_230848


namespace NUMINAMATH_GPT_positive_integer_solutions_l2308_230809

theorem positive_integer_solutions :
  ∀ m n : ℕ, 0 < m ∧ 0 < n ∧ 3^m - 2^n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 3) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_l2308_230809


namespace NUMINAMATH_GPT_smallest_positive_period_l2308_230849

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem smallest_positive_period (ω : ℝ) (hω : ω > 0)
  (H : ∀ x1 x2 : ℝ, abs (f ω x1 - f ω x2) = 2 → abs (x1 - x2) = Real.pi / 2) :
  ∃ T > 0, T = Real.pi ∧ (∀ x : ℝ, f ω (x + T) = f ω x) := 
sorry

end NUMINAMATH_GPT_smallest_positive_period_l2308_230849


namespace NUMINAMATH_GPT_magnitude_of_2a_minus_b_l2308_230843

/-- Definition of the vectors a and b --/
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 3)

/-- Proposition stating the magnitude of 2a - b --/
theorem magnitude_of_2a_minus_b : 
  (Real.sqrt ((2 * a.1 - b.1) ^ 2 + (2 * a.2 - b.2) ^ 2)) = Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_2a_minus_b_l2308_230843


namespace NUMINAMATH_GPT_better_offer_saves_800_l2308_230886

theorem better_offer_saves_800 :
  let initial_order := 20000
  let discount1 (x : ℝ) := x * 0.70 * 0.90 - 800
  let discount2 (x : ℝ) := x * 0.75 * 0.80 - 1000
  discount1 initial_order - discount2 initial_order = 800 :=
by
  sorry

end NUMINAMATH_GPT_better_offer_saves_800_l2308_230886


namespace NUMINAMATH_GPT_cost_price_percentage_l2308_230866

theorem cost_price_percentage (MP CP SP : ℝ) (h1 : SP = 0.88 * MP) (h2 : SP = 1.375 * CP) :
  (CP / MP) * 100 = 64 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_percentage_l2308_230866


namespace NUMINAMATH_GPT_range_of_expression_l2308_230831

theorem range_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
  0 < (x * y + y * z + z * x - 2 * x * y * z) ∧ (x * y + y * z + z * x - 2 * x * y * z) ≤ 7 / 27 := by
  sorry

end NUMINAMATH_GPT_range_of_expression_l2308_230831


namespace NUMINAMATH_GPT_fourth_powers_sum_is_8432_l2308_230894

def sum_fourth_powers (x y : ℝ) : ℝ := x^4 + y^4

theorem fourth_powers_sum_is_8432 (x y : ℝ) (h₁ : x + y = 10) (h₂ : x * y = 4) : 
  sum_fourth_powers x y = 8432 :=
by
  sorry

end NUMINAMATH_GPT_fourth_powers_sum_is_8432_l2308_230894
