import Mathlib

namespace NUMINAMATH_CALUDE_lcm_ratio_sum_l670_67088

theorem lcm_ratio_sum (a b c : ℕ+) : 
  a.val * 3 = b.val * 2 →
  b.val * 7 = c.val * 3 →
  Nat.lcm a.val (Nat.lcm b.val c.val) = 126 →
  a.val + b.val + c.val = 216 := by
sorry

end NUMINAMATH_CALUDE_lcm_ratio_sum_l670_67088


namespace NUMINAMATH_CALUDE_spaghetti_tortellini_ratio_l670_67079

theorem spaghetti_tortellini_ratio : 
  ∀ (total_students : ℕ) 
    (spaghetti_students tortellini_students : ℕ) 
    (grade_levels : ℕ),
  total_students = 800 →
  spaghetti_students = 300 →
  tortellini_students = 120 →
  grade_levels = 4 →
  (spaghetti_students / grade_levels) / (tortellini_students / grade_levels) = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_spaghetti_tortellini_ratio_l670_67079


namespace NUMINAMATH_CALUDE_walkway_time_against_l670_67009

/-- Represents the scenario of a person walking on a moving walkway. -/
structure WalkwayScenario where
  length : ℝ  -- Length of the walkway in meters
  time_with : ℝ  -- Time to walk with the walkway in seconds
  time_stationary : ℝ  -- Time to walk when the walkway is not moving in seconds

/-- Calculates the time to walk against the walkway given a WalkwayScenario. -/
def time_against (scenario : WalkwayScenario) : ℝ :=
  sorry

/-- Theorem stating that for the given scenario, the time to walk against the walkway is 120 seconds. -/
theorem walkway_time_against 
  (scenario : WalkwayScenario)
  (h1 : scenario.length = 60)
  (h2 : scenario.time_with = 30)
  (h3 : scenario.time_stationary = 48) :
  time_against scenario = 120 :=
sorry

end NUMINAMATH_CALUDE_walkway_time_against_l670_67009


namespace NUMINAMATH_CALUDE_propositions_truth_values_l670_67055

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the lines and planes
variable (m n : Line) (α β : Plane)

-- State the theorem
theorem propositions_truth_values :
  -- Proposition ① is false
  ¬(∀ m n α β, parallelLP m α → parallelLP n β → parallelPP α β → parallel m n) ∧
  -- Proposition ② is true
  (∀ m n α β, parallel m n → contains α m → perpendicularLP n β → perpendicularPP α β) ∧
  -- Proposition ③ is false
  ¬(∀ m n α β, intersect α β m → parallel m n → parallelLP n α ∧ parallelLP n β) ∧
  -- Proposition ④ is true
  (∀ m n α β, perpendicular m n → intersect α β m → perpendicularLP n α ∨ perpendicularLP n β) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_values_l670_67055


namespace NUMINAMATH_CALUDE_square_sum_and_reciprocal_l670_67034

theorem square_sum_and_reciprocal (x : ℝ) (h : x + (1/x) = 2) : x^2 + (1/x^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_and_reciprocal_l670_67034


namespace NUMINAMATH_CALUDE_function_constraint_l670_67011

theorem function_constraint (a : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → a * x + 6 ≤ 10) ↔ (a = -4 ∨ a = 2 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_constraint_l670_67011


namespace NUMINAMATH_CALUDE_number_percentage_equality_l670_67057

theorem number_percentage_equality : ∃ x : ℚ, (3 / 10 : ℚ) * x = (2 / 10 : ℚ) * 40 ∧ x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_equality_l670_67057


namespace NUMINAMATH_CALUDE_unique_solution_to_system_l670_67027

theorem unique_solution_to_system :
  ∃! (x y z : ℝ), 
    x^2 - 23*y + 66*z + 612 = 0 ∧
    y^2 + 62*x - 20*z + 296 = 0 ∧
    z^2 - 22*x + 67*y + 505 = 0 ∧
    x = -20 ∧ y = -22 ∧ z = -23 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_system_l670_67027


namespace NUMINAMATH_CALUDE_sqrt_27_minus_3tan60_plus_power_equals_1_l670_67098

theorem sqrt_27_minus_3tan60_plus_power_equals_1 :
  Real.sqrt 27 - 3 * Real.tan (60 * π / 180) + (π - Real.sqrt 2) ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_minus_3tan60_plus_power_equals_1_l670_67098


namespace NUMINAMATH_CALUDE_total_stamps_l670_67062

theorem total_stamps (a b : ℕ) (h1 : a * 4 = b * 5) (h2 : (a - 5) * 5 = (b + 5) * 4) : a + b = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_stamps_l670_67062


namespace NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l670_67073

theorem complex_fraction_equals_neg_i : (1 - Complex.I) / (1 + Complex.I) = -Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l670_67073


namespace NUMINAMATH_CALUDE_inscribed_square_area_ratio_l670_67018

theorem inscribed_square_area_ratio : 
  ∀ (large_square_side : ℝ) (inscribed_square_side : ℝ),
    large_square_side = 4 →
    inscribed_square_side = 2 →
    (inscribed_square_side^2) / (large_square_side^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_ratio_l670_67018


namespace NUMINAMATH_CALUDE_ben_win_probability_l670_67086

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 5 / 8) 
  (h2 : ¬ ∃ (draw_prob : ℚ), draw_prob ≠ 0) : 
  1 - lose_prob = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_ben_win_probability_l670_67086


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_fourth_l670_67083

theorem r_fourth_plus_inverse_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_fourth_l670_67083


namespace NUMINAMATH_CALUDE_equation_proof_l670_67005

theorem equation_proof : Real.sqrt (72 * 2) + (5568 / 87) ^ (1/3) = Real.sqrt 256 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l670_67005


namespace NUMINAMATH_CALUDE_xy_value_l670_67048

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 16)
  (h2 : (16:ℝ)^(x+y) / (4:ℝ)^(7*y) = 256) : 
  x * y = 48 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l670_67048


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l670_67094

def trailing_zeros (n : ℕ) : ℕ := sorry

theorem product_trailing_zeros : trailing_zeros (45 * 160 * 7) = 2 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l670_67094


namespace NUMINAMATH_CALUDE_jenny_essay_copies_l670_67035

/-- Represents the problem of determining how many copies Jenny wants to print -/
theorem jenny_essay_copies : 
  let cost_per_page : ℚ := 1 / 10
  let essay_pages : ℕ := 25
  let num_pens : ℕ := 7
  let cost_per_pen : ℚ := 3 / 2
  let payment : ℕ := 2 * 20
  let change : ℕ := 12
  
  let total_spent : ℚ := payment - change
  let pen_cost : ℚ := num_pens * cost_per_pen
  let printing_cost : ℚ := total_spent - pen_cost
  let cost_per_copy : ℚ := cost_per_page * essay_pages
  let num_copies : ℚ := printing_cost / cost_per_copy
  
  num_copies = 7 := by sorry

end NUMINAMATH_CALUDE_jenny_essay_copies_l670_67035


namespace NUMINAMATH_CALUDE_lisa_flight_distance_l670_67004

/-- Given a speed of 32 miles per hour and a time of 8 hours, 
    the distance traveled is equal to 256 miles. -/
theorem lisa_flight_distance : 
  let speed : ℝ := 32
  let time : ℝ := 8
  let distance := speed * time
  distance = 256 := by sorry

end NUMINAMATH_CALUDE_lisa_flight_distance_l670_67004


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l670_67077

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 36 * Real.pi) :
  A = Real.pi * r^2 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l670_67077


namespace NUMINAMATH_CALUDE_correct_equation_by_moving_digit_l670_67082

theorem correct_equation_by_moving_digit : ∃ (a b c : ℕ), 
  (a = 101 ∧ b = 10 ∧ c = 2) ∧ 
  (a - b^c = 1) ∧
  (∃ (x y : ℕ), x * 10 + y = 102 ∧ (x = 10 ∧ y = c)) :=
by
  sorry

end NUMINAMATH_CALUDE_correct_equation_by_moving_digit_l670_67082


namespace NUMINAMATH_CALUDE_miles_driven_l670_67031

-- Define the efficiency of the car in miles per gallon
def miles_per_gallon : ℝ := 45

-- Define the price of gas per gallon
def price_per_gallon : ℝ := 5

-- Define the amount spent on gas
def amount_spent : ℝ := 25

-- Theorem to prove
theorem miles_driven : 
  miles_per_gallon * (amount_spent / price_per_gallon) = 225 := by
sorry

end NUMINAMATH_CALUDE_miles_driven_l670_67031


namespace NUMINAMATH_CALUDE_fly_distance_l670_67042

/-- The distance flown by a fly between two approaching pedestrians -/
theorem fly_distance (d : ℝ) (v_ped : ℝ) (v_fly : ℝ) (h1 : d > 0) (h2 : v_ped > 0) (h3 : v_fly > 0) :
  let t := d / (2 * v_ped)
  v_fly * t = v_fly * d / (2 * v_ped) := by sorry

#check fly_distance

end NUMINAMATH_CALUDE_fly_distance_l670_67042


namespace NUMINAMATH_CALUDE_compound_interest_proof_l670_67064

/-- Calculates the final amount after compound interest --/
def final_amount (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Proves that $30,000 increased by 55% annually for 2 years results in $72,075 --/
theorem compound_interest_proof :
  let principal : ℝ := 30000
  let rate : ℝ := 0.55
  let years : ℕ := 2
  final_amount principal rate years = 72075 := by sorry

end NUMINAMATH_CALUDE_compound_interest_proof_l670_67064


namespace NUMINAMATH_CALUDE_text_files_deleted_l670_67061

theorem text_files_deleted (pictures_deleted : ℕ) (songs_deleted : ℕ) (total_deleted : ℕ) :
  pictures_deleted = 2 →
  songs_deleted = 8 →
  total_deleted = 17 →
  total_deleted = pictures_deleted + songs_deleted + (total_deleted - pictures_deleted - songs_deleted) →
  total_deleted - pictures_deleted - songs_deleted = 7 :=
by sorry

end NUMINAMATH_CALUDE_text_files_deleted_l670_67061


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l670_67068

/-- 
Given two points A and B in the Cartesian coordinate system,
where A has coordinates (2, m) and B has coordinates (n, -1),
if A and B are symmetric with respect to the x-axis,
then m + n = 3.
-/
theorem symmetric_points_sum (m n : ℝ) : 
  (2 : ℝ) = n ∧ m = -(-1 : ℝ) → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l670_67068


namespace NUMINAMATH_CALUDE_f_of_tan_squared_l670_67067

noncomputable def f (x : ℝ) : ℝ := 1 / (((x / (x - 1)) - 1) / (x / (x - 1)))^2

theorem f_of_tan_squared (t : ℝ) (h : 0 ≤ t ∧ t ≤ π/4) :
  f (Real.tan t)^2 = (Real.cos (2*t) / Real.sin t^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_tan_squared_l670_67067


namespace NUMINAMATH_CALUDE_g_value_at_half_l670_67066

/-- Given a function g : ℝ → ℝ satisfying the equation
    g(x) - 3g(1/x) = 4^x + e^x for all x ≠ 0,
    prove that g(1/2) = (3e^2 - 13√e + 82) / 8 -/
theorem g_value_at_half (g : ℝ → ℝ) 
    (h : ∀ x : ℝ, x ≠ 0 → g x - 3 * g (1/x) = 4^x + Real.exp x) : 
  g (1/2) = (3 * Real.exp 2 - 13 * Real.sqrt (Real.exp 1) + 82) / 8 := by
sorry

end NUMINAMATH_CALUDE_g_value_at_half_l670_67066


namespace NUMINAMATH_CALUDE_max_value_product_l670_67060

theorem max_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5*x + 2*y < 50) :
  xy*(50 - 5*x - 2*y) ≤ 125000/432 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 5*x₀ + 2*y₀ < 50 ∧ x₀*y₀*(50 - 5*x₀ - 2*y₀) = 125000/432 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l670_67060


namespace NUMINAMATH_CALUDE_words_with_e_count_l670_67039

/-- The number of letters in the alphabet we're using -/
def n : ℕ := 5

/-- The length of the words we're creating -/
def k : ℕ := 4

/-- The number of letters in the alphabet excluding E -/
def m : ℕ := 4

/-- The number of 4-letter words that can be made from the letters A, B, C, D, and E, 
    allowing repetition and using the letter E at least once -/
def words_with_e : ℕ := n^k - m^k

theorem words_with_e_count : words_with_e = 369 := by sorry

end NUMINAMATH_CALUDE_words_with_e_count_l670_67039


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l670_67020

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 7).sum (λ k => (Nat.choose 6 k) * (-2)^(6 - k) * x^k) = 
  240 * x^2 + (Finset.range 7).sum (λ k => if k ≠ 2 then (Nat.choose 6 k) * (-2)^(6 - k) * x^k else 0) := by
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l670_67020


namespace NUMINAMATH_CALUDE_percent_of_percent_l670_67085

theorem percent_of_percent (y : ℝ) : 0.21 * y = 0.3 * (0.7 * y) := by sorry

end NUMINAMATH_CALUDE_percent_of_percent_l670_67085


namespace NUMINAMATH_CALUDE_womens_average_age_l670_67080

theorem womens_average_age (n : ℕ) (A : ℝ) (age_increase : ℝ) (man1_age man2_age : ℕ) :
  n = 8 ∧ age_increase = 2 ∧ man1_age = 20 ∧ man2_age = 28 →
  ∃ W1 W2 : ℝ,
    W1 + W2 = n * (A + age_increase) - (n * A - man1_age - man2_age) ∧
    (W1 + W2) / 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_womens_average_age_l670_67080


namespace NUMINAMATH_CALUDE_parametric_to_regular_equation_l670_67002

theorem parametric_to_regular_equation 
  (t : ℝ) (ht : t ≠ 0) 
  (x : ℝ) (hx : x = t + 1/t) 
  (y : ℝ) (hy : y = t^2 + 1/t^2) : 
  x^2 - y - 2 = 0 ∧ y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_regular_equation_l670_67002


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l670_67025

theorem employee_pay_percentage (total_pay B_pay : ℝ) (h1 : total_pay = 550) (h2 : B_pay = 249.99999999999997) :
  let A_pay := total_pay - B_pay
  (A_pay / B_pay) * 100 = 120 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l670_67025


namespace NUMINAMATH_CALUDE_mean_temperature_l670_67029

def temperatures : List ℝ := [82, 80, 83, 88, 84, 90, 92, 85, 89, 90]

theorem mean_temperature (temps := temperatures) : 
  (temps.sum / temps.length : ℝ) = 86.3 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l670_67029


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l670_67099

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 3*y + m = 0 → y = x) → 
  m = 9/4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l670_67099


namespace NUMINAMATH_CALUDE_irreducible_fraction_l670_67000

theorem irreducible_fraction (n : ℕ) : 
  (Nat.gcd (21 * n + 4) (14 * n + 1) = 1) ↔ (n % 5 ≠ 1) := by sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l670_67000


namespace NUMINAMATH_CALUDE_bills_omelet_time_l670_67084

/-- The time it takes to prepare and cook omelets -/
def total_time (pepper_chop_time onion_chop_time cheese_grate_time assemble_cook_time : ℕ) 
               (num_peppers num_onions num_omelets : ℕ) : ℕ :=
  (pepper_chop_time * num_peppers) + 
  (onion_chop_time * num_onions) + 
  ((cheese_grate_time + assemble_cook_time) * num_omelets)

/-- Theorem stating that Bill's total preparation and cooking time for five omelets is 50 minutes -/
theorem bills_omelet_time : 
  total_time 3 4 1 5 4 2 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bills_omelet_time_l670_67084


namespace NUMINAMATH_CALUDE_problem_statement_l670_67032

theorem problem_statement (x y : ℕ) (h1 : y > 3) 
  (h2 : x^2 + y^4 = 2*((x-6)^2 + (y+1)^2)) : 
  x^2 + y^4 = 1994 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l670_67032


namespace NUMINAMATH_CALUDE_triangle_area_tripled_sides_l670_67095

/-- Given a triangle, prove that tripling its sides multiplies its area by 9 -/
theorem triangle_area_tripled_sides (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let s' := ((3 * a) + (3 * b) + (3 * c)) / 2
  let area' := Real.sqrt (s' * (s' - 3 * a) * (s' - 3 * b) * (s' - 3 * c))
  area' = 9 * area := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_tripled_sides_l670_67095


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_squared_l670_67016

theorem largest_x_sqrt_3x_eq_5x_squared :
  let f : ℝ → ℝ := λ x => Real.sqrt (3 * x) - 5 * x^2
  ∃ (max_x : ℝ), max_x = (3/25)^(1/3) ∧
    (∀ x : ℝ, f x = 0 → x ≤ max_x) ∧
    f max_x = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_squared_l670_67016


namespace NUMINAMATH_CALUDE_angle_xpy_is_45_deg_l670_67017

/-- A rectangle WXYZ with a point P on side WZ -/
structure RectangleWithPoint where
  /-- Length of side WZ -/
  wz : ℝ
  /-- Length of side XY -/
  xy : ℝ
  /-- Distance from W to P -/
  wp : ℝ
  /-- Angle WPY in radians -/
  angle_wpy : ℝ
  /-- Angle XPY in radians -/
  angle_xpy : ℝ
  /-- WZ is positive -/
  wz_pos : 0 < wz
  /-- XY is positive -/
  xy_pos : 0 < xy
  /-- P is on WZ -/
  wp_le_wz : 0 ≤ wp ∧ wp ≤ wz
  /-- Sine ratio condition -/
  sine_ratio : Real.sin angle_wpy / Real.sin angle_xpy = 2

/-- Theorem: If WZ = 8, XY = 4, and the sine ratio condition holds, then ∠XPY = 45° -/
theorem angle_xpy_is_45_deg (r : RectangleWithPoint) 
  (h1 : r.wz = 8) (h2 : r.xy = 4) : r.angle_xpy = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_xpy_is_45_deg_l670_67017


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l670_67087

/-- The magnitude of the vector corresponding to the complex number 2/(1+i) is √2 -/
theorem magnitude_of_complex_fraction : Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l670_67087


namespace NUMINAMATH_CALUDE_henry_jill_age_ratio_l670_67023

/-- Proves that the ratio of Henry's age to Jill's age 11 years ago is 2:1 -/
theorem henry_jill_age_ratio : 
  ∀ (henry_age jill_age : ℕ),
  henry_age + jill_age = 40 →
  henry_age = 23 →
  jill_age = 17 →
  ∃ (k : ℕ), (henry_age - 11) = k * (jill_age - 11) →
  (henry_age - 11) / (jill_age - 11) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_henry_jill_age_ratio_l670_67023


namespace NUMINAMATH_CALUDE_expected_worth_unfair_coin_l670_67059

/-- The expected worth of an unfair coin flip -/
theorem expected_worth_unfair_coin : 
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let gain_heads : ℤ := 5
  let loss_tails : ℤ := 6
  p_heads * gain_heads - p_tails * loss_tails = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_unfair_coin_l670_67059


namespace NUMINAMATH_CALUDE_unique_equilateral_hyperbola_l670_67091

/-- An equilateral hyperbola passing through (3, -1) with axes of symmetry on coordinate axes -/
def equilateral_hyperbola (a : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 - y^2 = a ∧ (x = 3 ∧ y = -1)

/-- The unique value of 'a' for which the hyperbola is equilateral and passes through (3, -1) -/
theorem unique_equilateral_hyperbola :
  ∃! a : ℝ, equilateral_hyperbola a ∧ a = 8 := by sorry

end NUMINAMATH_CALUDE_unique_equilateral_hyperbola_l670_67091


namespace NUMINAMATH_CALUDE_composition_theorem_l670_67052

def f (x : ℝ) : ℝ := 1 - 2 * x
def g (x : ℝ) : ℝ := x^2 + 3

theorem composition_theorem :
  (∀ x : ℝ, f (g x) = -2 * x^2 - 5) ∧
  (∀ x : ℝ, g (f x) = 4 * x^2 - 4 * x + 4) := by
sorry

end NUMINAMATH_CALUDE_composition_theorem_l670_67052


namespace NUMINAMATH_CALUDE_minimize_constant_term_l670_67043

/-- The function representing the constant term in the expansion -/
def f (a : ℝ) : ℝ := a^3 - 9*a

/-- Theorem stating that √3 minimizes f(a) for a > 0 -/
theorem minimize_constant_term (a : ℝ) (h : a > 0) :
  f a ≥ f (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_minimize_constant_term_l670_67043


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l670_67071

theorem discount_percentage_proof (jacket_price shirt_price shoes_price : ℝ)
  (jacket_discount shirt_discount shoes_discount : ℝ) :
  jacket_price = 120 ∧ shirt_price = 60 ∧ shoes_price = 90 ∧
  jacket_discount = 0.30 ∧ shirt_discount = 0.50 ∧ shoes_discount = 0.25 →
  (jacket_price * jacket_discount + shirt_price * shirt_discount + shoes_price * shoes_discount) /
  (jacket_price + shirt_price + shoes_price) = 0.328 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l670_67071


namespace NUMINAMATH_CALUDE_second_number_in_first_set_l670_67081

theorem second_number_in_first_set (X : ℝ) : 
  ((20 + X + 60) / 3 = (10 + 50 + 45) / 3 + 5) → X = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_number_in_first_set_l670_67081


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l670_67040

theorem quadratic_inequality_solution (x : ℝ) : 
  (3 * x^2 - 2 * x - 8 ≤ 0) ↔ (-4/3 ≤ x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l670_67040


namespace NUMINAMATH_CALUDE_quadratic_abs_inequality_l670_67076

theorem quadratic_abs_inequality (x : ℝ) : 
  x^2 + 4*x - 96 > |x| ↔ x < -12 ∨ x > 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_abs_inequality_l670_67076


namespace NUMINAMATH_CALUDE_intersection_point_of_f_and_inverse_l670_67097

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

-- Theorem statement
theorem intersection_point_of_f_and_inverse :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-1, -1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_f_and_inverse_l670_67097


namespace NUMINAMATH_CALUDE_intersection_A_B_l670_67037

-- Define set A
def A : Set ℝ := {x : ℝ | |x| < 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 5*x - 6 < 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l670_67037


namespace NUMINAMATH_CALUDE_sin_five_pi_sixths_minus_two_alpha_l670_67069

theorem sin_five_pi_sixths_minus_two_alpha 
  (h : Real.cos (π / 6 - α) = 1 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_five_pi_sixths_minus_two_alpha_l670_67069


namespace NUMINAMATH_CALUDE_divide_by_three_l670_67075

theorem divide_by_three (n : ℚ) (h : n / 4 = 12) : n / 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_three_l670_67075


namespace NUMINAMATH_CALUDE_coopers_fence_depth_l670_67015

/-- Proves that the depth of each wall in Cooper's fence is 2 bricks -/
theorem coopers_fence_depth (num_walls : ℕ) (wall_length : ℕ) (wall_height : ℕ) (total_bricks : ℕ) :
  num_walls = 4 →
  wall_length = 20 →
  wall_height = 5 →
  total_bricks = 800 →
  (total_bricks / (num_walls * wall_length * wall_height) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_coopers_fence_depth_l670_67015


namespace NUMINAMATH_CALUDE_smallest_factorial_not_divisible_by_62_l670_67054

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_factorial_not_divisible_by_62 :
  (∀ n : ℕ, n < 31 → is_factor 62 (factorial n)) ∧
  ¬ is_factor 62 (factorial 31) ∧
  (∀ k : ℕ, k < 62 → (∃ m : ℕ, is_factor k (factorial m)) ∨ is_prime k) ∧
  ¬ is_prime 62 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factorial_not_divisible_by_62_l670_67054


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l670_67024

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l670_67024


namespace NUMINAMATH_CALUDE_guide_is_native_l670_67090

/-- Represents the two tribes on the island -/
inductive Tribe
  | Native
  | Alien

/-- Represents a person on the island -/
structure Person where
  tribe : Tribe

/-- Represents the claim a person makes about their tribe -/
def claim (p : Person) : Tribe :=
  match p.tribe with
  | Tribe.Native => Tribe.Native
  | Tribe.Alien => Tribe.Native

/-- Represents the report a guide makes about another person's claim -/
def report (guide : Person) (other : Person) : Tribe :=
  match guide.tribe with
  | Tribe.Native => claim other
  | Tribe.Alien => claim other

theorem guide_is_native (guide : Person) (other : Person) :
  report guide other = Tribe.Native → guide.tribe = Tribe.Native :=
by
  sorry

#check guide_is_native

end NUMINAMATH_CALUDE_guide_is_native_l670_67090


namespace NUMINAMATH_CALUDE_min_value_a2_plus_b2_l670_67096

theorem min_value_a2_plus_b2 (a b : ℝ) (h : a^2 + 2*a*b - 3*b^2 = 1) :
  a^2 + b^2 ≥ (Real.sqrt 5 + 1) / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a2_plus_b2_l670_67096


namespace NUMINAMATH_CALUDE_video_views_proof_l670_67021

/-- Calculates the total number of views for a video given initial views,
    a multiplier for increase after 4 days, and additional views after 2 more days. -/
def totalViews (initialViews : ℕ) (increaseMultiplier : ℕ) (additionalViews : ℕ) : ℕ :=
  initialViews + (increaseMultiplier * initialViews) + additionalViews

/-- Proves that given the specific conditions from the problem,
    the total number of views is 94000. -/
theorem video_views_proof :
  totalViews 4000 10 50000 = 94000 := by
  sorry

end NUMINAMATH_CALUDE_video_views_proof_l670_67021


namespace NUMINAMATH_CALUDE_parabola_intersection_range_l670_67046

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line passing through the focus
def line (x y : ℝ) : Prop := y = x - 1

-- Define the circle E with AB as diameter
def circle_E (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 16

-- Define the point D
def point_D (t : ℝ) : ℝ × ℝ := (-2, t)

theorem parabola_intersection_range (t : ℝ) :
  (∃ (A B P Q : ℝ × ℝ),
    parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    circle_E P.1 P.2 ∧ circle_E Q.1 Q.2 ∧
    (∃ (r : ℝ), (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4*r^2 ∧
                ((point_D t).1 - P.1)^2 + ((point_D t).2 - P.2)^2 = r^2)) →
  2 - Real.sqrt 7 ≤ t ∧ t ≤ 2 + Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_range_l670_67046


namespace NUMINAMATH_CALUDE_seventh_term_is_384_l670_67022

/-- The nth term of a geometric sequence -/
def geometricSequenceTerm (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ (n - 1)

/-- The seventh term of the specific geometric sequence -/
def seventhTerm : ℝ :=
  geometricSequenceTerm 6 (-2) 7

theorem seventh_term_is_384 : seventhTerm = 384 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_384_l670_67022


namespace NUMINAMATH_CALUDE_circle_radius_problem_l670_67058

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def collinear (a b c : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, b = a + t • (c - a)

def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

def share_common_tangent (c1 c2 c3 : Circle) : Prop :=
  ∃ (l : ℝ × ℝ → Prop), ∀ (p : ℝ × ℝ),
    l p → (∃ (q : ℝ × ℝ), l q ∧ 
      ((p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∨
       (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2 ∨
       (p.1 - c3.center.1)^2 + (p.2 - c3.center.2)^2 = c3.radius^2))

theorem circle_radius_problem (A B C : Circle) 
  (h1 : collinear A.center B.center C.center)
  (h2 : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ B.center = A.center + t • (C.center - A.center))
  (h3 : externally_tangent A B)
  (h4 : externally_tangent B C)
  (h5 : share_common_tangent A B C)
  (h6 : A.radius = 12)
  (h7 : B.radius = 42) :
  C.radius = 147 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l670_67058


namespace NUMINAMATH_CALUDE_restaurant_group_composition_l670_67070

/-- Proves that in a group of 11 people, where adult meals cost $8 each and kids eat free,
    if the total cost is $72, then the number of kids in the group is 2. -/
theorem restaurant_group_composition (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_people = 11)
  (h2 : adult_meal_cost = 8)
  (h3 : total_cost = 72) :
  total_people - (total_cost / adult_meal_cost) = 2 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_group_composition_l670_67070


namespace NUMINAMATH_CALUDE_product_of_extreme_roots_l670_67065

-- Define the equation
def equation (x : ℝ) : Prop := x * |x| - 5 * |x| + 6 = 0

-- Define the set of roots
def roots : Set ℝ := {x : ℝ | equation x}

-- Statement to prove
theorem product_of_extreme_roots :
  ∃ (max_root min_root : ℝ),
    max_root ∈ roots ∧
    min_root ∈ roots ∧
    (∀ x ∈ roots, x ≤ max_root) ∧
    (∀ x ∈ roots, x ≥ min_root) ∧
    max_root * min_root = -3 :=
sorry

end NUMINAMATH_CALUDE_product_of_extreme_roots_l670_67065


namespace NUMINAMATH_CALUDE_vector_problem_l670_67093

def a : ℝ × ℝ := (1, 2)

theorem vector_problem (b : ℝ × ℝ) (θ : ℝ) :
  (b.1 ^ 2 + b.2 ^ 2 = 20) →
  (∃ (k : ℝ), b = k • a) →
  (b = (2, 4) ∨ b = (-2, -4)) ∧
  ((2 * a.1 - 3 * b.1) * (2 * a.1 + b.1) + (2 * a.2 - 3 * b.2) * (2 * a.2 + b.2) = -20) →
  θ = 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l670_67093


namespace NUMINAMATH_CALUDE_no_alpha_sequence_exists_l670_67030

theorem no_alpha_sequence_exists : ¬∃ (α : ℝ) (a : ℕ → ℝ), 
  (0 < α ∧ α < 1) ∧ 
  (∀ n, 0 < a n) ∧
  (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) := by
  sorry

end NUMINAMATH_CALUDE_no_alpha_sequence_exists_l670_67030


namespace NUMINAMATH_CALUDE_committee_meeting_attendance_l670_67007

theorem committee_meeting_attendance :
  ∀ (assoc_prof asst_prof : ℕ),
  2 * assoc_prof + asst_prof = 11 →
  assoc_prof + 2 * asst_prof = 16 →
  assoc_prof + asst_prof = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_committee_meeting_attendance_l670_67007


namespace NUMINAMATH_CALUDE_transformed_area_theorem_l670_67045

/-- A 2x2 matrix representing the transformation --/
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 5, -2]

/-- The area of the original region T --/
def original_area : ℝ := 12

/-- Theorem stating that applying the transformation matrix to a region with area 12 results in a new region with area 312 --/
theorem transformed_area_theorem :
  abs (Matrix.det transformation_matrix) * original_area = 312 := by
  sorry

#check transformed_area_theorem

end NUMINAMATH_CALUDE_transformed_area_theorem_l670_67045


namespace NUMINAMATH_CALUDE_f_derivative_at_sqrt2_over_2_l670_67089

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem f_derivative_at_sqrt2_over_2 :
  (deriv f) (Real.sqrt 2 / 2) = -3/2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_sqrt2_over_2_l670_67089


namespace NUMINAMATH_CALUDE_double_inequality_l670_67001

theorem double_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (0 < 1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1))) ∧
  (1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1)) ≤ 1 / 8) ∧
  (1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1)) = 1 / 8 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_double_inequality_l670_67001


namespace NUMINAMATH_CALUDE_fraction_equality_l670_67053

theorem fraction_equality (a b c d e : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 1 / 2) :
  e / a = 8 / 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l670_67053


namespace NUMINAMATH_CALUDE_car_fuel_usage_l670_67051

/-- Proves that a car traveling for 5 hours at 60 mph with a fuel efficiency of 1 gallon per 30 miles
    uses 5/6 of a 12-gallon tank. -/
theorem car_fuel_usage (speed : ℝ) (fuel_efficiency : ℝ) (tank_capacity : ℝ) (travel_time : ℝ) :
  speed = 60 →
  fuel_efficiency = 30 →
  tank_capacity = 12 →
  travel_time = 5 →
  (speed * travel_time / fuel_efficiency) / tank_capacity = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_car_fuel_usage_l670_67051


namespace NUMINAMATH_CALUDE_x_value_l670_67041

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 3, 9}

theorem x_value (x : ℕ) (h1 : x ∈ A) (h2 : x ∉ B) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l670_67041


namespace NUMINAMATH_CALUDE_pizza_toppings_l670_67026

theorem pizza_toppings (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h1 : total_slices = 16)
  (h2 : pepperoni_slices = 8)
  (h3 : mushroom_slices = 12)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range pepperoni_slices ∨ 
     slice ∈ Finset.range mushroom_slices)) :
  ∃ both : ℕ, both = pepperoni_slices + mushroom_slices - total_slices :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l670_67026


namespace NUMINAMATH_CALUDE_seashells_problem_l670_67019

theorem seashells_problem (given_away : ℕ) (remaining : ℕ) :
  given_away = 18 → remaining = 17 → given_away + remaining = 35 :=
by sorry

end NUMINAMATH_CALUDE_seashells_problem_l670_67019


namespace NUMINAMATH_CALUDE_blue_spotted_fish_count_l670_67014

theorem blue_spotted_fish_count (total_fish : ℕ) (blue_percentage : ℚ) (spotted_fraction : ℚ) : 
  total_fish = 150 →
  blue_percentage = 2/5 →
  spotted_fraction = 3/5 →
  (total_fish : ℚ) * blue_percentage * spotted_fraction = 36 := by
sorry

end NUMINAMATH_CALUDE_blue_spotted_fish_count_l670_67014


namespace NUMINAMATH_CALUDE_football_player_average_increase_l670_67033

theorem football_player_average_increase :
  ∀ (total_goals : ℕ) (goals_fifth_match : ℕ) (num_matches : ℕ),
    total_goals = 16 →
    goals_fifth_match = 4 →
    num_matches = 5 →
    (total_goals : ℚ) / num_matches - ((total_goals - goals_fifth_match) : ℚ) / (num_matches - 1) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_football_player_average_increase_l670_67033


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a3_l670_67047

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_a3 (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h6 : a 6 = 6) (h9 : a 9 = 9) : a 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a3_l670_67047


namespace NUMINAMATH_CALUDE_bombardment_death_percentage_l670_67006

/-- Represents the percentage of people who died by bombardment -/
def bombardment_percentage : ℝ := 10

/-- The initial population of the village -/
def initial_population : ℕ := 4200

/-- The final population after bombardment and departure -/
def final_population : ℕ := 3213

/-- The percentage of people who left after the bombardment -/
def departure_percentage : ℝ := 15

theorem bombardment_death_percentage :
  let remaining_after_bombardment := initial_population - (bombardment_percentage / 100) * initial_population
  let departed := (departure_percentage / 100) * remaining_after_bombardment
  initial_population - (bombardment_percentage / 100) * initial_population - departed = final_population :=
by sorry

end NUMINAMATH_CALUDE_bombardment_death_percentage_l670_67006


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l670_67012

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^8 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + 
    a₄*(x - 1)^4 + a₅*(x - 1)^5 + a₆*(x - 1)^6 + a₇*(x - 1)^7 + a₈*(x - 1)^8 + 
    a₉*(x - 1)^9 + a₁₀*(x - 1)^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l670_67012


namespace NUMINAMATH_CALUDE_expression_evaluation_l670_67010

theorem expression_evaluation :
  let x : ℚ := 1/3
  let y : ℚ := -1/2
  5 * x^2 - 2 * (3 * y^2 + 6 * x * y) - (2 * x^2 - 6 * y^2) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l670_67010


namespace NUMINAMATH_CALUDE_sea_creatures_lost_l670_67063

-- Define the initial number of items collected
def sea_stars : ℕ := 34
def seashells : ℕ := 21
def snails : ℕ := 29

-- Define the number of items left at the end
def items_left : ℕ := 59

-- Define the total number of items collected
def total_collected : ℕ := sea_stars + seashells + snails

-- Define the number of items lost
def items_lost : ℕ := total_collected - items_left

-- Theorem statement
theorem sea_creatures_lost : items_lost = 25 := by
  sorry

end NUMINAMATH_CALUDE_sea_creatures_lost_l670_67063


namespace NUMINAMATH_CALUDE_sum_of_factors_144_l670_67072

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_144 : sum_of_factors 144 = 403 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_144_l670_67072


namespace NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l670_67013

theorem arithmetic_progression_of_primes (p : ℕ) (a : ℕ → ℕ) (d : ℕ) :
  Prime p →
  (∀ i, i ∈ Finset.range p → Prime (a i)) →
  (∀ i j, i < j → j < p → a j - a i = (j - i) * d) →
  a 0 > p →
  p ∣ d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l670_67013


namespace NUMINAMATH_CALUDE_largest_n_for_seq_containment_l670_67036

/-- A bi-infinite sequence of natural numbers -/
def BiInfiniteSeq := ℤ → ℕ

/-- A sequence is periodic with period p if it repeats every p elements -/
def IsPeriodic (s : BiInfiniteSeq) (p : ℕ) : Prop :=
  ∀ i : ℤ, s i = s (i + p)

/-- A subsequence of length n starting at index i is contained in another sequence -/
def SubseqContained (sub main : BiInfiniteSeq) (n : ℕ) (i : ℤ) : Prop :=
  ∀ k : ℕ, k < n → ∃ j : ℤ, sub (i + k) = main j

/-- The main theorem stating the largest possible n -/
theorem largest_n_for_seq_containment :
  ∃ (n : ℕ) (A B : BiInfiniteSeq),
    IsPeriodic A 1995 ∧
    ¬ IsPeriodic B 1995 ∧
    (∀ i : ℤ, SubseqContained B A n i) ∧
    (∀ m : ℕ, m > n →
      ¬ ∃ (C D : BiInfiniteSeq),
        IsPeriodic C 1995 ∧
        ¬ IsPeriodic D 1995 ∧
        (∀ i : ℤ, SubseqContained D C m i)) ∧
    n = 1995 :=
  sorry

end NUMINAMATH_CALUDE_largest_n_for_seq_containment_l670_67036


namespace NUMINAMATH_CALUDE_expression_value_l670_67074

theorem expression_value (a b c : ℝ) : 
  a * (-2)^5 + b * (-2)^3 + c * (-2) - 5 = 7 →
  a * 2^5 + b * 2^3 + c * 2 - 5 = -17 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l670_67074


namespace NUMINAMATH_CALUDE_remainder_eight_power_2010_mod_100_l670_67056

theorem remainder_eight_power_2010_mod_100 : 8^2010 % 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eight_power_2010_mod_100_l670_67056


namespace NUMINAMATH_CALUDE_reciprocal_sum_equality_l670_67008

theorem reciprocal_sum_equality (a b c : ℝ) (n : ℕ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : 1 / a + 1 / b + 1 / c = 1 / (a + b + c)) : 
  1 / a^(2*n+1) + 1 / b^(2*n+1) + 1 / c^(2*n+1) = 
  1 / (a^(2*n+1) + b^(2*n+1) + c^(2*n+1)) := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equality_l670_67008


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l670_67078

theorem inequality_system_integer_solutions (x : ℤ) : 
  (2 * (1 - x) ≤ 4 ∧ x - 4 < (x - 8) / 3) ↔ (x = -1 ∨ x = 0 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l670_67078


namespace NUMINAMATH_CALUDE_milk_left_over_problem_l670_67038

/-- Calculates the amount of milk left over given the total milk production,
    percentage consumed by kids, and percentage of remainder used for cooking. -/
def milk_left_over (total_milk : ℝ) (kids_consumption_percent : ℝ) (cooking_percent : ℝ) : ℝ :=
  let remaining_after_kids := total_milk * (1 - kids_consumption_percent)
  let used_for_cooking := remaining_after_kids * cooking_percent
  remaining_after_kids - used_for_cooking

/-- Proves that given 16 cups of milk, with 75% consumed by kids and 50% of the remainder
    used for cooking, the amount of milk left over is 2 cups. -/
theorem milk_left_over_problem :
  milk_left_over 16 0.75 0.50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_milk_left_over_problem_l670_67038


namespace NUMINAMATH_CALUDE_triangle_side_length_l670_67044

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  c = 2 → b = 6 → B = 2 * π / 3 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l670_67044


namespace NUMINAMATH_CALUDE_volume_difference_equals_negative_thirteen_l670_67049

/-- A rectangular prism with given face perimeters -/
structure RectangularPrism where
  p1 : ℝ
  p2 : ℝ
  p3 : ℝ

/-- Calculate the volume of a rectangular prism given its face perimeters -/
def volume (prism : RectangularPrism) : ℝ := sorry

/-- The difference in volumes between two rectangular prisms -/
def volumeDifference (a b : RectangularPrism) : ℝ :=
  volume a - volume b

theorem volume_difference_equals_negative_thirteen :
  let a : RectangularPrism := ⟨12, 16, 24⟩
  let b : RectangularPrism := ⟨12, 16, 20⟩
  volumeDifference a b = -13 := by sorry

end NUMINAMATH_CALUDE_volume_difference_equals_negative_thirteen_l670_67049


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l670_67028

/-- Represents a rectangular park with given dimensions and fencing cost -/
structure RectangularPark where
  ratio : Rat  -- Ratio of longer side to shorter side
  area : ℝ     -- Area in square meters
  fencingCost : ℝ  -- Cost of fencing per meter in paise

/-- Calculates the cost of fencing a rectangular park -/
def calculateFencingCost (park : RectangularPark) : ℝ :=
  sorry

/-- Theorem: The cost of fencing the given park is 175 rupees -/
theorem fencing_cost_theorem (park : RectangularPark) 
  (h1 : park.ratio = 3/2)
  (h2 : park.area = 7350)
  (h3 : park.fencingCost = 50) : 
  calculateFencingCost park = 175 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_theorem_l670_67028


namespace NUMINAMATH_CALUDE_six_lines_regions_l670_67050

/-- The number of regions created by n lines in a plane where no two are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- The property that no two lines are parallel and no three are concurrent -/
def general_position (n : ℕ) : Prop := sorry

theorem six_lines_regions :
  general_position 6 → num_regions 6 = 22 := by sorry

end NUMINAMATH_CALUDE_six_lines_regions_l670_67050


namespace NUMINAMATH_CALUDE_increase_by_percentage_seventy_five_increased_by_ninety_percent_l670_67092

theorem increase_by_percentage (x : ℝ) (p : ℝ) : 
  x * (1 + p / 100) = x + x * (p / 100) := by sorry

theorem seventy_five_increased_by_ninety_percent : 
  75 * (1 + 90 / 100) = 142.5 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_seventy_five_increased_by_ninety_percent_l670_67092


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l670_67003

theorem greatest_integer_difference (x y : ℝ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) :
  ∃ (n : ℕ), n = ⌊y - x⌋ ∧ n ≤ 6 ∧ ∀ (m : ℕ), m = ⌊y - x⌋ → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l670_67003
