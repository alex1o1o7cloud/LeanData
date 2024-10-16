import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_constraint_l580_58061

theorem tangent_line_constraint (a : ℝ) : 
  (∀ b : ℝ, ¬∃ x : ℝ, (x^3 - 3*a*x + x = b ∧ 3*x^2 - 3*a = -1)) → 
  a < 1/3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_constraint_l580_58061


namespace NUMINAMATH_CALUDE_sallys_number_l580_58087

theorem sallys_number (n : ℕ) : 
  (1000 ≤ n ∧ n ≤ 9999) ∧ 
  (∀ d : ℕ, 2 ≤ d ∧ d ≤ 9 → n % d = 1) ↔ 
  (n = 2521 ∨ n = 5041 ∨ n = 7561) :=
sorry

end NUMINAMATH_CALUDE_sallys_number_l580_58087


namespace NUMINAMATH_CALUDE_antifreeze_solution_l580_58015

def antifreeze_problem (x : ℝ) : Prop :=
  let solution1_percent : ℝ := 10
  let total_volume : ℝ := 20
  let target_percent : ℝ := 15
  let volume_each : ℝ := 7.5
  (solution1_percent * volume_each + x * volume_each) / total_volume = target_percent

theorem antifreeze_solution : 
  ∃ x : ℝ, antifreeze_problem x ∧ x = 30 := by
sorry

end NUMINAMATH_CALUDE_antifreeze_solution_l580_58015


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l580_58046

/-- The surface area of a cylinder with diameter 4 units and height 3 units is 20π square units. -/
theorem cylinder_surface_area : 
  let d : ℝ := 4  -- diameter
  let h : ℝ := 3  -- height
  let r : ℝ := d / 2  -- radius
  let surface_area : ℝ := 2 * Real.pi * r^2 + 2 * Real.pi * r * h
  surface_area = 20 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l580_58046


namespace NUMINAMATH_CALUDE_expression_equality_l580_58057

theorem expression_equality : 
  (2 / 3 * Real.sqrt 15 - Real.sqrt 20) / (1 / 3 * Real.sqrt 5) = 2 * Real.sqrt 3 - 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l580_58057


namespace NUMINAMATH_CALUDE_tom_bike_miles_per_day_l580_58019

theorem tom_bike_miles_per_day 
  (total_miles : ℕ) 
  (days_in_year : ℕ) 
  (first_period_days : ℕ) 
  (miles_per_day_first_period : ℕ) 
  (h1 : total_miles = 11860)
  (h2 : days_in_year = 365)
  (h3 : first_period_days = 183)
  (h4 : miles_per_day_first_period = 30) :
  (total_miles - miles_per_day_first_period * first_period_days) / (days_in_year - first_period_days) = 35 :=
by sorry

end NUMINAMATH_CALUDE_tom_bike_miles_per_day_l580_58019


namespace NUMINAMATH_CALUDE_rectangle_max_area_l580_58089

theorem rectangle_max_area (a b : ℝ) (h : a > 0 ∧ b > 0) :
  2 * (a + b) = 60 → a * b ≤ 225 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l580_58089


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l580_58025

theorem sufficient_not_necessary_condition :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧
  (∃ a b : ℝ, a * b > 1 ∧ (a ≤ 1 ∨ b ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l580_58025


namespace NUMINAMATH_CALUDE_cyclingProblemSolution_l580_58049

/-- Natalia's cycling distances over four days --/
def cyclingProblem (tuesday : ℕ) : Prop :=
  let monday : ℕ := 40
  let wednesday : ℕ := tuesday / 2
  let thursday : ℕ := monday + wednesday
  monday + tuesday + wednesday + thursday = 180

/-- The solution to Natalia's cycling problem --/
theorem cyclingProblemSolution : ∃ (tuesday : ℕ), cyclingProblem tuesday ∧ tuesday = 33 := by
  sorry

#check cyclingProblemSolution

end NUMINAMATH_CALUDE_cyclingProblemSolution_l580_58049


namespace NUMINAMATH_CALUDE_polynomial_factorization_l580_58031

theorem polynomial_factorization (x : ℝ) : 
  x^6 + 6*x^5 + 15*x^4 + 20*x^3 + 15*x^2 + 6*x + 1 = (x+1)^6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l580_58031


namespace NUMINAMATH_CALUDE_rearrange_incongruent_sums_l580_58069

/-- Given two lists of 2014 integers that are pairwise incongruent modulo 2014,
    there exists a permutation of the second list such that the pairwise sums
    of corresponding elements from both lists are incongruent modulo 4028. -/
theorem rearrange_incongruent_sums
  (x y : Fin 2014 → ℤ)
  (hx : ∀ i j, i ≠ j → x i % 2014 ≠ x j % 2014)
  (hy : ∀ i j, i ≠ j → y i % 2014 ≠ y j % 2014) :
  ∃ σ : Equiv.Perm (Fin 2014),
    ∀ i j, i ≠ j → (x i + y (σ i)) % 4028 ≠ (x j + y (σ j)) % 4028 :=
by sorry

end NUMINAMATH_CALUDE_rearrange_incongruent_sums_l580_58069


namespace NUMINAMATH_CALUDE_rent_comparison_l580_58010

theorem rent_comparison (last_year_earnings : ℝ) : 
  let last_year_rent := 0.25 * last_year_earnings
  let this_year_earnings := last_year_earnings * 1.35
  let this_year_rent := 0.40 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 216 := by
sorry

end NUMINAMATH_CALUDE_rent_comparison_l580_58010


namespace NUMINAMATH_CALUDE_milk_needed_for_recipe_l580_58011

-- Define the ratio of milk to flour
def milk_to_flour_ratio : ℚ := 75 / 250

-- Define the amount of flour Luca wants to use
def flour_amount : ℚ := 1250

-- Theorem: The amount of milk needed for 1250 mL of flour is 375 mL
theorem milk_needed_for_recipe : 
  milk_to_flour_ratio * flour_amount = 375 := by
  sorry


end NUMINAMATH_CALUDE_milk_needed_for_recipe_l580_58011


namespace NUMINAMATH_CALUDE_special_ellipse_eccentricity_l580_58040

/-- An ellipse with the property that the minimum distance from a point on the ellipse to a directrix is equal to the semi-latus rectum. -/
structure SpecialEllipse where
  /-- The eccentricity of the ellipse -/
  eccentricity : ℝ
  /-- The semi-latus rectum of the ellipse -/
  semiLatusRectum : ℝ
  /-- The minimum distance from a point on the ellipse to a directrix -/
  minDirectrixDistance : ℝ
  /-- The condition that the minimum distance to a directrix equals the semi-latus rectum -/
  distance_eq_semiLatusRectum : minDirectrixDistance = semiLatusRectum

/-- The eccentricity of a special ellipse is √2/2 -/
theorem special_ellipse_eccentricity (e : SpecialEllipse) : e.eccentricity = Real.sqrt 2 / 2 :=
  sorry

end NUMINAMATH_CALUDE_special_ellipse_eccentricity_l580_58040


namespace NUMINAMATH_CALUDE_eliza_siblings_l580_58062

/-- Represents the number of siblings Eliza has -/
def num_siblings : ℕ := 4

/-- Represents the total height of all siblings in inches -/
def total_height : ℕ := 330

/-- Represents the height of the two siblings who are 66 inches tall each -/
def tall_siblings_height : ℕ := 66

/-- Represents the height of the sibling who is 60 inches tall -/
def medium_sibling_height : ℕ := 60

/-- Represents the height difference between Eliza and the last sibling -/
def height_difference : ℕ := 2

theorem eliza_siblings :
  ∃ (last_sibling_height : ℕ) (eliza_height : ℕ),
    tall_siblings_height * 2 + medium_sibling_height + last_sibling_height + eliza_height = total_height ∧
    eliza_height + height_difference = last_sibling_height ∧
    num_siblings = 4 :=
by sorry

end NUMINAMATH_CALUDE_eliza_siblings_l580_58062


namespace NUMINAMATH_CALUDE_loan_duration_is_seven_years_l580_58016

/-- Calculates the duration of a loan given the principal, interest rate, and interest paid. -/
def loanDuration (principal interestPaid interestRate : ℚ) : ℚ :=
  (interestPaid * 100) / (principal * interestRate)

/-- Theorem stating that for the given loan conditions, the duration is 7 years. -/
theorem loan_duration_is_seven_years 
  (principal : ℚ) 
  (interestPaid : ℚ) 
  (interestRate : ℚ) 
  (h1 : principal = 1500)
  (h2 : interestPaid = 735)
  (h3 : interestRate = 7) :
  loanDuration principal interestPaid interestRate = 7 := by
  sorry

#eval loanDuration 1500 735 7

end NUMINAMATH_CALUDE_loan_duration_is_seven_years_l580_58016


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l580_58001

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : interest = 4016.25)
  (h2 : rate = 12)
  (h3 : time = 5) :
  ∃ principal : ℝ,
    interest = principal * rate * time / 100 ∧
    principal = 6693.75 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l580_58001


namespace NUMINAMATH_CALUDE_shelf_theorem_l580_58080

/-- Given two shelves, with the second twice as long as the first, and book thicknesses,
    prove the relation between the number of books on each shelf. -/
theorem shelf_theorem (A' H' S' M' E' : ℕ) (x y : ℝ) : 
  A' ≠ H' ∧ A' ≠ S' ∧ A' ≠ M' ∧ A' ≠ E' ∧ 
  H' ≠ S' ∧ H' ≠ M' ∧ H' ≠ E' ∧ 
  S' ≠ M' ∧ S' ≠ E' ∧ 
  M' ≠ E' ∧
  A' > 0 ∧ H' > 0 ∧ S' > 0 ∧ M' > 0 ∧ E' > 0 ∧
  y > x ∧ 
  A' * x + H' * y = S' * x + M' * y ∧ 
  E' * x = 2 * (A' * x + H' * y) →
  E' = (2 * A' * M' - 2 * S' * H') / (M' - H') :=
by sorry

end NUMINAMATH_CALUDE_shelf_theorem_l580_58080


namespace NUMINAMATH_CALUDE_binomial_product_l580_58021

theorem binomial_product (x : ℝ) : (2 * x^2 + 3 * x - 4) * (x + 6) = 2 * x^3 + 15 * x^2 + 14 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l580_58021


namespace NUMINAMATH_CALUDE_max_sales_revenue_l580_58058

/-- Sales price as a function of time -/
def sales_price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

/-- Daily sales volume as a function of time -/
def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

/-- Daily sales revenue as a function of time -/
def sales_revenue (t : ℕ) : ℝ :=
  sales_price t * sales_volume t

/-- The maximum daily sales revenue and the day it occurs -/
theorem max_sales_revenue :
  (∃ (t : ℕ), 0 < t ∧ t ≤ 30 ∧ 
    sales_revenue t = 1125 ∧
    ∀ (s : ℕ), 0 < s ∧ s ≤ 30 → sales_revenue s ≤ sales_revenue t) ∧
  (∀ (s : ℕ), 0 < s ∧ s ≤ 30 ∧ sales_revenue s = 1125 → s = 25) :=
by sorry

end NUMINAMATH_CALUDE_max_sales_revenue_l580_58058


namespace NUMINAMATH_CALUDE_cosine_C_value_l580_58029

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- State the theorem
theorem cosine_C_value (t : Triangle) 
  (h1 : t.c = 2 * t.a)  -- Given condition: c = 2a
  (h2 : Real.sin t.A / Real.sin t.B = 2/3)  -- Given condition: sin A / sin B = 2/3
  : Real.cos t.C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_C_value_l580_58029


namespace NUMINAMATH_CALUDE_least_possible_n_l580_58055

/-- The type of rational coefficients for the polynomial terms -/
structure Coefficient where
  a : ℚ
  b : ℚ

/-- 
Checks if a list of coefficients satisfies the equation
x^2 + x + 4 = ∑(i=1 to n) (a_i * x + b_i)^2 for all real x
-/
def satisfies_equation (coeffs : List Coefficient) : Prop :=
  ∀ (x : ℝ), x^2 + x + 4 = (coeffs.map (fun c => (c.a * x + c.b)^2)).sum

/-- The main theorem stating that 5 is the least possible value of n -/
theorem least_possible_n :
  (∃ (coeffs : List Coefficient), coeffs.length = 5 ∧ satisfies_equation coeffs) ∧
  (∀ (n : ℕ) (coeffs : List Coefficient), n < 5 → coeffs.length = n → ¬satisfies_equation coeffs) :=
sorry

end NUMINAMATH_CALUDE_least_possible_n_l580_58055


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l580_58053

/-- The quadratic equation (a-1)x^2 + x + a^2 - 1 = 0 has 0 as one of its roots if and only if a = -1 -/
theorem quadratic_root_zero (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 + x + a^2 - 1 = 0 ∧ x = 0) ↔ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l580_58053


namespace NUMINAMATH_CALUDE_simplify_expression_l580_58059

theorem simplify_expression : 
  (Real.sqrt 308 / Real.sqrt 77) - (Real.sqrt 245 / Real.sqrt 49) = 2 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l580_58059


namespace NUMINAMATH_CALUDE_probability_bound_l580_58082

def is_divisible_by_four (n : ℕ) : Prop := n % 4 = 0

def count_even (n : ℕ) : ℕ := n / 2

def count_divisible_by_four (n : ℕ) : ℕ := n / 4

def probability_three_integers (n : ℕ) : ℚ :=
  let total := n.choose 3
  let favorable := (count_even n).choose 3 + (count_divisible_by_four n) * ((n - count_divisible_by_four n).choose 2)
  favorable / total

theorem probability_bound (n : ℕ) (h : n = 2017) :
  1/8 < probability_three_integers n ∧ probability_three_integers n < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_bound_l580_58082


namespace NUMINAMATH_CALUDE_golden_delicious_per_pint_l580_58056

/-- The number of pink lady apples required to make one pint of cider -/
def pink_lady_per_pint : ℕ := 40

/-- The number of apples a farmhand can pick per hour -/
def apples_per_hour : ℕ := 240

/-- The number of farmhands -/
def num_farmhands : ℕ := 6

/-- The number of hours worked -/
def hours_worked : ℕ := 5

/-- The ratio of golden delicious to pink lady apples -/
def apple_ratio : ℚ := 1 / 3

/-- The number of pints of cider that can be made -/
def pints_of_cider : ℕ := 120

theorem golden_delicious_per_pint : ℕ := by
  sorry

end NUMINAMATH_CALUDE_golden_delicious_per_pint_l580_58056


namespace NUMINAMATH_CALUDE_technician_round_trip_l580_58085

theorem technician_round_trip 
  (D : ℝ) 
  (P : ℝ) 
  (h1 : D > 0) -- Ensure distance is positive
  (h2 : 0 ≤ P ∧ P ≤ 100) -- Ensure percentage is between 0 and 100
  (h3 : D + (P / 100) * D = 0.7 * (2 * D)) -- Total distance traveled equals 70% of round-trip
  : P = 40 := by
sorry

end NUMINAMATH_CALUDE_technician_round_trip_l580_58085


namespace NUMINAMATH_CALUDE_jerrys_age_l580_58097

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 18 →
  mickey_age = 2 * jerry_age - 6 →
  jerry_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_age_l580_58097


namespace NUMINAMATH_CALUDE_factorization_of_x4_plus_81_l580_58045

theorem factorization_of_x4_plus_81 (x : ℝ) : 
  x^4 + 81 = (x^2 + 3*x + 4.5) * (x^2 - 3*x + 4.5) := by sorry

end NUMINAMATH_CALUDE_factorization_of_x4_plus_81_l580_58045


namespace NUMINAMATH_CALUDE_rectangle_segment_length_l580_58064

/-- Given a rectangle ABCD with side lengths AB = 6 and BC = 5,
    and a segment GH through B perpendicular to DB,
    with A on DG and C on DH, prove that GH = 11√61/6 -/
theorem rectangle_segment_length (A B C D G H : ℝ × ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let DB := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  let GH := Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2)
  AB = 6 →
  BC = 5 →
  (G.1 - H.1) * (D.1 - B.1) + (G.2 - H.2) * (D.2 - B.2) = 0 →  -- GH ⟂ DB
  ∃ t₁ : ℝ, A = t₁ • (G - D) + D →  -- A lies on DG
  ∃ t₂ : ℝ, C = t₂ • (H - D) + D →  -- C lies on DH
  GH = 11 * Real.sqrt 61 / 6 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_segment_length_l580_58064


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l580_58026

theorem unique_two_digit_integer (s : ℕ) : 
  (s ≥ 10 ∧ s < 100) ∧ (13 * s) % 100 = 52 ↔ s = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l580_58026


namespace NUMINAMATH_CALUDE_f_decreasing_range_l580_58028

/-- A piecewise function f(x) defined by a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

/-- The theorem stating the range of a for which f is strictly decreasing -/
theorem f_decreasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) ↔ 
  (0 < a ∧ a ≤ 1/4) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_range_l580_58028


namespace NUMINAMATH_CALUDE_decimal_0_04_is_4_percent_l580_58073

/-- Converts a decimal fraction to a percentage -/
def decimal_to_percentage (d : ℝ) : ℝ := d * 100

/-- The decimal fraction we're working with -/
def given_decimal : ℝ := 0.04

/-- Theorem: The percentage representation of 0.04 is 4% -/
theorem decimal_0_04_is_4_percent : decimal_to_percentage given_decimal = 4 := by
  sorry

end NUMINAMATH_CALUDE_decimal_0_04_is_4_percent_l580_58073


namespace NUMINAMATH_CALUDE_square_sum_of_three_l580_58008

theorem square_sum_of_three (a b c : ℝ) 
  (h1 : a * b + b * c + c * a = 10) 
  (h2 : a + b + c = 31) : 
  a^2 + b^2 + c^2 = 941 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_three_l580_58008


namespace NUMINAMATH_CALUDE_expand_product_l580_58063

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l580_58063


namespace NUMINAMATH_CALUDE_angle_value_in_connected_triangles_l580_58032

theorem angle_value_in_connected_triangles : ∀ x : ℝ,
  (∃ α β : ℝ,
    -- Left triangle
    3 * x + 4 * x + α = 180 ∧
    -- Middle triangle
    α + 5 * x + β = 180 ∧
    -- Right triangle
    β + 2 * x + 6 * x = 180) →
  x = 18 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_in_connected_triangles_l580_58032


namespace NUMINAMATH_CALUDE_brians_breath_holding_l580_58091

theorem brians_breath_holding (T : ℝ) : 
  T > 0 → (T * 2 * 2 * 1.5 = 60) → T = 10 := by
  sorry

end NUMINAMATH_CALUDE_brians_breath_holding_l580_58091


namespace NUMINAMATH_CALUDE_scientific_notation_of_9600000_l580_58084

/-- Proves that 9600000 is equal to 9.6 × 10^6 -/
theorem scientific_notation_of_9600000 : 9600000 = 9.6 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_9600000_l580_58084


namespace NUMINAMATH_CALUDE_point_on_line_with_given_y_l580_58048

/-- A straight line in the xy-plane with given slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem point_on_line_with_given_y (l : Line) (p : Point) :
  l.slope = 4 →
  l.yIntercept = 100 →
  p.y = 300 →
  pointOnLine l p →
  p.x = 50 := by
  sorry

#check point_on_line_with_given_y

end NUMINAMATH_CALUDE_point_on_line_with_given_y_l580_58048


namespace NUMINAMATH_CALUDE_homework_theorem_l580_58070

def homework_problem (total_time math_percentage other_time : ℝ) : Prop :=
  let math_time := math_percentage * total_time
  let science_time := total_time - math_time - other_time
  (science_time / total_time) * 100 = 40

theorem homework_theorem :
  ∀ (total_time math_percentage other_time : ℝ),
    total_time = 150 →
    math_percentage = 0.3 →
    other_time = 45 →
    homework_problem total_time math_percentage other_time :=
by sorry

end NUMINAMATH_CALUDE_homework_theorem_l580_58070


namespace NUMINAMATH_CALUDE_number_problem_l580_58012

theorem number_problem : ∃ x : ℝ, 3 * (2 * x + 8) = 84 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l580_58012


namespace NUMINAMATH_CALUDE_f_s_not_multiplicative_other_l580_58024

/-- r_s(n) is the number of solutions to x_1^2 + x_2^2 + ... + x_s^2 = n in integers x_1, x_2, ..., x_s -/
def r_s (s : ℕ) (n : ℕ) : ℕ := sorry

/-- f_s(n) = (2s)^(-1) * r_s(n) -/
def f_s (s : ℕ) (n : ℕ) : ℚ :=
  (2 * s : ℚ)⁻¹ * (r_s s n : ℚ)

/-- f_s is multiplicative for s = 1, 2, 4, 8 -/
axiom f_s_multiplicative_special (s : ℕ) (m n : ℕ) (h : s = 1 ∨ s = 2 ∨ s = 4 ∨ s = 8) :
  Nat.Coprime m n → f_s s (m * n) = f_s s m * f_s s n

/-- f_s is not multiplicative for any other value of s -/
theorem f_s_not_multiplicative_other (s : ℕ) (h : s ≠ 1 ∧ s ≠ 2 ∧ s ≠ 4 ∧ s ≠ 8) :
  ∃ m n : ℕ, Nat.Coprime m n ∧ f_s s (m * n) ≠ f_s s m * f_s s n := by
  sorry

end NUMINAMATH_CALUDE_f_s_not_multiplicative_other_l580_58024


namespace NUMINAMATH_CALUDE_expand_expression_l580_58043

theorem expand_expression (x y : ℝ) : (3*x + 5) * (4*y^2 + 15) = 12*x*y^2 + 45*x + 20*y^2 + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l580_58043


namespace NUMINAMATH_CALUDE_complex_number_powers_l580_58060

theorem complex_number_powers (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^97 + z^98 + z^99 + z^100 + z^101 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_powers_l580_58060


namespace NUMINAMATH_CALUDE_interest_groups_participation_l580_58013

theorem interest_groups_participation (total_students : ℕ) (total_participants : ℕ) 
  (sports_and_literature : ℕ) (sports_and_math : ℕ) (literature_and_math : ℕ) (all_three : ℕ) :
  total_students = 120 →
  total_participants = 135 →
  sports_and_literature = 15 →
  sports_and_math = 10 →
  literature_and_math = 8 →
  all_three = 4 →
  total_students - (total_participants - sports_and_literature - sports_and_math - literature_and_math + all_three) = 14 :=
by sorry

end NUMINAMATH_CALUDE_interest_groups_participation_l580_58013


namespace NUMINAMATH_CALUDE_abc_inequality_l580_58086

theorem abc_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h_sum : a^2 + b^2 + c^2 + a*b*c = 4) : 
  0 ≤ a*b + b*c + c*a - a*b*c ∧ a*b + b*c + c*a - a*b*c ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l580_58086


namespace NUMINAMATH_CALUDE_circle_tangent_theorem_l580_58033

/-- Given two externally tangent circles and a tangent line satisfying certain conditions,
    prove the relationship between r, R, and p, and the length of BC. -/
theorem circle_tangent_theorem (r R p : ℝ) (h_pos_r : 0 < r) (h_pos_R : 0 < R) (h_pos_p : 0 < p) :
  -- Condition for the geometric configuration
  (p^2 / (4 * (p + 1)) < r / R ∧ r / R < p^2 / (2 * (p + 1))) →
  -- Length of BC
  ∃ (BC : ℝ), BC = p / (p + 1) * Real.sqrt (4 * (p + 1) * R * r - p^2 * R^2) := by
  sorry


end NUMINAMATH_CALUDE_circle_tangent_theorem_l580_58033


namespace NUMINAMATH_CALUDE_triangle_property_l580_58039

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : 3 * t.b * Real.cos t.A = t.c * Real.cos t.A + t.a * Real.cos t.C)
  (h2 : t.a = 4 * Real.sqrt 2) : 
  Real.tan t.A = 2 * Real.sqrt 2 ∧ 
  (∃ (S : ℝ), S ≤ 8 * Real.sqrt 2 ∧ 
    ∀ (S' : ℝ), S' = t.a * t.b * Real.sin t.C / 2 → S' ≤ S) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l580_58039


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l580_58072

theorem line_hyperbola_intersection (k : ℝ) : 
  (∀ x y : ℝ, y = k * x - 1 ∧ x^2 - y^2 = 4 → ∃ x y : ℝ, y = k * x - 1 ∧ x^2 - y^2 = 4) ↔ 
  (k = 1 ∨ k = -1 ∨ (-Real.sqrt 5 / 2 ≤ k ∧ k ≤ Real.sqrt 5 / 2)) :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l580_58072


namespace NUMINAMATH_CALUDE_flame_shooting_time_l580_58095

theorem flame_shooting_time (firing_interval : ℝ) (flame_duration : ℝ) (total_time : ℝ) :
  firing_interval = 15 →
  flame_duration = 5 →
  total_time = 60 →
  (total_time / firing_interval) * flame_duration = 20 := by
  sorry

end NUMINAMATH_CALUDE_flame_shooting_time_l580_58095


namespace NUMINAMATH_CALUDE_stratified_sample_size_l580_58099

/-- Represents the number of athletes in a sample -/
structure Sample where
  male : ℕ
  female : ℕ

/-- Represents the total population of athletes -/
structure Population where
  male : ℕ
  female : ℕ

/-- Checks if a sample is stratified with respect to a population -/
def isStratifiedSample (pop : Population) (samp : Sample) : Prop :=
  samp.male * pop.female = samp.female * pop.male

/-- The main theorem to prove -/
theorem stratified_sample_size 
  (pop : Population) 
  (samp : Sample) 
  (h1 : pop.male = 42)
  (h2 : pop.female = 30)
  (h3 : samp.female = 5)
  (h4 : isStratifiedSample pop samp) :
  samp.male + samp.female = 12 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_size_l580_58099


namespace NUMINAMATH_CALUDE_parabola_focus_theorem_l580_58098

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line type -/
structure Line

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle type -/
structure Circle where
  center : Point
  radius : ℝ

def intersects (l : Line) (para : Parabola) (A B : Point) : Prop :=
  sorry

def focus_on_line (para : Parabola) (l : Line) : Prop :=
  sorry

def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.x)^2 + (y - c.center.y)^2 = c.radius^2

def is_diameter (A B : Point) (c : Circle) : Prop :=
  sorry

theorem parabola_focus_theorem (para : Parabola) (l : Line) (A B : Point) (c : Circle) :
  intersects l para A B →
  focus_on_line para l →
  is_diameter A B c →
  circle_equation c 3 2 →
  c.radius = 4 →
  para.p = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_theorem_l580_58098


namespace NUMINAMATH_CALUDE_structure_has_112_cubes_l580_58005

/-- A structure made of cubes with 5 layers -/
structure CubeStructure where
  middle_layer : ℕ
  other_layers : ℕ
  total_layers : ℕ
  h_middle : middle_layer = 16
  h_other : other_layers = 24
  h_total : total_layers = 5

/-- The total number of cubes in the structure -/
def total_cubes (s : CubeStructure) : ℕ :=
  s.middle_layer + (s.total_layers - 1) * s.other_layers

/-- Theorem stating that the structure contains 112 cubes -/
theorem structure_has_112_cubes (s : CubeStructure) : total_cubes s = 112 := by
  sorry


end NUMINAMATH_CALUDE_structure_has_112_cubes_l580_58005


namespace NUMINAMATH_CALUDE_soap_brand_survey_l580_58081

theorem soap_brand_survey (total : ℕ) (neither : ℕ) (only_w : ℕ) :
  total = 200 →
  neither = 80 →
  only_w = 60 →
  ∃ (both : ℕ),
    both * 4 = total - neither - only_w ∧
    both = 15 :=
by sorry

end NUMINAMATH_CALUDE_soap_brand_survey_l580_58081


namespace NUMINAMATH_CALUDE_job_completion_time_l580_58038

theorem job_completion_time (x : ℝ) : 
  x > 0 → 
  5 * (1/x + 1/20) = 1 - 0.41666666666666663 → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l580_58038


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l580_58083

theorem quadratic_inequality_solution (a c : ℝ) : 
  (∀ x, (1/3 < x ∧ x < 1/2) ↔ a*x^2 + 5*x + c > 0) →
  a = -6 ∧ c = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l580_58083


namespace NUMINAMATH_CALUDE_arcade_spending_l580_58096

theorem arcade_spending (allowance : ℚ) (arcade_fraction : ℚ) (remaining : ℚ) :
  allowance = 2.25 →
  remaining = 0.60 →
  remaining = (1 - arcade_fraction) * allowance - (1/3) * ((1 - arcade_fraction) * allowance) →
  arcade_fraction = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_arcade_spending_l580_58096


namespace NUMINAMATH_CALUDE_walking_competition_analysis_l580_58054

/-- The Chi-square statistic for a 2x2 contingency table -/
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for 90% confidence in a Chi-square test with 1 degree of freedom -/
def critical_value : ℚ := 2706 / 1000

/-- The probability of selecting a Female Walking Star -/
def p_female_walking_star : ℚ := 14 / 70

/-- The number of trials in the binomial distribution -/
def num_trials : ℕ := 3

/-- The expected value of X (number of Female Walking Stars in a sample of 3) -/
def expected_value : ℚ := num_trials * p_female_walking_star

theorem walking_competition_analysis :
  let k_squared := chi_square 24 16 16 14
  k_squared < critical_value ∧ expected_value = 3/5 := by sorry

end NUMINAMATH_CALUDE_walking_competition_analysis_l580_58054


namespace NUMINAMATH_CALUDE_license_plate_theorem_l580_58000

def license_plate_combinations : ℕ :=
  let alphabet_size : ℕ := 26
  let plate_length : ℕ := 5
  let repeated_letters : ℕ := 2
  let non_zero_digits : ℕ := 9

  let choose_repeated_letters := Nat.choose alphabet_size repeated_letters
  let assign_first_repeat := Nat.choose plate_length repeated_letters
  let assign_second_repeat := Nat.choose (plate_length - repeated_letters) repeated_letters
  let remaining_letter_choices := alphabet_size - repeated_letters
  
  choose_repeated_letters * assign_first_repeat * assign_second_repeat * remaining_letter_choices * non_zero_digits

theorem license_plate_theorem : license_plate_combinations = 210600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l580_58000


namespace NUMINAMATH_CALUDE_inequality_solution_set_l580_58022

theorem inequality_solution_set (x : ℝ) :
  (5 * x - 2 ≤ 3 * (1 + x)) ↔ (x ≤ 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l580_58022


namespace NUMINAMATH_CALUDE_equation_solutions_l580_58090

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 7 ∧ x2 = 2 - Real.sqrt 7 ∧
    x1^2 - 4*x1 - 3 = 0 ∧ x2^2 - 4*x2 - 3 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = 4 ∧ x2 = 4/3 ∧
    (x1 + 1)^2 = (2*x1 - 3)^2 ∧ (x2 + 1)^2 = (2*x2 - 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l580_58090


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l580_58004

/-- The area of a rectangular garden -/
def garden_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular garden with length 12 m and width 5 m is 60 square meters -/
theorem rectangular_garden_area :
  garden_area 12 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l580_58004


namespace NUMINAMATH_CALUDE_sqrt_two_div_sqrt_half_equals_two_l580_58052

theorem sqrt_two_div_sqrt_half_equals_two : 
  Real.sqrt 2 / Real.sqrt (1/2) = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_two_div_sqrt_half_equals_two_l580_58052


namespace NUMINAMATH_CALUDE_problem_proof_l580_58006

theorem problem_proof (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l580_58006


namespace NUMINAMATH_CALUDE_gingerbread_problem_l580_58014

theorem gingerbread_problem (total : ℕ) (red_hats blue_boots both : ℕ) : 
  red_hats = 6 →
  blue_boots = 9 →
  2 * red_hats = total →
  both = red_hats + blue_boots - total →
  both = 3 := by
sorry

end NUMINAMATH_CALUDE_gingerbread_problem_l580_58014


namespace NUMINAMATH_CALUDE_blueberry_lake_fish_count_l580_58018

/-- The number of fish associated with each white duck -/
def white_duck_fish : ℕ := 8

/-- The number of fish associated with each black duck -/
def black_duck_fish : ℕ := 15

/-- The number of fish associated with each multicolor duck -/
def multicolor_duck_fish : ℕ := 20

/-- The number of fish associated with each golden duck -/
def golden_duck_fish : ℕ := 25

/-- The number of fish associated with each teal duck -/
def teal_duck_fish : ℕ := 30

/-- The number of white ducks in Blueberry Lake -/
def white_ducks : ℕ := 10

/-- The number of black ducks in Blueberry Lake -/
def black_ducks : ℕ := 12

/-- The number of multicolor ducks in Blueberry Lake -/
def multicolor_ducks : ℕ := 8

/-- The number of golden ducks in Blueberry Lake -/
def golden_ducks : ℕ := 6

/-- The number of teal ducks in Blueberry Lake -/
def teal_ducks : ℕ := 14

/-- The total number of fish in Blueberry Lake -/
def total_fish : ℕ := white_duck_fish * white_ducks + 
                      black_duck_fish * black_ducks + 
                      multicolor_duck_fish * multicolor_ducks + 
                      golden_duck_fish * golden_ducks + 
                      teal_duck_fish * teal_ducks

theorem blueberry_lake_fish_count : total_fish = 990 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_lake_fish_count_l580_58018


namespace NUMINAMATH_CALUDE_point_on_line_l580_58071

/-- Given two points (m, n) and (m + a, n + 1.5) on the line x = 2y + 5, prove that a = 3 -/
theorem point_on_line (m n a : ℝ) : 
  (m = 2 * n + 5) → 
  (m + a = 2 * (n + 1.5) + 5) → 
  a = 3 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l580_58071


namespace NUMINAMATH_CALUDE_max_value_theorem_l580_58076

theorem max_value_theorem (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0) 
  (h2 : b - a - 1 ≤ 0) 
  (h3 : a ≤ 1) : 
  (∀ x y : ℝ, x + y - 2 ≥ 0 → y - x - 1 ≤ 0 → x ≤ 1 → 
    (x + 2*y) / (2*x + y) ≤ (a + 2*b) / (2*a + b)) ∧ 
  (a + 2*b) / (2*a + b) = 7/5 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l580_58076


namespace NUMINAMATH_CALUDE_ellipse_m_value_l580_58094

/-- Represents an ellipse with equation x^2 + my^2 = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ x y : ℝ, x^2 + m * y^2 = 1

/-- Represents the property that the foci of the ellipse lie on the y-axis -/
def foci_on_y_axis (e : Ellipse m) : Prop :=
  ∃ c : ℝ, c^2 = 1/m - 1 ∧ c ≠ 0

/-- Represents the property that the length of the major axis is twice the length of the minor axis -/
def major_axis_twice_minor (e : Ellipse m) : Prop :=
  2 * (1 / Real.sqrt m) = 2 * 2 * 1

theorem ellipse_m_value (m : ℝ) (e : Ellipse m)
  (h1 : foci_on_y_axis e)
  (h2 : major_axis_twice_minor e) :
  m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l580_58094


namespace NUMINAMATH_CALUDE_permutations_of_repeated_letters_l580_58078

def phrase : String := "mathstest"

def repeated_letters (s : String) : List Char :=
  s.toList.filter (fun c => s.toList.count c > 1)

def unique_permutations (letters : List Char) : ℕ :=
  Nat.factorial letters.length / (Nat.factorial (letters.count 's') * Nat.factorial (letters.count 't'))

theorem permutations_of_repeated_letters :
  unique_permutations (repeated_letters phrase) = 10 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_repeated_letters_l580_58078


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l580_58077

theorem quadratic_roots_expression (r s : ℝ) : 
  (3 * r^2 - 5 * r - 8 = 0) → 
  (3 * s^2 - 5 * s - 8 = 0) → 
  r ≠ s →
  (9 * r^2 - 9 * s^2) / (r - s) = 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l580_58077


namespace NUMINAMATH_CALUDE_copy_paper_purchase_solution_l580_58017

/-- Represents the purchase of copy papers -/
structure CopyPaperPurchase where
  white : ℕ
  colored : ℕ

/-- The total cost of the purchase in yuan -/
def total_cost (p : CopyPaperPurchase) : ℕ := 80 * p.white + 180 * p.colored

/-- The relationship between white and colored paper quantities -/
def quantity_relation (p : CopyPaperPurchase) : Prop :=
  p.white = 5 * p.colored - 3

/-- The main theorem stating the solution to the problem -/
theorem copy_paper_purchase_solution :
  ∃ (p : CopyPaperPurchase),
    total_cost p = 2660 ∧
    quantity_relation p ∧
    p.white = 22 ∧
    p.colored = 5 := by
  sorry

end NUMINAMATH_CALUDE_copy_paper_purchase_solution_l580_58017


namespace NUMINAMATH_CALUDE_fuel_consumption_population_l580_58023

/-- Represents a car model -/
structure CarModel where
  name : String

/-- Represents a car of a specific model -/
structure Car where
  model : CarModel

/-- Represents fuel consumption measurement -/
structure FuelConsumption where
  amount : ℝ
  distance : ℝ

/-- Represents a survey of fuel consumption -/
structure FuelConsumptionSurvey where
  model : CarModel
  sample_size : ℕ
  measurements : List FuelConsumption

/-- Definition of population for a fuel consumption survey -/
def survey_population (survey : FuelConsumptionSurvey) : Set FuelConsumption :=
  {fc | ∃ (car : Car), car.model = survey.model ∧ fc.distance = 100}

theorem fuel_consumption_population 
  (survey : FuelConsumptionSurvey) 
  (h1 : survey.sample_size = 20) 
  (h2 : ∀ fc ∈ survey.measurements, fc.distance = 100) :
  survey_population survey = 
    {fc | ∃ (car : Car), car.model = survey.model ∧ fc.distance = 100} := by
  sorry

end NUMINAMATH_CALUDE_fuel_consumption_population_l580_58023


namespace NUMINAMATH_CALUDE_pond_capacity_l580_58042

theorem pond_capacity 
  (normal_rate : ℝ) 
  (drought_factor : ℝ) 
  (fill_time : ℝ) 
  (h1 : normal_rate = 6) 
  (h2 : drought_factor = 2/3) 
  (h3 : fill_time = 50) : 
  normal_rate * drought_factor * fill_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_pond_capacity_l580_58042


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_80_nines_80_sevens_l580_58041

/-- A function that returns a natural number consisting of n repetitions of a given digit --/
def repeatDigit (digit : Nat) (n : Nat) : Nat :=
  if n = 0 then 0 else digit + 10 * repeatDigit digit (n - 1)

/-- A function that calculates the sum of digits of a natural number --/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem --/
theorem sum_of_digits_of_product_80_nines_80_sevens :
  sumOfDigits (repeatDigit 9 80 * repeatDigit 7 80) = 720 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_digits_of_product_80_nines_80_sevens_l580_58041


namespace NUMINAMATH_CALUDE_bracket_difference_l580_58035

theorem bracket_difference (a b c : ℝ) : (a - (b - c)) - ((a - b) - c) = 2 * c := by
  sorry

end NUMINAMATH_CALUDE_bracket_difference_l580_58035


namespace NUMINAMATH_CALUDE_sum_a_t_equals_41_l580_58074

theorem sum_a_t_equals_41 (a t : ℝ) (ha : a > 0) (ht : t > 0) 
  (h : Real.sqrt (6 + a / t) = 6 * Real.sqrt (a / t)) : a + t = 41 :=
sorry

end NUMINAMATH_CALUDE_sum_a_t_equals_41_l580_58074


namespace NUMINAMATH_CALUDE_work_earnings_problem_l580_58065

theorem work_earnings_problem (t : ℚ) : 
  (t + 2) * (4*t - 2) = (4*t - 7) * (t + 1) + 4 → t = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_problem_l580_58065


namespace NUMINAMATH_CALUDE_number_solution_l580_58092

theorem number_solution (x : ℝ) (n : ℝ) (h1 : x > 0) (h2 : x / n + x / 25 = 0.06 * x) : n = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l580_58092


namespace NUMINAMATH_CALUDE_half_cutting_line_exists_l580_58002

/-- Triangle ABC with vertices A(0, 10), B(4, 0), and C(10, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- A line represented by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

/-- The area of a triangle given its vertices -/
def triangle_area (t : Triangle) : ℝ := sorry

/-- Check if a line cuts a triangle in half -/
def cuts_in_half (l : Line) (t : Triangle) : Prop := sorry

/-- The theorem stating the existence of a line that cuts the triangle in half
    and the sum of its slope and y-intercept -/
theorem half_cutting_line_exists (t : Triangle) 
  (h1 : t.A = (0, 10))
  (h2 : t.B = (4, 0))
  (h3 : t.C = (10, 0)) :
  ∃ l : Line, cuts_in_half l t ∧ l.slope + l.y_intercept = 5.625 := by
  sorry

end NUMINAMATH_CALUDE_half_cutting_line_exists_l580_58002


namespace NUMINAMATH_CALUDE_basketball_card_price_basketball_card_price_proof_l580_58075

/-- The price of a basketball card pack given the following conditions:
  * Olivia bought 2 packs of basketball cards
  * She bought 5 decks of baseball cards at $4 each
  * She had one $50 bill and received $24 in change
-/
theorem basketball_card_price : ℝ :=
  let baseball_card_price : ℝ := 4
  let baseball_card_count : ℕ := 5
  let basketball_card_count : ℕ := 2
  let total_money : ℝ := 50
  let change : ℝ := 24
  let spent_money : ℝ := total_money - change
  let baseball_total : ℝ := baseball_card_price * baseball_card_count
  3

theorem basketball_card_price_proof :
  let baseball_card_price : ℝ := 4
  let baseball_card_count : ℕ := 5
  let basketball_card_count : ℕ := 2
  let total_money : ℝ := 50
  let change : ℝ := 24
  let spent_money : ℝ := total_money - change
  let baseball_total : ℝ := baseball_card_price * baseball_card_count
  basketball_card_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_basketball_card_price_basketball_card_price_proof_l580_58075


namespace NUMINAMATH_CALUDE_system_solution_l580_58037

theorem system_solution (x y : ℝ) : 
  (x + y = 5 ∧ 2 * x - 3 * y = 20) ↔ (x = 7 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l580_58037


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l580_58047

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x : ℝ, (x - 2)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) → 
  a₀ + a₂ + a₄ = -122 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l580_58047


namespace NUMINAMATH_CALUDE_sport_participation_l580_58067

theorem sport_participation (total : ℕ) (cyclists : ℕ) (swimmers : ℕ) (skiers : ℕ) (unsatisfactory : ℕ)
  (h1 : total = 25)
  (h2 : cyclists = 17)
  (h3 : swimmers = 13)
  (h4 : skiers = 8)
  (h5 : unsatisfactory = 6)
  (h6 : ∀ s : ℕ, s ≤ total → s ≤ cyclists + swimmers + skiers - 2)
  (h7 : cyclists + swimmers + skiers = 2 * (total - unsatisfactory)) :
  ∃ swim_and_ski : ℕ, swim_and_ski = 2 ∧ swim_and_ski ≤ swimmers ∧ swim_and_ski ≤ skiers :=
by sorry

end NUMINAMATH_CALUDE_sport_participation_l580_58067


namespace NUMINAMATH_CALUDE_circle_properties_l580_58079

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*y + 5*m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x - y + 1 = 0

-- Theorem statement
theorem circle_properties :
  -- Part 1: Range of m
  (∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → (m < 1 ∨ m > 4)) ∧
  -- Part 2: Length of chord when m = -2
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ (-2) ∧
    circle_equation x₂ y₂ (-2) ∧
    line_equation x₁ y₁ ∧
    line_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 26) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l580_58079


namespace NUMINAMATH_CALUDE_average_speed_bicycle_and_walk_l580_58009

/-- Proves that the average speed of a pedestrian who rode a bicycle for 40 minutes at 5 m/s
    and then walked for 2 hours at 5 km/h is 8.25 km/h. -/
theorem average_speed_bicycle_and_walk (
  bicycle_time : Real) (bicycle_speed : Real) (walk_time : Real) (walk_speed : Real)
  (h1 : bicycle_time = 40 / 60) -- 40 minutes in hours
  (h2 : bicycle_speed = 5 * 3.6) -- 5 m/s converted to km/h
  (h3 : walk_time = 2) -- 2 hours
  (h4 : walk_speed = 5) -- 5 km/h
  : (bicycle_time * bicycle_speed + walk_time * walk_speed) / (bicycle_time + walk_time) = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_bicycle_and_walk_l580_58009


namespace NUMINAMATH_CALUDE_problem_statement_l580_58020

theorem problem_statement (m n : ℝ) (h1 : m ≠ n) (h2 : m^2 = n + 2) (h3 : n^2 = m + 2) :
  4 * m * n - m^3 - n^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l580_58020


namespace NUMINAMATH_CALUDE_aftershave_dilution_l580_58066

/-- Proves that adding 30 ml of pure water to 50 ml of 30% alcohol solution results in an 18.75% alcohol solution. -/
theorem aftershave_dilution (initial_volume : ℝ) (initial_concentration : ℝ) (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.30 →
  added_water = 30 →
  final_concentration = 0.1875 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_aftershave_dilution_l580_58066


namespace NUMINAMATH_CALUDE_polygon_diagonals_l580_58007

/-- The number of diagonals in a polygon with exterior angles of 10 degrees each -/
theorem polygon_diagonals (n : ℕ) (h1 : n * 10 = 360) : 
  (n * (n - 3)) / 2 = 594 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l580_58007


namespace NUMINAMATH_CALUDE_fruit_distribution_l580_58093

theorem fruit_distribution (num_students : ℕ) 
  (h1 : 2 * num_students + 6 = num_apples)
  (h2 : 7 * num_students - 5 = num_oranges)
  (h3 : num_oranges = 3 * num_apples + 3) : 
  num_students = 26 := by
sorry

end NUMINAMATH_CALUDE_fruit_distribution_l580_58093


namespace NUMINAMATH_CALUDE_square_sum_eighteen_l580_58027

theorem square_sum_eighteen (x y : ℝ) 
  (h1 : y + 9 = (x - 3)^3)
  (h2 : x + 9 = (y - 3)^3)
  (h3 : x ≠ y) : 
  x^2 + y^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_eighteen_l580_58027


namespace NUMINAMATH_CALUDE_sum_x_y_z_l580_58036

theorem sum_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 2 * y) :
  x + y + z = 10 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_z_l580_58036


namespace NUMINAMATH_CALUDE_theta_value_l580_58003

theorem theta_value : ∃! (θ : ℕ), θ ∈ Finset.range 10 ∧ 378 / θ = 40 + 2 * θ := by sorry

end NUMINAMATH_CALUDE_theta_value_l580_58003


namespace NUMINAMATH_CALUDE_one_pair_probability_l580_58050

/-- The number of colors of socks -/
def num_colors : ℕ := 5

/-- The number of socks per color -/
def socks_per_color : ℕ := 2

/-- The total number of socks -/
def total_socks : ℕ := num_colors * socks_per_color

/-- The number of socks drawn -/
def drawn_socks : ℕ := 5

/-- The probability of drawing exactly one pair of the same color and three different colored socks -/
def probability_one_pair : ℚ := 20 / 21

theorem one_pair_probability :
  probability_one_pair = (num_colors.choose 4 * 4 * 8) / total_socks.choose drawn_socks :=
by sorry

end NUMINAMATH_CALUDE_one_pair_probability_l580_58050


namespace NUMINAMATH_CALUDE_net_income_for_130_tax_l580_58088

/-- Calculates the net income after tax given a pre-tax income -/
def net_income (pre_tax_income : ℝ) : ℝ :=
  pre_tax_income - ((pre_tax_income - 800) * 0.2)

/-- Theorem stating that for a pre-tax income resulting in 130 yuan tax, the net income is 1320 yuan -/
theorem net_income_for_130_tax :
  ∃ (pre_tax_income : ℝ),
    (pre_tax_income - 800) * 0.2 = 130 ∧
    net_income pre_tax_income = 1320 :=
by sorry

end NUMINAMATH_CALUDE_net_income_for_130_tax_l580_58088


namespace NUMINAMATH_CALUDE_final_position_theorem_supplement_angle_beta_theorem_l580_58044

-- Define the initial position of point A
def initial_position : Int := -5

-- Define the movement of point A
def move_right : Int := 4
def move_left : Int := 1

-- Define the angle α
def angle_alpha : Int := 40

-- Theorem for the final position of point A
theorem final_position_theorem :
  initial_position + move_right - move_left = -2 := by sorry

-- Theorem for the supplement of angle β
theorem supplement_angle_beta_theorem :
  180 - (90 - angle_alpha) = 130 := by sorry

end NUMINAMATH_CALUDE_final_position_theorem_supplement_angle_beta_theorem_l580_58044


namespace NUMINAMATH_CALUDE_systematic_sampling_l580_58030

/-- Systematic sampling problem -/
theorem systematic_sampling
  (total_employees : ℕ)
  (sample_size : ℕ)
  (fifth_group_number : ℕ)
  (h1 : total_employees = 200)
  (h2 : sample_size = 40)
  (h3 : fifth_group_number = 22) :
  let first_group_number := 2
  let group_difference := (fifth_group_number - first_group_number) / 4
  (9 * group_difference + first_group_number) = 47 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_l580_58030


namespace NUMINAMATH_CALUDE_each_person_share_l580_58034

/-- The cost to send a person to Mars in billions of dollars -/
def mars_cost : ℚ := 30

/-- The cost to establish a base on the Moon in billions of dollars -/
def moon_base_cost : ℚ := 10

/-- The number of people sharing the cost in millions -/
def number_of_people : ℚ := 200

/-- The total cost in billions of dollars -/
def total_cost : ℚ := mars_cost + moon_base_cost

/-- Theorem: Each person's share of the total cost is $200 -/
theorem each_person_share :
  (total_cost * 1000) / number_of_people = 200 := by sorry

end NUMINAMATH_CALUDE_each_person_share_l580_58034


namespace NUMINAMATH_CALUDE_bus_travel_fraction_l580_58068

/-- Given a journey with a total distance of 90 kilometers, where 1/5 of the distance is traveled by foot,
    12 kilometers are traveled by car, and the remaining distance is traveled by bus,
    prove that the fraction of the total distance traveled by bus is 2/3. -/
theorem bus_travel_fraction (total_distance : ℝ) (foot_fraction : ℝ) (car_distance : ℝ) :
  total_distance = 90 ∧ foot_fraction = 1/5 ∧ car_distance = 12 →
  (total_distance - foot_fraction * total_distance - car_distance) / total_distance = 2/3 := by
sorry

end NUMINAMATH_CALUDE_bus_travel_fraction_l580_58068


namespace NUMINAMATH_CALUDE_hyperbola_equation_l580_58051

/-- A hyperbola with center at the origin -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The equation of a hyperbola -/
def Hyperbola.equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of an asymptote of a hyperbola -/
def Hyperbola.asymptote_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y = (h.b / h.a) * x ∨ y = -(h.b / h.a) * x

theorem hyperbola_equation (h : Hyperbola) 
  (h_asymptote : h.asymptote_equation 4 3)
  (h_focus : h.a^2 - h.b^2 = 25) :
  h.equation = fun x y => x^2 / 16 - y^2 / 9 = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l580_58051
