import Mathlib

namespace bookstore_discount_proof_l3600_360064

/-- Calculates the final price as a percentage of the original price
    given an initial discount and an additional discount on the already discounted price. -/
def final_price_percentage (initial_discount : ℝ) (additional_discount : ℝ) : ℝ :=
  (1 - initial_discount) * (1 - additional_discount) * 100

/-- Proves that with a 40% initial discount and an additional 20% discount,
    the final price is 48% of the original price. -/
theorem bookstore_discount_proof :
  final_price_percentage 0.4 0.2 = 48 := by
sorry

end bookstore_discount_proof_l3600_360064


namespace division_theorem_l3600_360014

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 131 → divisor = 14 → remainder = 5 → 
  dividend = divisor * quotient + remainder → quotient = 9 := by
sorry

end division_theorem_l3600_360014


namespace ellipse_tangent_max_area_l3600_360045

/-- An ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  (positive_a : 0 < a)
  (positive_b : 0 < b)

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the xy-plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Check if a line is tangent to an ellipse -/
def isTangent (l : Line) (e : Ellipse) : Prop :=
  sorry -- Definition of tangent line to ellipse

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  sorry -- Formula for triangle area

/-- The main theorem -/
theorem ellipse_tangent_max_area
  (e : Ellipse)
  (A B C : Point)
  (h1 : e.a^2 = 1 ∧ e.b^2 = 3)
  (h2 : A.x = 1 ∧ A.y = 1)
  (h3 : pointOnEllipse A e)
  (h4 : pointOnEllipse B e)
  (h5 : pointOnEllipse C e)
  (h6 : ∃ (l : Line), isTangent l e ∧ l.a * B.x + l.b * B.y + l.c = 0 ∧ l.a * C.x + l.b * C.y + l.c = 0)
  (h7 : ∀ (B' C' : Point), pointOnEllipse B' e → pointOnEllipse C' e →
        triangleArea A B' C' ≤ triangleArea A B C) :
  ∃ (l : Line), l.a = 1 ∧ l.b = 3 ∧ l.c = 2 ∧
    isTangent l e ∧
    l.a * B.x + l.b * B.y + l.c = 0 ∧
    l.a * C.x + l.b * C.y + l.c = 0 ∧
    triangleArea A B C = 3 :=
  sorry

end ellipse_tangent_max_area_l3600_360045


namespace regression_unit_change_l3600_360003

/-- Represents a linear regression equation of the form y = mx + b -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- The change in y for a unit change in x in a linear regression -/
def unitChange (reg : LinearRegression) : ℝ := reg.slope

theorem regression_unit_change 
  (reg : LinearRegression) 
  (h : reg = { slope := -1.5, intercept := 2 }) : 
  unitChange reg = -1.5 := by sorry

end regression_unit_change_l3600_360003


namespace heartsuit_properties_l3600_360077

def heartsuit (x y : ℝ) : ℝ := |x - y| + 1

theorem heartsuit_properties :
  (∀ x y : ℝ, heartsuit x y = heartsuit y x) ∧
  (∃ x y : ℝ, 2 * (heartsuit x y) ≠ heartsuit (2 * x) (2 * y)) ∧
  (∃ x : ℝ, heartsuit x 0 ≠ x + 1) ∧
  (∀ x : ℝ, heartsuit x x = 1) ∧
  (∀ x y : ℝ, x ≠ y → heartsuit x y > 1) :=
by sorry

end heartsuit_properties_l3600_360077


namespace g_of_60_l3600_360082

/-- Given a function g satisfying the specified properties, prove that g(60) = 11.25 -/
theorem g_of_60 (g : ℝ → ℝ) 
    (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → g (x * y) = g x / y)
    (h2 : g 45 = 15) : 
  g 60 = 11.25 := by
  sorry

end g_of_60_l3600_360082


namespace calendar_reuse_2080_l3600_360067

/-- A year is reusable after a fixed number of years if both years have the same leap year status and start on the same day of the week. -/
def is_calendar_reusable (initial_year target_year : ℕ) : Prop :=
  (initial_year % 4 = 0 ↔ target_year % 4 = 0) ∧
  (initial_year + 1) % 7 = (target_year + 1) % 7

/-- The theorem states that the calendar of the year 2080 can be reused after 28 years. -/
theorem calendar_reuse_2080 :
  let initial_year := 2080
  let year_difference := 28
  is_calendar_reusable initial_year (initial_year + year_difference) :=
by sorry

end calendar_reuse_2080_l3600_360067


namespace quadratic_inequality_l3600_360079

theorem quadratic_inequality (x : ℝ) : x^2 < x + 6 ↔ -2 < x ∧ x < 3 := by
  sorry

end quadratic_inequality_l3600_360079


namespace restaurant_bill_calculation_l3600_360054

/-- Calculates the total amount spent at a restaurant given the prices of items,
    discount rates, service fee rates, and tipping percentage. -/
def restaurant_bill (seafood_price rib_eye_price wine_price dessert_price : ℚ)
                    (wine_quantity : ℕ)
                    (food_discount service_fee_low service_fee_high tip_rate : ℚ) : ℚ :=
  let food_cost := seafood_price + rib_eye_price + dessert_price
  let wine_cost := wine_price * wine_quantity
  let total_before_discount := food_cost + wine_cost
  let discounted_food_cost := food_cost * (1 - food_discount)
  let after_discount := discounted_food_cost + wine_cost
  let service_fee_rate := if after_discount > 80 then service_fee_high else service_fee_low
  let service_fee := after_discount * service_fee_rate
  let total_after_service := after_discount + service_fee
  let tip := total_after_service * tip_rate
  total_after_service + tip

/-- The theorem states that given the specific prices and rates from the problem,
    the total amount spent at the restaurant is $167.67. -/
theorem restaurant_bill_calculation :
  restaurant_bill 45 38 18 12 2 0.1 0.12 0.15 0.2 = 167.67 := by
  sorry

end restaurant_bill_calculation_l3600_360054


namespace direct_proportion_equation_l3600_360040

/-- A direct proportion function passing through (-1, 2) -/
def direct_proportion_through_neg1_2 (k : ℝ) (x : ℝ) : Prop :=
  k ≠ 0 ∧ 2 = k * (-1)

/-- The equation of the direct proportion function -/
def equation_of_direct_proportion (x : ℝ) : ℝ := -2 * x

theorem direct_proportion_equation :
  ∀ k : ℝ, direct_proportion_through_neg1_2 k x →
  ∀ x : ℝ, k * x = equation_of_direct_proportion x :=
sorry

end direct_proportion_equation_l3600_360040


namespace no_solution_to_equation_l3600_360030

theorem no_solution_to_equation : ¬∃ x : ℝ, (x - 2) / (x + 2) - 16 / (x^2 - 4) = (x + 2) / (x - 2) := by
  sorry

end no_solution_to_equation_l3600_360030


namespace midpoint_triangle_half_area_l3600_360043

/-- A rectangle with midpoints on longer sides -/
structure RectangleWithMidpoints where
  length : ℝ
  width : ℝ
  width_half_length : width = length / 2
  p : ℝ × ℝ
  q : ℝ × ℝ
  p_midpoint : p = (0, length / 2)
  q_midpoint : q = (length, length / 2)

/-- The area of the triangle formed by midpoints and corner is half the rectangle area -/
theorem midpoint_triangle_half_area (r : RectangleWithMidpoints) :
    let triangle_area := (r.length * r.length / 2) / 2
    let rectangle_area := r.length * r.width
    triangle_area = rectangle_area / 2 := by
  sorry

end midpoint_triangle_half_area_l3600_360043


namespace sum_of_squares_problem_l3600_360024

theorem sum_of_squares_problem (a b c d k p : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 390)
  (h2 : a*b + b*c + c*a + a*d + b*d + c*d = 5)
  (h3 : a*d + b*d + c*d = k)
  (h4 : a^2*b^2*c^2*d^2 = p) :
  a + b + c + d = 20 := by
  sorry

end sum_of_squares_problem_l3600_360024


namespace exists_valid_configuration_two_thirds_not_exists_valid_configuration_three_fourths_not_exists_valid_configuration_seven_tenths_l3600_360088

/-- Represents the fraction of difficult problems and well-performing students -/
structure TestConfiguration (α : ℚ) where
  difficultProblems : ℚ
  wellPerformingStudents : ℚ
  difficultProblems_ge : difficultProblems ≥ α
  wellPerformingStudents_ge : wellPerformingStudents ≥ α

/-- Theorem stating the existence of a valid configuration for α = 2/3 -/
theorem exists_valid_configuration_two_thirds :
  ∃ (config : TestConfiguration (2/3)), True :=
sorry

/-- Theorem stating the non-existence of a valid configuration for α = 3/4 -/
theorem not_exists_valid_configuration_three_fourths :
  ¬ ∃ (config : TestConfiguration (3/4)), True :=
sorry

/-- Theorem stating the non-existence of a valid configuration for α = 7/10 -/
theorem not_exists_valid_configuration_seven_tenths :
  ¬ ∃ (config : TestConfiguration (7/10)), True :=
sorry

end exists_valid_configuration_two_thirds_not_exists_valid_configuration_three_fourths_not_exists_valid_configuration_seven_tenths_l3600_360088


namespace man_age_difference_l3600_360051

/-- Proves that a man is 37 years older than his son given the conditions. -/
theorem man_age_difference (son_age man_age : ℕ) : 
  son_age = 35 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 37 := by
  sorry

end man_age_difference_l3600_360051


namespace third_term_of_arithmetic_sequence_l3600_360066

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℕ → ℝ :=
  fun k => a₁ + (k - 1 : ℝ) * d

theorem third_term_of_arithmetic_sequence :
  ∀ (a₁ aₙ : ℝ) (n : ℕ),
  n = 10 →
  a₁ = 5 →
  aₙ = 32 →
  let d := (aₙ - a₁) / (n - 1 : ℝ)
  let seq := arithmetic_sequence a₁ d n
  seq 3 = 11 := by
sorry

end third_term_of_arithmetic_sequence_l3600_360066


namespace largest_common_value_less_than_500_l3600_360029

theorem largest_common_value_less_than_500 
  (ap1 : ℕ → ℕ) 
  (ap2 : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, ap1 n = 5 + 4 * n) 
  (h2 : ∀ n : ℕ, ap2 n = 7 + 8 * n) : 
  (∃ k : ℕ, k < 500 ∧ (∃ n m : ℕ, ap1 n = k ∧ ap2 m = k)) ∧
  (∀ l : ℕ, l < 500 → (∃ n m : ℕ, ap1 n = l ∧ ap2 m = l) → l ≤ 497) ∧
  (∃ n m : ℕ, ap1 n = 497 ∧ ap2 m = 497) :=
by sorry

end largest_common_value_less_than_500_l3600_360029


namespace root_sum_theorem_l3600_360008

theorem root_sum_theorem (a b c d r : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (r - a) * (r - b) * (r - c) * (r - d) = 4 →
  4 * r = a + b + c + d := by
sorry

end root_sum_theorem_l3600_360008


namespace periodic_function_smallest_period_l3600_360033

/-- A function satisfying the given periodic property -/
def PeriodicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) + f (x - 4) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x

/-- The smallest positive period of a function -/
def SmallestPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  IsPeriod f T ∧ T > 0 ∧ ∀ S : ℝ, IsPeriod f S ∧ S > 0 → T ≤ S

/-- The main theorem stating that functions satisfying the given condition have a smallest period of 24 -/
theorem periodic_function_smallest_period (f : ℝ → ℝ) (h : PeriodicFunction f) :
    SmallestPeriod f 24 := by
  sorry

end periodic_function_smallest_period_l3600_360033


namespace infinite_valid_points_l3600_360091

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 25}

-- Define the center of the circle
def Center : ℝ × ℝ := (0, 0)

-- Define the diameter endpoints
def DiameterEndpoints : (ℝ × ℝ) × (ℝ × ℝ) := ((-5, 0), (5, 0))

-- Define the condition for points P
def ValidPoint (p : ℝ × ℝ) : Prop :=
  let (a, b) := DiameterEndpoints
  ((p.1 - a.1)^2 + (p.2 - a.2)^2) + ((p.1 - b.1)^2 + (p.2 - b.2)^2) = 50 ∧
  (p.1^2 + p.2^2 < 25)

-- Theorem statement
theorem infinite_valid_points : ∃ (S : Set (ℝ × ℝ)), Set.Infinite S ∧ ∀ p ∈ S, p ∈ Circle ∧ ValidPoint p :=
sorry

end infinite_valid_points_l3600_360091


namespace max_min_a_plus_b_l3600_360011

noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt x

theorem max_min_a_plus_b (a b : ℝ) (h : f (a + 1) + f (b + 2) = 3) :
  (a + b ≤ 1 + Real.sqrt 7) ∧ (a + b ≥ (1 + Real.sqrt 13) / 2) :=
by sorry

end max_min_a_plus_b_l3600_360011


namespace frogs_eat_pests_l3600_360052

/-- The number of pests a single frog eats per day -/
def pests_per_frog_per_day : ℕ := 80

/-- The number of frogs -/
def num_frogs : ℕ := 5

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: 5 frogs eat 2800 pests in a week -/
theorem frogs_eat_pests : 
  pests_per_frog_per_day * num_frogs * days_in_week = 2800 := by
  sorry

end frogs_eat_pests_l3600_360052


namespace digit_appearance_l3600_360071

def digit_free (n : ℕ) (d : Finset ℕ) : Prop :=
  ∀ (i : ℕ), i ∈ d → (n / 10^i % 10 ≠ i)

def contains_digit (n : ℕ) (d : Finset ℕ) : Prop :=
  ∃ (i : ℕ), i ∈ d ∧ (n / 10^i % 10 = i)

theorem digit_appearance (n : ℕ) (h1 : n ≥ 1) (h2 : digit_free n {1, 2, 9}) :
  contains_digit (3 * n) {1, 2, 9} := by
  sorry

end digit_appearance_l3600_360071


namespace medication_forgotten_days_l3600_360092

theorem medication_forgotten_days (total_days : ℕ) (taken_days : ℕ) : 
  total_days = 31 → taken_days = 29 → total_days - taken_days = 2 := by
  sorry

end medication_forgotten_days_l3600_360092


namespace book_arrangement_count_l3600_360069

/-- The number of ways to arrange books on a shelf -/
def arrange_books (arabic : ℕ) (german : ℕ) (spanish : ℕ) : ℕ :=
  Nat.factorial (arabic + german + spanish - 2) * Nat.factorial arabic * Nat.factorial spanish

/-- Theorem stating the number of arrangements for the given book configuration -/
theorem book_arrangement_count :
  arrange_books 2 3 4 = 5760 := by
  sorry

end book_arrangement_count_l3600_360069


namespace money_ratio_problem_l3600_360073

theorem money_ratio_problem (a b : ℕ) (ha : a = 800) (hb : b = 500) : 
  (a : ℚ) / b = 8 / 5 ∧ 
  ((a - 50 : ℚ) / (b + 100) = 5 / 4) := by
sorry

end money_ratio_problem_l3600_360073


namespace parallel_lines_slope_l3600_360061

/-- Two lines are parallel if and only if their slopes are equal -/
theorem parallel_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, (a + 2) * x + (a + 3) * y - 5 = 0 ∧ 
               6 * x + (2 * a - 1) * y - 5 = 0) →
  (a + 2) / 6 = (a + 3) / (2 * a - 1) →
  a = -5/2 := by
sorry


end parallel_lines_slope_l3600_360061


namespace tank_fill_time_l3600_360095

def fill_time_A : ℝ := 30
def fill_rate_B_multiplier : ℝ := 5

theorem tank_fill_time :
  let fill_time_B := fill_time_A / fill_rate_B_multiplier
  let fill_rate_A := 1 / fill_time_A
  let fill_rate_B := 1 / fill_time_B
  let combined_fill_rate := fill_rate_A + fill_rate_B
  1 / combined_fill_rate = 5 := by sorry

end tank_fill_time_l3600_360095


namespace tenth_term_of_specific_sequence_l3600_360074

/-- The nth term of a geometric sequence -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The 10th term of a geometric sequence with first term 4 and common ratio 5/3 -/
theorem tenth_term_of_specific_sequence :
  geometric_sequence 4 (5/3) 10 = 7812500/19683 := by
sorry

end tenth_term_of_specific_sequence_l3600_360074


namespace circle_radius_with_max_inscribed_rectangle_l3600_360078

theorem circle_radius_with_max_inscribed_rectangle (r : ℝ) : 
  r > 0 → 
  (∃ (rect_area : ℝ), rect_area = 50 ∧ rect_area = 2 * r^2) → 
  r = 5 :=
by sorry

end circle_radius_with_max_inscribed_rectangle_l3600_360078


namespace linear_function_quadrants_l3600_360023

/-- A linear function that passes through the first, third, and fourth quadrants -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k + 1) * x + k - 2

/-- Condition for the function to have a positive slope -/
def positive_slope (k : ℝ) : Prop := k + 1 > 0

/-- Condition for the y-intercept to be negative -/
def negative_y_intercept (k : ℝ) : Prop := k - 2 < 0

/-- Theorem stating the range of k for the linear function to pass through the first, third, and fourth quadrants -/
theorem linear_function_quadrants (k : ℝ) : 
  (∀ x, ∃ y, y = linear_function k x) ∧ 
  positive_slope k ∧ 
  negative_y_intercept k ↔ 
  -1 < k ∧ k < 2 :=
sorry

end linear_function_quadrants_l3600_360023


namespace integral_equals_eighteen_implies_a_equals_three_l3600_360057

theorem integral_equals_eighteen_implies_a_equals_three (a : ℝ) :
  (∫ (x : ℝ) in -a..a, x^2 + Real.sin x) = 18 → a = 3 := by
  sorry

end integral_equals_eighteen_implies_a_equals_three_l3600_360057


namespace baker_initial_cakes_l3600_360038

/-- 
Given that Baker made some initial cakes, then made 149 more, 
sold 144, and still has 67 cakes, prove that he initially made 62 cakes.
-/
theorem baker_initial_cakes : 
  ∀ (initial : ℕ), 
  (initial + 149 : ℕ) - 144 = 67 → 
  initial = 62 := by
sorry

end baker_initial_cakes_l3600_360038


namespace article_price_proof_l3600_360000

/-- The normal price of an article before discounts -/
def normal_price : ℝ := 100

/-- The final price after discounts -/
def final_price : ℝ := 72

/-- The first discount rate -/
def discount1 : ℝ := 0.1

/-- The second discount rate -/
def discount2 : ℝ := 0.2

theorem article_price_proof :
  normal_price * (1 - discount1) * (1 - discount2) = final_price :=
by sorry

end article_price_proof_l3600_360000


namespace factor_expression_l3600_360076

theorem factor_expression (c : ℝ) : 210 * c^3 + 35 * c^2 = 35 * c^2 * (6 * c + 1) := by
  sorry

end factor_expression_l3600_360076


namespace sum_of_repeating_decimals_four_and_six_l3600_360017

/-- Represents a repeating decimal with a single digit repetend -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals_four_and_six :
  RepeatingDecimal 4 + RepeatingDecimal 6 = 10 / 9 := by sorry

end sum_of_repeating_decimals_four_and_six_l3600_360017


namespace rectangle_configuration_CG_length_l3600_360021

/-- A configuration of two rectangles ABCD and EFGH with parallel sides -/
structure RectangleConfiguration where
  /-- The length of segment AE -/
  AE : ℝ
  /-- The length of segment BF -/
  BF : ℝ
  /-- The length of segment DH -/
  DH : ℝ
  /-- The length of segment CG -/
  CG : ℝ

/-- The theorem stating the length of CG given the other lengths -/
theorem rectangle_configuration_CG_length 
  (config : RectangleConfiguration) 
  (h1 : config.AE = 10) 
  (h2 : config.BF = 20) 
  (h3 : config.DH = 30) : 
  config.CG = 20 * Real.sqrt 3 := by
  sorry

end rectangle_configuration_CG_length_l3600_360021


namespace correct_stratified_sample_l3600_360063

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  middleAged : ℕ
  young : ℕ
  elderly : ℕ

/-- Calculates the stratified sample sizes for each age group -/
def stratifiedSample (total : ℕ) (ratio : EmployeeCount) (sampleSize : ℕ) : EmployeeCount :=
  let totalRatio := ratio.middleAged + ratio.young + ratio.elderly
  { middleAged := (sampleSize * ratio.middleAged) / totalRatio,
    young := (sampleSize * ratio.young) / totalRatio,
    elderly := (sampleSize * ratio.elderly) / totalRatio }

theorem correct_stratified_sample :
  let total := 3200
  let ratio := EmployeeCount.mk 5 3 2
  let sampleSize := 400
  let result := stratifiedSample total ratio sampleSize
  result.middleAged = 200 ∧ result.young = 120 ∧ result.elderly = 80 := by
  sorry

end correct_stratified_sample_l3600_360063


namespace evaluate_expression_l3600_360048

theorem evaluate_expression : 3 + (-3)^2 = 12 := by sorry

end evaluate_expression_l3600_360048


namespace twenty_seven_power_minus_log_eight_two_equals_zero_l3600_360080

theorem twenty_seven_power_minus_log_eight_two_equals_zero :
  Real.rpow 27 (-1/3) - Real.log 2 / Real.log 8 = 0 := by
  sorry

end twenty_seven_power_minus_log_eight_two_equals_zero_l3600_360080


namespace intersection_complement_equals_interval_l3600_360068

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt x + 1}

-- State the theorem
theorem intersection_complement_equals_interval :
  A ∩ (U \ B) = Set.Icc 0 1 := by sorry

end intersection_complement_equals_interval_l3600_360068


namespace base_x_is_8_l3600_360084

/-- The base of the numeral system in which 1728 (decimal) is represented as 3362 -/
def base_x : ℕ :=
  sorry

/-- The representation of 1728 in base x -/
def representation : List ℕ :=
  [3, 3, 6, 2]

theorem base_x_is_8 :
  (base_x ^ 3 * representation[0]! +
   base_x ^ 2 * representation[1]! +
   base_x ^ 1 * representation[2]! +
   base_x ^ 0 * representation[3]!) = 1728 ∧
  base_x = 8 :=
sorry

end base_x_is_8_l3600_360084


namespace travel_time_calculation_l3600_360049

theorem travel_time_calculation (total_distance : ℝ) (average_speed : ℝ) (return_time : ℝ) :
  total_distance = 2000 ∧ 
  average_speed = 142.85714285714286 ∧ 
  return_time = 4 →
  total_distance / average_speed - return_time = 10 := by
  sorry

end travel_time_calculation_l3600_360049


namespace complex_vector_sum_l3600_360013

theorem complex_vector_sum (z₁ z₂ z₃ : ℂ) (x y : ℝ) 
  (h₁ : z₁ = -1 + I)
  (h₂ : z₂ = 1 + I)
  (h₃ : z₃ = 1 + 4*I)
  (h₄ : z₃ = x • z₁ + y • z₂) : 
  x + y = 4 := by sorry

end complex_vector_sum_l3600_360013


namespace sqrt_10_greater_than_3_l3600_360055

theorem sqrt_10_greater_than_3 : Real.sqrt 10 > 3 := by
  sorry

end sqrt_10_greater_than_3_l3600_360055


namespace l_shaped_area_is_ten_l3600_360098

/-- The area of an L-shaped region formed by subtracting four smaller squares from a larger square -/
def l_shaped_area (large_side : ℝ) (small_side1 small_side2 : ℝ) : ℝ :=
  large_side^2 - 2 * small_side1^2 - 2 * small_side2^2

/-- Theorem stating that the area of the L-shaped region is 10 -/
theorem l_shaped_area_is_ten :
  l_shaped_area 6 2 3 = 10 := by
  sorry

end l_shaped_area_is_ten_l3600_360098


namespace max_divisor_of_f_l3600_360056

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem max_divisor_of_f :
  ∃ (m : ℕ), m = 36 ∧ 
  (∀ (n : ℕ), n > 0 → m ∣ f n) ∧
  (∀ (k : ℕ), k > 36 → ∃ (n : ℕ), n > 0 ∧ ¬(k ∣ f n)) :=
sorry

end max_divisor_of_f_l3600_360056


namespace absolute_value_inequality_l3600_360053

theorem absolute_value_inequality (b : ℝ) :
  (∃ x : ℝ, |x - 5| + |x - 7| < b) ↔ b > 2 :=
by sorry

end absolute_value_inequality_l3600_360053


namespace simple_interest_rate_l3600_360099

theorem simple_interest_rate : 
  ∀ (P : ℝ) (R : ℝ),
  P > 0 →
  (P * R * 10) / 100 = (3 / 5) * P →
  R = 6 := by
sorry

end simple_interest_rate_l3600_360099


namespace fibonacci_properties_l3600_360015

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_properties :
  (∃ f : ℕ → ℕ → ℕ, ∀ k : ℕ, ∃ p q : ℕ, p > k ∧ q > k ∧ ∃ m : ℕ, (fibonacci p * fibonacci q - 1 = m^2)) ∧
  (∃ g : ℕ → ℕ → ℕ × ℕ, ∀ k : ℕ, ∃ m n : ℕ, m > k ∧ n > k ∧ 
    (fibonacci m ∣ fibonacci n^2 + 1) ∧ (fibonacci n ∣ fibonacci m^2 + 1)) :=
by sorry

end fibonacci_properties_l3600_360015


namespace first_terrific_tuesday_l3600_360012

/-- Represents a date with a day and a month -/
structure Date where
  day : ℕ
  month : ℕ

/-- Represents a day of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the number of days in a given month (assuming non-leap year) -/
def daysInMonth (month : ℕ) : ℕ :=
  if month == 2 then 28 else if month ∈ [4, 6, 9, 11] then 30 else 31

/-- Returns the weekday of a given date, assuming February 1 is a Tuesday -/
def weekdayOfDate (d : Date) : Weekday :=
  sorry

/-- Returns true if the given date is a Tuesday -/
def isTuesday (d : Date) : Prop :=
  weekdayOfDate d = Weekday.Tuesday

/-- Returns true if the given date is a Terrific Tuesday (5th Tuesday of the month) -/
def isTerrificTuesday (d : Date) : Prop :=
  isTuesday d ∧ d.day > 28

/-- The main theorem: The first Terrific Tuesday after February 1 is March 29 -/
theorem first_terrific_tuesday : 
  ∃ (d : Date), d.month = 3 ∧ d.day = 29 ∧ isTerrificTuesday d ∧
  ∀ (d' : Date), (d'.month < 3 ∨ (d'.month = 3 ∧ d'.day < 29)) → ¬isTerrificTuesday d' :=
  sorry

end first_terrific_tuesday_l3600_360012


namespace right_triangle_properties_l3600_360036

/-- Right triangle ABC with given properties -/
structure RightTriangleABC where
  -- AB is the hypotenuse
  AB : ℝ
  BC : ℝ
  angleC : ℝ
  hypotenuse_length : AB = 5
  leg_length : BC = 3
  right_angle : angleC = 90

/-- The length of the altitude to the hypotenuse in the right triangle ABC -/
def altitude_to_hypotenuse (t : RightTriangleABC) : ℝ := 2.4

/-- The length of the median to the hypotenuse in the right triangle ABC -/
def median_to_hypotenuse (t : RightTriangleABC) : ℝ := 2.5

/-- Theorem stating the properties of the right triangle ABC -/
theorem right_triangle_properties (t : RightTriangleABC) :
  altitude_to_hypotenuse t = 2.4 ∧ median_to_hypotenuse t = 2.5 := by
  sorry

end right_triangle_properties_l3600_360036


namespace train_length_l3600_360019

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 6 → speed * time * (1000 / 3600) = 150 :=
by sorry

end train_length_l3600_360019


namespace bridge_painting_l3600_360089

theorem bridge_painting (painted_section : ℚ) : 
  (1.2 * painted_section = 1/2) → 
  (1/2 - painted_section = 1/12) := by
sorry

end bridge_painting_l3600_360089


namespace odd_function_a_range_l3600_360090

/-- An odd function f: ℝ → ℝ satisfying given conditions -/
def OddFunction (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x < 0, f x = 9*x + a^2/x + 7) ∧
  (∀ x ≥ 0, f x ≥ 0)

/-- Theorem stating the range of values for a -/
theorem odd_function_a_range (f : ℝ → ℝ) (a : ℝ) 
  (h : OddFunction f a) : 
  a ≥ 7/6 ∨ a ≤ -7/6 :=
sorry

end odd_function_a_range_l3600_360090


namespace triangle_sine_cosine_sum_l3600_360028

theorem triangle_sine_cosine_sum (A B C x y z : ℝ) :
  A + B + C = π →
  x * Real.sin A + y * Real.sin B + z * Real.sin C = 0 →
  (y + z * Real.cos A) * (z + x * Real.cos B) * (x + y * Real.cos C) + 
  (y * Real.cos A + z) * (z * Real.cos B + x) * (x * Real.cos C + y) = 0 :=
by sorry

end triangle_sine_cosine_sum_l3600_360028


namespace book_arrangement_count_l3600_360097

/-- The number of ways to arrange books on a shelf --/
def arrange_books (n_science : ℕ) (n_literature : ℕ) : ℕ :=
  n_literature * (n_literature - 1) * Nat.factorial (n_science + n_literature - 2)

/-- Theorem stating the number of arrangements for the given problem --/
theorem book_arrangement_count :
  arrange_books 4 5 = 10080 :=
sorry

end book_arrangement_count_l3600_360097


namespace ab_value_l3600_360016

theorem ab_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b - 2| = 9) (h3 : a + b > 0) :
  ab = 33 ∨ ab = -33 := by
  sorry

end ab_value_l3600_360016


namespace gift_wrap_sale_total_l3600_360046

/-- Calculates the total amount of money collected from selling gift wrap rolls -/
def total_amount_collected (total_rolls : ℕ) (print_rolls : ℕ) (solid_price : ℚ) (print_price : ℚ) : ℚ :=
  (print_rolls * print_price) + ((total_rolls - print_rolls) * solid_price)

/-- Proves that the total amount collected from selling gift wrap rolls is $2340 -/
theorem gift_wrap_sale_total : 
  total_amount_collected 480 210 4 6 = 2340 := by
  sorry

end gift_wrap_sale_total_l3600_360046


namespace contact_list_count_is_45_l3600_360007

/-- The number of people on Jerome's contact list at the end of the month -/
def contact_list_count : ℕ :=
  let classmates : ℕ := 20
  let out_of_school_friends : ℕ := classmates / 2
  let immediate_family : ℕ := 3
  let extended_family_added : ℕ := 5
  let acquaintances_added : ℕ := 7
  let coworkers_added : ℕ := 10
  let extended_family_removed : ℕ := 3
  let acquaintances_removed : ℕ := 4
  let coworkers_removed : ℕ := (coworkers_added * 3) / 10

  let total_added : ℕ := classmates + out_of_school_friends + immediate_family + 
                         extended_family_added + acquaintances_added + coworkers_added
  let total_removed : ℕ := extended_family_removed + acquaintances_removed + coworkers_removed

  total_added - total_removed

theorem contact_list_count_is_45 : contact_list_count = 45 := by
  sorry

end contact_list_count_is_45_l3600_360007


namespace system_solution_l3600_360081

/-- A solution to the system of equations is a triple (x, y, z) that satisfies all three equations. -/
def IsSolution (x y z : ℝ) : Prop :=
  x + y - z = 4 ∧
  x^2 + y^2 - z^2 = 12 ∧
  x^3 + y^3 - z^3 = 34

/-- The theorem states that the only solutions to the system of equations are (2, 3, 1) and (3, 2, 1). -/
theorem system_solution :
  ∀ x y z : ℝ, IsSolution x y z ↔ ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 2 ∧ z = 1)) := by
  sorry

end system_solution_l3600_360081


namespace trig_identities_l3600_360010

theorem trig_identities (α : Real) (h : Real.tan α = 2) :
  (Real.sin (α - 3 * Real.pi) + Real.cos (Real.pi + α)) /
  (Real.sin (-α) - Real.cos (Real.pi + α)) = 3 ∧
  Real.cos α ^ 2 - 2 * Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end trig_identities_l3600_360010


namespace andrew_payment_l3600_360087

/-- The total amount Andrew paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_price grape_weight mango_price mango_weight : ℕ) : ℕ :=
  grape_price * grape_weight + mango_price * mango_weight

/-- Theorem stating that Andrew paid 1428 to the shopkeeper -/
theorem andrew_payment :
  total_amount 98 11 50 7 = 1428 := by
  sorry

end andrew_payment_l3600_360087


namespace centroid_division_area_difference_l3600_360039

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Represents the centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Represents a line passing through a point -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Calculates the areas of two parts of a triangle divided by a line -/
def dividedAreas (t : Triangle) (l : Line) : ℝ × ℝ := sorry

/-- Theorem: The difference in areas of two parts of a triangle divided by a line
    through its centroid is not greater than 1/9 of the triangle's total area -/
theorem centroid_division_area_difference (t : Triangle) (l : Line) :
  let (A1, A2) := dividedAreas t l
  let G := centroid t
  l.point = G →
  |A1 - A2| ≤ (1/9) * triangleArea t :=
sorry

end centroid_division_area_difference_l3600_360039


namespace geometric_arithmetic_ratio_l3600_360047

/-- Given a geometric sequence {a_n} where 2a_3, a_5/2, 3a_1 forms an arithmetic sequence,
    prove that (a_2 + a_5) / (a_9 + a_6) = 1/9 -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) (hq : q > 0) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (2 * a 3 - a 5 / 2 = a 5 / 2 - 3 * a 1) →  -- arithmetic sequence condition
  (a 2 + a 5) / (a 9 + a 6) = 1 / 9 := by
  sorry

end geometric_arithmetic_ratio_l3600_360047


namespace line_L_equation_l3600_360058

-- Define the point A
def A : ℝ × ℝ := (2, 4)

-- Define the parallel lines
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the line on which the midpoint lies
def midpoint_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the equation of line L
def line_L (x y : ℝ) : Prop := 3*x - y - 2 = 0

-- Theorem statement
theorem line_L_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- L passes through A
    line_L 2 4 ∧
    -- L intersects the parallel lines
    line1 x₁ y₁ ∧ line2 x₂ y₂ ∧ line_L x₁ y₁ ∧ line_L x₂ y₂ ∧
    -- Midpoint of the segment lies on the given line
    midpoint_line ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) :=
by sorry

end line_L_equation_l3600_360058


namespace min_absolute_value_complex_l3600_360096

theorem min_absolute_value_complex (z : ℂ) (h : Complex.abs (z - 10) + Complex.abs (z + 3*I) = 15) :
  ∃ (w : ℂ), Complex.abs w = 2 ∧ ∀ (v : ℂ), (Complex.abs (v - 10) + Complex.abs (v + 3*I) = 15) → Complex.abs v ≥ Complex.abs w :=
by sorry

end min_absolute_value_complex_l3600_360096


namespace coordinate_sum_of_point_B_l3600_360042

theorem coordinate_sum_of_point_B (A B : ℝ × ℝ) : 
  A = (0, 0) →
  B.2 = 5 →
  (B.2 - A.2) / (B.1 - A.1) = 3/4 →
  B.1 + B.2 = 35/3 := by
sorry

end coordinate_sum_of_point_B_l3600_360042


namespace solution_set_of_inequality_range_of_a_l3600_360075

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Define the solution set of f(x) ≤ 0
def solution_set (a : ℝ) : Set ℝ := {x | f a x ≤ 0}

-- Theorem 1
theorem solution_set_of_inequality (a : ℝ) :
  solution_set a = Set.Icc 1 2 →
  {x : ℝ | f a x ≥ 1 - x^2} = Set.Iic (1/2) ∪ Set.Ici 1 :=
sorry

-- Theorem 2
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2*a*(x-1) + 4) →
  a ∈ Set.Iic (1/3) :=
sorry

end solution_set_of_inequality_range_of_a_l3600_360075


namespace monkey_climbing_theorem_l3600_360060

/-- Represents the climbing problem of a monkey on a tree -/
structure ClimbingProblem where
  treeHeight : ℕ
  climbRate : ℕ
  slipRate : ℕ
  restPeriod : ℕ
  restDuration : ℕ

/-- Calculates the time taken for the monkey to reach the top of the tree -/
def timeTakenToClimb (problem : ClimbingProblem) : ℕ :=
  sorry

/-- The theorem stating the solution to the specific climbing problem -/
theorem monkey_climbing_theorem :
  let problem : ClimbingProblem := {
    treeHeight := 253,
    climbRate := 7,
    slipRate := 4,
    restPeriod := 4,
    restDuration := 1
  }
  timeTakenToClimb problem = 109 := by
  sorry

end monkey_climbing_theorem_l3600_360060


namespace green_silk_calculation_l3600_360005

/-- The number of yards of silk dyed for an order -/
def total_yards : ℕ := 111421

/-- The number of yards of silk dyed pink -/
def pink_yards : ℕ := 49500

/-- The number of yards of silk dyed green -/
def green_yards : ℕ := total_yards - pink_yards

theorem green_silk_calculation : green_yards = 61921 := by
  sorry

end green_silk_calculation_l3600_360005


namespace min_abs_w_l3600_360020

theorem min_abs_w (w : ℂ) (h : Complex.abs (w + 2 - 2*I) + Complex.abs (w - 5*I) = 7) :
  Complex.abs w ≥ 10/7 ∧ ∃ w₀ : ℂ, Complex.abs (w₀ + 2 - 2*I) + Complex.abs (w₀ - 5*I) = 7 ∧ Complex.abs w₀ = 10/7 :=
sorry

end min_abs_w_l3600_360020


namespace longest_chord_length_l3600_360034

theorem longest_chord_length (r : ℝ) (h : r = 11) : 
  2 * r = 22 := by sorry

end longest_chord_length_l3600_360034


namespace min_of_four_expressions_bound_l3600_360093

theorem min_of_four_expressions_bound (r s u v : ℝ) :
  min (r - s^2) (min (s - u^2) (min (u - v^2) (v - r^2))) ≤ 1/4 := by
  sorry

end min_of_four_expressions_bound_l3600_360093


namespace greatest_3digit_base8_div_by_5_l3600_360027

/-- Converts a base 8 number to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Checks if a number is a 3-digit base 8 number -/
def is_3digit_base8 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_div_by_5 :
  ∀ n : ℕ, is_3digit_base8 n → (base8_to_base10 n) % 5 = 0 → n ≤ 776 :=
by sorry

end greatest_3digit_base8_div_by_5_l3600_360027


namespace unique_card_combination_l3600_360083

/-- Represents the colors of the cards --/
inductive Color
  | Green
  | Yellow
  | Blue
  | Red

/-- Represents the set of cards with their numbers --/
def CardSet := Color → Nat

/-- Checks if a number is a valid card number (positive integer less than 10) --/
def isValidCardNumber (n : Nat) : Prop := 0 < n ∧ n < 10

/-- Checks if all numbers in a card set are valid --/
def allValidNumbers (cards : CardSet) : Prop :=
  ∀ c, isValidCardNumber (cards c)

/-- Checks if all numbers in a card set are different --/
def allDifferentNumbers (cards : CardSet) : Prop :=
  ∀ c₁ c₂, c₁ ≠ c₂ → cards c₁ ≠ cards c₂

/-- Checks if the product of green and yellow numbers is the green number --/
def greenYellowProduct (cards : CardSet) : Prop :=
  cards Color.Green * cards Color.Yellow = cards Color.Green

/-- Checks if the blue number is the same as the red number --/
def blueRedSame (cards : CardSet) : Prop :=
  cards Color.Blue = cards Color.Red

/-- Checks if the product of red and blue numbers forms a two-digit number
    with green and yellow digits in that order --/
def redBlueProductCondition (cards : CardSet) : Prop :=
  cards Color.Red * cards Color.Blue = 10 * cards Color.Green + cards Color.Yellow

/-- The main theorem stating that the only valid combination is 8, 1, 9, 9 --/
theorem unique_card_combination :
  ∀ cards : CardSet,
    allValidNumbers cards →
    allDifferentNumbers cards →
    greenYellowProduct cards →
    blueRedSame cards →
    redBlueProductCondition cards →
    (cards Color.Green = 8 ∧
     cards Color.Yellow = 1 ∧
     cards Color.Blue = 9 ∧
     cards Color.Red = 9) :=
by sorry

end unique_card_combination_l3600_360083


namespace union_covers_reals_l3600_360050

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem union_covers_reals (a : ℝ) :
  A ∪ B a = Set.univ → a > 2 := by
  sorry

end union_covers_reals_l3600_360050


namespace negative_a_squared_times_a_cubed_l3600_360001

theorem negative_a_squared_times_a_cubed (a : ℝ) : (-a)^2 * a^3 = a^5 := by
  sorry

end negative_a_squared_times_a_cubed_l3600_360001


namespace constant_function_invariant_l3600_360022

theorem constant_function_invariant (g : ℝ → ℝ) (h : ∀ x : ℝ, g x = -3) :
  ∀ x : ℝ, g (3 * x - 5) = -3 := by
  sorry

end constant_function_invariant_l3600_360022


namespace product_of_digits_not_divisible_by_five_l3600_360094

def numbers : List Nat := [3546, 3550, 3565, 3570, 3585]

def is_divisible_by_five (n : Nat) : Bool :=
  n % 5 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_not_divisible_by_five :
  ∃ n ∈ numbers, ¬is_divisible_by_five n ∧
  units_digit n * tens_digit n = 24 :=
by
  sorry

end product_of_digits_not_divisible_by_five_l3600_360094


namespace quadratic_order_l3600_360026

theorem quadratic_order (m y₁ y₂ y₃ : ℝ) (hm : m < -2) 
  (h₁ : y₁ = (m - 1)^2 + 2*(m - 1))
  (h₂ : y₂ = m^2 + 2*m)
  (h₃ : y₃ = (m + 1)^2 + 2*(m + 1)) :
  y₃ < y₂ ∧ y₂ < y₁ := by
sorry

end quadratic_order_l3600_360026


namespace smallest_common_multiple_of_6_and_15_l3600_360070

theorem smallest_common_multiple_of_6_and_15 :
  ∃ b : ℕ+, (∀ n : ℕ+, 6 ∣ n ∧ 15 ∣ n → b ≤ n) ∧ 6 ∣ b ∧ 15 ∣ b ∧ b = 30 := by
  sorry

end smallest_common_multiple_of_6_and_15_l3600_360070


namespace direction_vector_implies_a_eq_plus_minus_two_l3600_360004

/-- Two lines with equations ax + 2y + 3 = 0 and 2x + ay - 1 = 0 have the same direction vector -/
def same_direction_vector (a : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ k * a = 2 ∧ k * 2 = a

theorem direction_vector_implies_a_eq_plus_minus_two (a : ℝ) :
  same_direction_vector a → a = 2 ∨ a = -2 := by
  sorry

end direction_vector_implies_a_eq_plus_minus_two_l3600_360004


namespace meow_to_paw_ratio_l3600_360002

/-- The number of cats in Cat Cafe Cool -/
def cool_cats : ℕ := 5

/-- The number of cats in Cat Cafe Paw -/
def paw_cats : ℕ := 2 * cool_cats

/-- The total number of cats in Cat Cafe Meow and Cat Cafe Paw combined -/
def total_cats : ℕ := 40

/-- The number of cats in Cat Cafe Meow -/
def meow_cats : ℕ := total_cats - paw_cats

/-- The theorem stating that Cat Cafe Meow has 3 times as many cats as Cat Cafe Paw -/
theorem meow_to_paw_ratio : meow_cats = 3 * paw_cats := by
  sorry

end meow_to_paw_ratio_l3600_360002


namespace mayo_bottles_count_l3600_360031

/-- Given a ratio of ketchup : mustard : mayo bottles as 3 : 3 : 2, 
    and 6 ketchup bottles, prove that there are 4 mayo bottles. -/
theorem mayo_bottles_count 
  (ratio_ketchup : ℕ) 
  (ratio_mustard : ℕ) 
  (ratio_mayo : ℕ) 
  (ketchup_bottles : ℕ) 
  (h_ratio : ratio_ketchup = 3 ∧ ratio_mustard = 3 ∧ ratio_mayo = 2)
  (h_ketchup : ketchup_bottles = 6) : 
  ketchup_bottles * ratio_mayo / ratio_ketchup = 4 := by
sorry


end mayo_bottles_count_l3600_360031


namespace quadratic_real_solutions_range_l3600_360072

theorem quadratic_real_solutions_range (a : ℝ) :
  (∃ x : ℝ, x^2 - 4*x + a = 0) ↔ a ≤ 4 := by sorry

end quadratic_real_solutions_range_l3600_360072


namespace length_of_cd_l3600_360032

/-- Given points R and S on line segment CD, where R divides CD in the ratio 3:5,
    S divides CD in the ratio 4:7, and RS = 3, the length of CD is 264. -/
theorem length_of_cd (C D R S : Real) : 
  (∃ m n : Real, C + m = R ∧ R + n = D ∧ m / n = 3 / 5) →  -- R divides CD in ratio 3:5
  (∃ p q : Real, C + p = S ∧ S + q = D ∧ p / q = 4 / 7) →  -- S divides CD in ratio 4:7
  (S - R = 3) →                                            -- RS = 3
  (D - C = 264) :=                                         -- Length of CD is 264
by sorry

end length_of_cd_l3600_360032


namespace sin_70_degrees_l3600_360044

theorem sin_70_degrees (k : ℝ) (h : Real.sin (10 * π / 180) = k) :
  Real.sin (70 * π / 180) = 1 - 2 * k^2 := by
  sorry

end sin_70_degrees_l3600_360044


namespace max_value_of_trig_function_l3600_360037

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := fun x ↦ 2 * Real.sin x + 3 * Real.cos x
  ∃ M : ℝ, M = Real.sqrt 13 ∧ ∀ x : ℝ, f x ≤ M :=
by
  sorry

end max_value_of_trig_function_l3600_360037


namespace midpoint_endpoint_product_l3600_360085

/-- Given a segment CD with midpoint M and endpoint C, proves that the product of D's coordinates is -63 -/
theorem midpoint_endpoint_product (M C D : ℝ × ℝ) : 
  M = (4, -2) → C = (-1, 3) → M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → D.1 * D.2 = -63 := by
  sorry

end midpoint_endpoint_product_l3600_360085


namespace ratio_hcf_to_lcm_l3600_360086

/-- Given two positive integers with a ratio of 3:4 and HCF of 4, their LCM is 48 -/
theorem ratio_hcf_to_lcm (a b : ℕ+) (h_ratio : a.val * 4 = b.val * 3) (h_hcf : Nat.gcd a.val b.val = 4) :
  Nat.lcm a.val b.val = 48 := by
  sorry

end ratio_hcf_to_lcm_l3600_360086


namespace oliver_age_l3600_360041

/-- Given the ages of Oliver, Mia, and Lucas, prove that Oliver is 18 years old. -/
theorem oliver_age :
  ∀ (oliver_age mia_age lucas_age : ℕ),
    oliver_age = mia_age - 2 →
    mia_age = lucas_age + 5 →
    lucas_age = 15 →
    oliver_age = 18 := by
  sorry

end oliver_age_l3600_360041


namespace square_root_expression_evaluation_l3600_360035

theorem square_root_expression_evaluation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000000001 ∧ 
  |22 + Real.sqrt (-4 + 6 * 4 * 3) - 30.246211251| < ε := by
  sorry

end square_root_expression_evaluation_l3600_360035


namespace two_cookies_eaten_l3600_360025

/-- Given an initial number of cookies and the number of cookies left,
    calculate the number of cookies eaten. -/
def cookies_eaten (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

/-- Theorem: Given 7 initial cookies and 5 cookies left,
    prove that 2 cookies were eaten. -/
theorem two_cookies_eaten :
  cookies_eaten 7 5 = 2 := by
  sorry

end two_cookies_eaten_l3600_360025


namespace parabola_square_intersection_l3600_360065

/-- A parabola y = px^2 has a common point with the square defined by vertices A(1,1), B(2,1), C(2,2), and D(1,2) if and only if 1/4 ≤ p ≤ 2 -/
theorem parabola_square_intersection (p : ℝ) : 
  (∃ x y : ℝ, y = p * x^2 ∧ 
    ((x = 1 ∧ y = 1) ∨ 
     (x = 2 ∧ y = 1) ∨ 
     (x = 2 ∧ y = 2) ∨ 
     (x = 1 ∧ y = 2) ∨
     (1 ≤ x ∧ x ≤ 2 ∧ y = 1) ∨
     (x = 2 ∧ 1 ≤ y ∧ y ≤ 2) ∨
     (1 ≤ x ∧ x ≤ 2 ∧ y = 2) ∨
     (x = 1 ∧ 1 ≤ y ∧ y ≤ 2))) ↔ 
  (1/4 : ℝ) ≤ p ∧ p ≤ 2 :=
by sorry

end parabola_square_intersection_l3600_360065


namespace ceiling_minus_value_l3600_360059

theorem ceiling_minus_value (x ε : ℝ) 
  (h1 : ⌈x + ε⌉ - ⌊x + ε⌋ = 1) 
  (h2 : 0 < ε) 
  (h3 : ε < 1) : 
  ⌈x + ε⌉ - (x + ε) = 1 - ε := by
  sorry

end ceiling_minus_value_l3600_360059


namespace slide_count_l3600_360062

theorem slide_count (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 22 → additional = 13 → total = initial + additional → total = 35 := by
  sorry

end slide_count_l3600_360062


namespace tree_height_calculation_l3600_360006

/-- Given the height and shadow length of a person and the shadow length of a tree,
    calculate the height of the tree using the principle of similar triangles. -/
theorem tree_height_calculation (person_height person_shadow tree_shadow : ℝ)
  (h_person_height : person_height = 1.6)
  (h_person_shadow : person_shadow = 0.8)
  (h_tree_shadow : tree_shadow = 4.8)
  (h_positive : person_height > 0 ∧ person_shadow > 0 ∧ tree_shadow > 0) :
  (person_height / person_shadow) * tree_shadow = 9.6 :=
by
  sorry

#check tree_height_calculation

end tree_height_calculation_l3600_360006


namespace functional_equation_solution_l3600_360009

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
by sorry

end functional_equation_solution_l3600_360009


namespace train_speed_correct_l3600_360018

/-- The speed of the train in km/hr given the conditions -/
def train_speed : ℝ :=
  let train_length : ℝ := 110  -- meters
  let passing_time : ℝ := 4.399648028157747  -- seconds
  let man_speed : ℝ := 6  -- km/hr
  84  -- km/hr

/-- Theorem stating that the calculated train speed is correct -/
theorem train_speed_correct :
  let train_length : ℝ := 110  -- meters
  let passing_time : ℝ := 4.399648028157747  -- seconds
  let man_speed : ℝ := 6  -- km/hr
  train_speed = 84 := by sorry

end train_speed_correct_l3600_360018
