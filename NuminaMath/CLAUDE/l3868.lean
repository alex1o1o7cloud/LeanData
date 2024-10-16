import Mathlib

namespace NUMINAMATH_CALUDE_task_completion_probability_l3868_386837

theorem task_completion_probability (p1 p2 : ℚ) (h1 : p1 = 5/8) (h2 : p2 = 3/5) :
  p1 * (1 - p2) = 1/4 := by sorry

end NUMINAMATH_CALUDE_task_completion_probability_l3868_386837


namespace NUMINAMATH_CALUDE_wolf_catches_hare_in_problem_l3868_386851

/-- Represents the chase scenario between a wolf and a hare -/
structure ChaseScenario where
  initial_distance : ℝ
  hiding_spot_distance : ℝ
  wolf_speed : ℝ
  hare_speed : ℝ

/-- Determines if the wolf catches the hare in the given chase scenario -/
def wolf_catches_hare (scenario : ChaseScenario) : Prop :=
  let relative_speed := scenario.wolf_speed - scenario.hare_speed
  let chase_distance := scenario.hiding_spot_distance - scenario.initial_distance
  let chase_time := chase_distance / relative_speed
  scenario.hare_speed * chase_time ≤ scenario.hiding_spot_distance

/-- The specific chase scenario from the problem -/
def problem_scenario : ChaseScenario :=
  { initial_distance := 30
    hiding_spot_distance := 333
    wolf_speed := 600
    hare_speed := 550 }

/-- Theorem stating that the wolf catches the hare in the problem scenario -/
theorem wolf_catches_hare_in_problem : wolf_catches_hare problem_scenario := by
  sorry

end NUMINAMATH_CALUDE_wolf_catches_hare_in_problem_l3868_386851


namespace NUMINAMATH_CALUDE_gasoline_price_growth_rate_l3868_386841

theorem gasoline_price_growth_rate (initial_price final_price : ℝ) (months : ℕ) (x : ℝ) 
  (h1 : initial_price = 6.2)
  (h2 : final_price = 8.9)
  (h3 : months = 2)
  (h4 : x > 0)
  : initial_price * (1 + x)^months = final_price := by
  sorry

end NUMINAMATH_CALUDE_gasoline_price_growth_rate_l3868_386841


namespace NUMINAMATH_CALUDE_power_of_two_with_three_identical_digits_l3868_386892

theorem power_of_two_with_three_identical_digits :
  ∃ k : ℕ, k ≥ 10 ∧
  ∃ d : ℕ, d < 10 ∧
  (∃ n : ℕ, 2^k = 1000 * n + 111 * d) ∧
  (∀ m : ℕ, m < k → ¬∃ e : ℕ, e < 10 ∧ ∃ p : ℕ, 2^m = 1000 * p + 111 * e) ∧
  k = 39 :=
sorry

end NUMINAMATH_CALUDE_power_of_two_with_three_identical_digits_l3868_386892


namespace NUMINAMATH_CALUDE_tomato_seeds_problem_l3868_386869

/-- Represents the number of tomato seeds planted by Mike in the morning -/
def mike_morning : ℕ := sorry

/-- Represents the number of tomato seeds planted by Ted in the morning -/
def ted_morning : ℕ := sorry

/-- Represents the number of tomato seeds planted by Mike in the afternoon -/
def mike_afternoon : ℕ := 60

/-- Represents the number of tomato seeds planted by Ted in the afternoon -/
def ted_afternoon : ℕ := sorry

theorem tomato_seeds_problem :
  ted_morning = 2 * mike_morning ∧
  ted_afternoon = mike_afternoon - 20 ∧
  mike_morning + ted_morning + mike_afternoon + ted_afternoon = 250 →
  mike_morning = 50 := by sorry

end NUMINAMATH_CALUDE_tomato_seeds_problem_l3868_386869


namespace NUMINAMATH_CALUDE_employee_survey_40_50_l3868_386882

/-- Represents the number of employees to be selected in a stratified sampling -/
def stratified_sample (total : ℕ) (group : ℕ) (sample_size : ℕ) : ℚ :=
  (group : ℚ) / (total : ℚ) * (sample_size : ℚ)

/-- Proves that the number of employees aged 40-50 to be selected is 12 -/
theorem employee_survey_40_50 :
  let total_employees : ℕ := 350
  let over_50 : ℕ := 70
  let under_40 : ℕ := 175
  let survey_size : ℕ := 40
  let employees_40_50 : ℕ := total_employees - over_50 - under_40
  stratified_sample total_employees employees_40_50 survey_size = 12 := by
  sorry

end NUMINAMATH_CALUDE_employee_survey_40_50_l3868_386882


namespace NUMINAMATH_CALUDE_rectangle_cover_cost_l3868_386855

/-- Given a rectangle where the length is four times the width and the perimeter is 200 cm,
    the total cost to cover the rectangle at $5 per square centimeter is $8000. -/
theorem rectangle_cover_cost (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) :
  5 * (l * w) = 8000 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_cover_cost_l3868_386855


namespace NUMINAMATH_CALUDE_shane_sandwich_problem_l3868_386838

/-- The number of slices in each package of sliced bread -/
def slices_per_bread_package : ℕ := 20

/-- The number of packages of sliced bread Shane buys -/
def bread_packages : ℕ := 2

/-- The number of packages of sliced ham Shane buys -/
def ham_packages : ℕ := 2

/-- The number of ham slices in each package -/
def ham_slices_per_package : ℕ := 8

/-- The number of bread slices needed for each sandwich -/
def bread_slices_per_sandwich : ℕ := 2

/-- The number of bread slices leftover after making sandwiches -/
def leftover_bread_slices : ℕ := 8

theorem shane_sandwich_problem :
  slices_per_bread_package * bread_packages = 
    (ham_packages * ham_slices_per_package * bread_slices_per_sandwich) + leftover_bread_slices := by
  sorry

end NUMINAMATH_CALUDE_shane_sandwich_problem_l3868_386838


namespace NUMINAMATH_CALUDE_divisiblity_condition_l3868_386824

def recursive_sequence : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => (recursive_sequence (n + 1))^2 + recursive_sequence (n + 1) + 1 / recursive_sequence n

theorem divisiblity_condition (a b : ℕ) :
  a > 0 ∧ b > 0 →
  a ∣ (b^2 + b + 1) →
  b ∣ (a^2 + a + 1) →
  ((a = 1 ∧ b = 3) ∨ 
   (∃ n : ℕ, a = recursive_sequence n ∧ b = recursive_sequence (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_divisiblity_condition_l3868_386824


namespace NUMINAMATH_CALUDE_lcm_of_12_and_18_l3868_386856

theorem lcm_of_12_and_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_and_18_l3868_386856


namespace NUMINAMATH_CALUDE_symmetric_point_example_l3868_386804

/-- Given a point (x, y) in a 2D coordinate system, this function returns the point that is symmetric to (x, y) with respect to the origin. -/
def symmetricPointOrigin (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- Theorem stating that the point symmetric to (-2, 5) with respect to the origin is (2, -5). -/
theorem symmetric_point_example : symmetricPointOrigin (-2) 5 = (2, -5) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_example_l3868_386804


namespace NUMINAMATH_CALUDE_residue_of_11_pow_2010_mod_19_l3868_386842

theorem residue_of_11_pow_2010_mod_19 : (11 : ℤ) ^ 2010 ≡ 3 [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_residue_of_11_pow_2010_mod_19_l3868_386842


namespace NUMINAMATH_CALUDE_simplify_radical_product_l3868_386811

theorem simplify_radical_product (x : ℝ) (hx : x > 0) :
  Real.sqrt (45 * x) * Real.sqrt (32 * x) * Real.sqrt (18 * x) * (27 * x) ^ (1/3) = 72 * x ^ (1/3) * Real.sqrt (5 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l3868_386811


namespace NUMINAMATH_CALUDE_min_value_expression_l3868_386805

theorem min_value_expression (a b : ℝ) (h : a * b > 0) :
  (a^4 + 4*b^4 + 1) / (a*b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3868_386805


namespace NUMINAMATH_CALUDE_roof_area_l3868_386875

theorem roof_area (width length : ℝ) (h1 : length = 5 * width) (h2 : length - width = 48) :
  width * length = 720 := by
  sorry

end NUMINAMATH_CALUDE_roof_area_l3868_386875


namespace NUMINAMATH_CALUDE_smallest_angle_satisfying_trig_equation_l3868_386846

theorem smallest_angle_satisfying_trig_equation :
  ∃ y : ℝ, y > 0 ∧ y < (π / 180) * 360 ∧
  (∀ θ : ℝ, 0 < θ ∧ θ < y → ¬(Real.sin (4 * θ) * Real.sin (5 * θ) = Real.cos (4 * θ) * Real.cos (5 * θ))) ∧
  Real.sin (4 * y) * Real.sin (5 * y) = Real.cos (4 * y) * Real.cos (5 * y) ∧
  y = (π / 180) * 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_satisfying_trig_equation_l3868_386846


namespace NUMINAMATH_CALUDE_binomial_18_10_l3868_386897

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 45760 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l3868_386897


namespace NUMINAMATH_CALUDE_triangle_cos_inequality_l3868_386812

/-- For any real numbers A, B, C that are angles of a triangle, 
    the inequality 8 cos A · cos B · cos C ≤ 1 holds. -/
theorem triangle_cos_inequality (A B C : Real) (h : A + B + C = π) :
  8 * Real.cos A * Real.cos B * Real.cos C ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cos_inequality_l3868_386812


namespace NUMINAMATH_CALUDE_smallest_with_eight_odd_sixteen_even_divisors_l3868_386865

/-- Count of positive odd integer divisors of a number -/
def countOddDivisors (n : ℕ) : ℕ := sorry

/-- Count of positive even integer divisors of a number -/
def countEvenDivisors (n : ℕ) : ℕ := sorry

/-- Proposition: 3000 is the smallest positive integer with 8 odd and 16 even divisors -/
theorem smallest_with_eight_odd_sixteen_even_divisors :
  (∀ m : ℕ, m > 0 ∧ m < 3000 → 
    countOddDivisors m ≠ 8 ∨ countEvenDivisors m ≠ 16) ∧
  countOddDivisors 3000 = 8 ∧ 
  countEvenDivisors 3000 = 16 := by
  sorry

end NUMINAMATH_CALUDE_smallest_with_eight_odd_sixteen_even_divisors_l3868_386865


namespace NUMINAMATH_CALUDE_remaining_quantity_average_l3868_386894

theorem remaining_quantity_average (total : ℕ) (avg_all : ℚ) (avg_five : ℚ) (avg_two : ℚ) :
  total = 8 ∧ avg_all = 15 ∧ avg_five = 10 ∧ avg_two = 22 →
  (total * avg_all - 5 * avg_five - 2 * avg_two) = 26 := by
sorry

end NUMINAMATH_CALUDE_remaining_quantity_average_l3868_386894


namespace NUMINAMATH_CALUDE_coordinate_problem_l3868_386833

theorem coordinate_problem (x₁ y₁ x₂ y₂ : ℕ) : 
  (x₁ > 0) → (y₁ > 0) → (x₂ > 0) → (y₂ > 0) →  -- Positive integer coordinates
  (y₁ > x₁) →  -- Angle OA > 45°
  (x₂ > y₂) →  -- Angle OB < 45°
  (x₂ * y₂ = x₁ * y₁ + 67) →  -- Area difference condition
  (x₁ = 1 ∧ y₁ = 5 ∧ x₂ = 9 ∧ y₂ = 8) := by
sorry

end NUMINAMATH_CALUDE_coordinate_problem_l3868_386833


namespace NUMINAMATH_CALUDE_max_absolute_value_of_z_l3868_386876

theorem max_absolute_value_of_z (a b c z : ℂ) 
  (h1 : Complex.abs a = Complex.abs b)
  (h2 : Complex.abs b = Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : a * z^2 + b * z + c = 0) :
  Complex.abs z ≤ (1 + Real.sqrt 5) / 2 := by sorry

end NUMINAMATH_CALUDE_max_absolute_value_of_z_l3868_386876


namespace NUMINAMATH_CALUDE_product_increase_fifteen_times_l3868_386800

theorem product_increase_fifteen_times :
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℕ),
    ((a₁ - 3) * (a₂ - 3) * (a₃ - 3) * (a₄ - 3) * (a₅ - 3) : ℤ) = 
    15 * (a₁ * a₂ * a₃ * a₄ * a₅) := by
  sorry

end NUMINAMATH_CALUDE_product_increase_fifteen_times_l3868_386800


namespace NUMINAMATH_CALUDE_fruit_seller_pricing_l3868_386821

/-- Given a fruit seller's pricing scenario, calculate the current selling price. -/
theorem fruit_seller_pricing (loss_percentage : ℝ) (profit_percentage : ℝ) (profit_price : ℝ) :
  loss_percentage = 20 →
  profit_percentage = 5 →
  profit_price = 10.5 →
  ∃ (current_price : ℝ),
    current_price = (1 - loss_percentage / 100) * (profit_price / (1 + profit_percentage / 100)) ∧
    current_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_pricing_l3868_386821


namespace NUMINAMATH_CALUDE_angle_A_triangle_area_l3868_386871

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 7 ∧
  t.b = 2 ∧
  Real.sqrt 3 * t.b * Real.cos t.A = t.a * Real.sin t.B

-- Theorem for angle A
theorem angle_A (t : Triangle) (h : triangle_conditions t) : t.A = π / 3 :=
sorry

-- Theorem for area of triangle ABC
theorem triangle_area (t : Triangle) (h : triangle_conditions t) : 
  (1 / 2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_angle_A_triangle_area_l3868_386871


namespace NUMINAMATH_CALUDE_day_of_week_in_consecutive_years_l3868_386808

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  number : ℕ
  is_leap : Bool

/-- Returns the day of the week for a given day number in a year -/
def day_of_week (y : Year) (day_number : ℕ) : DayOfWeek :=
  sorry

/-- Returns the next year -/
def next_year (y : Year) : Year :=
  sorry

/-- Returns the previous year -/
def prev_year (y : Year) : Year :=
  sorry

theorem day_of_week_in_consecutive_years 
  (y : Year)
  (h1 : day_of_week y 250 = DayOfWeek.Friday)
  (h2 : day_of_week (next_year y) 150 = DayOfWeek.Friday) :
  day_of_week (prev_year y) 50 = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_day_of_week_in_consecutive_years_l3868_386808


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3868_386810

theorem cubic_equation_roots (x : ℝ) : 
  ∃ (r₁ r₂ r₃ : ℝ), 
    (r₁ < 0 ∧ r₂ < 0 ∧ r₃ > 0) ∧
    (∀ y : ℝ, y^3 + 3*y^2 - 4*y - 12 = 0 ↔ y = r₁ ∨ y = r₂ ∨ y = r₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3868_386810


namespace NUMINAMATH_CALUDE_sum_divisors_2_3_power_l3868_386864

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of divisors of 2^i * 3^j is 360, then i + j = 6 -/
theorem sum_divisors_2_3_power (i j : ℕ) :
  sum_of_divisors (2^i * 3^j) = 360 → i + j = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_divisors_2_3_power_l3868_386864


namespace NUMINAMATH_CALUDE_min_unboxed_balls_l3868_386830

/-- Represents the number of balls that can be stored in a big box -/
def big_box_capacity : ℕ := 25

/-- Represents the number of balls that can be stored in a small box -/
def small_box_capacity : ℕ := 20

/-- Represents the total number of balls to be stored -/
def total_balls : ℕ := 104

/-- 
Given:
- Big boxes can store 25 balls each
- Small boxes can store 20 balls each
- There are 104 balls to be stored

Prove that the minimum number of balls that cannot be completely boxed is 4.
-/
theorem min_unboxed_balls : 
  ∀ (big_boxes small_boxes : ℕ), 
    big_boxes * big_box_capacity + small_boxes * small_box_capacity ≤ total_balls →
    4 ≤ total_balls - (big_boxes * big_box_capacity + small_boxes * small_box_capacity) :=
by sorry

end NUMINAMATH_CALUDE_min_unboxed_balls_l3868_386830


namespace NUMINAMATH_CALUDE_no_integer_solution_l3868_386859

theorem no_integer_solution (P : Int → Int) (a b c : Int) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  P a = 2 ∧ P b = 2 ∧ P c = 2 →
  ∀ k : Int, P k ≠ 3 := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3868_386859


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_y_coord_l3868_386858

/-- The y-coordinate of the intersection point of perpendicular tangents to y = 4x^2 -/
theorem perpendicular_tangents_intersection_y_coord (c d : ℝ) : 
  (c ≠ d) →                                  -- Ensure C and D are distinct points
  (4 * c^2 = (4 : ℝ) * c^2) →                -- C is on the parabola y = 4x^2
  (4 * d^2 = (4 : ℝ) * d^2) →                -- D is on the parabola y = 4x^2
  ((8 : ℝ) * c * (8 * d) = -1) →             -- Tangent lines are perpendicular
  (4 : ℝ) * c * d = -(1/16) :=               -- y-coordinate of intersection point Q is -1/16
by sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_y_coord_l3868_386858


namespace NUMINAMATH_CALUDE_apple_distribution_l3868_386854

theorem apple_distribution (total_apples : Nat) (total_bags : Nat) (x : Nat) : 
  total_apples = 109 →
  total_bags = 20 →
  (∃ (a b : Nat), a + b = total_bags ∧ a * x + b * 3 = total_apples) →
  (x = 10 ∨ x = 52) := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l3868_386854


namespace NUMINAMATH_CALUDE_interest_calculation_l3868_386860

theorem interest_calculation (P : ℝ) : 
  P * (1 + 5/100)^2 - P - (P * 5 * 2 / 100) = 17 → P = 6800 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l3868_386860


namespace NUMINAMATH_CALUDE_tv_watching_weeks_l3868_386881

/-- Represents Flynn's TV watching habits and total time --/
structure TVWatching where
  weekdayMinutes : ℕ  -- Minutes watched per weekday night
  weekendHours : ℕ    -- Additional hours watched on weekends
  totalHours : ℕ      -- Total hours watched

/-- Calculates the number of weeks based on TV watching habits --/
def calculateWeeks (tw : TVWatching) : ℚ :=
  let weekdayHours : ℚ := (tw.weekdayMinutes * 5 : ℚ) / 60
  let totalWeeklyHours : ℚ := weekdayHours + tw.weekendHours
  tw.totalHours / totalWeeklyHours

/-- Theorem stating that 234 hours of TV watching corresponds to 52 weeks --/
theorem tv_watching_weeks (tw : TVWatching) 
  (h1 : tw.weekdayMinutes = 30)
  (h2 : tw.weekendHours = 2)
  (h3 : tw.totalHours = 234) :
  calculateWeeks tw = 52 := by
  sorry

end NUMINAMATH_CALUDE_tv_watching_weeks_l3868_386881


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a20_l3868_386879

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a20 (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 1 + a 3 + a 5 = 105 →
  a 2 + a 4 + a 6 = 99 →
  a 20 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a20_l3868_386879


namespace NUMINAMATH_CALUDE_parking_cost_proof_l3868_386895

-- Define the initial cost for up to 2 hours
def initial_cost : ℝ := 9

-- Define the total parking duration in hours
def total_hours : ℝ := 9

-- Define the average cost per hour for the total duration
def average_cost_per_hour : ℝ := 2.361111111111111

-- Define the cost for each hour in excess of 2 hours
def excess_hour_cost : ℝ := 1.75

-- Theorem statement
theorem parking_cost_proof :
  excess_hour_cost * (total_hours - 2) + initial_cost = average_cost_per_hour * total_hours :=
by sorry

end NUMINAMATH_CALUDE_parking_cost_proof_l3868_386895


namespace NUMINAMATH_CALUDE_retail_profit_calculation_l3868_386877

/-- Represents the pricing and profit calculations for a retail scenario -/
def RetailScenario (costPrice : ℝ) : Prop :=
  let markupPercentage : ℝ := 65
  let discountPercentage : ℝ := 25
  let actualProfitPercentage : ℝ := 23.75
  let markedPrice : ℝ := costPrice * (1 + markupPercentage / 100)
  let sellingPrice : ℝ := markedPrice * (1 - discountPercentage / 100)
  let actualProfit : ℝ := sellingPrice - costPrice
  let intendedProfit : ℝ := markedPrice - costPrice
  (actualProfit / costPrice * 100 = actualProfitPercentage) ∧
  (intendedProfit / costPrice * 100 = markupPercentage)

/-- Theorem stating that under the given retail scenario, the initially expected profit percentage is 65% -/
theorem retail_profit_calculation (costPrice : ℝ) (h : costPrice > 0) :
  RetailScenario costPrice → 65 = (65 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_retail_profit_calculation_l3868_386877


namespace NUMINAMATH_CALUDE_power_expression_evaluation_l3868_386831

theorem power_expression_evaluation (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_evaluation_l3868_386831


namespace NUMINAMATH_CALUDE_cos_105_degrees_l3868_386878

theorem cos_105_degrees : Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l3868_386878


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l3868_386801

/-- Two points are symmetric about the y-axis if their y-coordinates are equal and their x-coordinates are opposite -/
def symmetric_about_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  y1 = y2 ∧ x1 = -x2

/-- The problem statement -/
theorem symmetric_points_difference (m n : ℝ) :
  symmetric_about_y_axis 3 m n 4 → m - n = 7 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l3868_386801


namespace NUMINAMATH_CALUDE_equation_solutions_l3868_386857

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 = 169 ↔ x = 15 ∨ x = -11) ∧
  (∀ x : ℝ, 3*(x - 3)^3 - 24 = 0 ↔ x = 5) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3868_386857


namespace NUMINAMATH_CALUDE_algae_growth_theorem_l3868_386828

/-- The time (in hours) for an algae population to grow from 200 to 145,800 cells, tripling every 3 hours. -/
def algae_growth_time : ℕ :=
  18

theorem algae_growth_theorem (initial_population : ℕ) (final_population : ℕ) (growth_factor : ℕ) (growth_interval : ℕ) :
  initial_population = 200 →
  final_population = 145800 →
  growth_factor = 3 →
  growth_interval = 3 →
  (growth_factor ^ (algae_growth_time / growth_interval)) * initial_population = final_population :=
by
  sorry

#check algae_growth_theorem

end NUMINAMATH_CALUDE_algae_growth_theorem_l3868_386828


namespace NUMINAMATH_CALUDE_gcd_problem_l3868_386886

theorem gcd_problem (a : ℕ+) : (Nat.gcd (Nat.gcd a 16) (Nat.gcd 18 a) = 2) → (a = 2) :=
by sorry

end NUMINAMATH_CALUDE_gcd_problem_l3868_386886


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3868_386898

/-- Given a quadratic function y = x^2 - px + q where the minimum value of y is 1,
    prove that q = 1 + (p^2 / 4) -/
theorem quadratic_minimum (p q : ℝ) : 
  (∀ x, x^2 - p*x + q ≥ 1) ∧ (∃ x, x^2 - p*x + q = 1) → 
  q = 1 + p^2 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3868_386898


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l3868_386816

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 + 3 * x

-- State the theorem
theorem function_satisfies_conditions :
  (f 1 = 1) ∧ 
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l3868_386816


namespace NUMINAMATH_CALUDE_hexagon_perimeter_sum_l3868_386891

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  perimeter : ℝ

/-- Represents a hexagon formed by two equilateral triangles -/
def hexagon_from_triangles (t1 t2 : EquilateralTriangle) : ℝ := t1.perimeter + t2.perimeter

theorem hexagon_perimeter_sum (t1 t2 : EquilateralTriangle) 
  (h1 : t1.perimeter = 12) (h2 : t2.perimeter = 15) : 
  hexagon_from_triangles t1 t2 = 27 := by
  sorry

#check hexagon_perimeter_sum

end NUMINAMATH_CALUDE_hexagon_perimeter_sum_l3868_386891


namespace NUMINAMATH_CALUDE_equal_prob_without_mult_higher_prob_even_with_mult_l3868_386868

/-- Represents a calculator with basic operations -/
structure Calculator where
  /-- The current display value -/
  display : ℕ
  /-- Whether multiplication is available -/
  mult_available : Bool

/-- Represents the parity of a number -/
inductive Parity
  | Even
  | Odd

/-- Get the parity of a natural number -/
def getParity (n : ℕ) : Parity :=
  if n % 2 = 0 then Parity.Even else Parity.Odd

/-- The probability of getting an odd result after a sequence of operations -/
def probOddResult (c : Calculator) : ℝ :=
  sorry

theorem equal_prob_without_mult (c : Calculator) (h : c.mult_available = false) :
  probOddResult c = 1 / 2 :=
sorry

theorem higher_prob_even_with_mult (c : Calculator) (h : c.mult_available = true) :
  probOddResult c < 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_equal_prob_without_mult_higher_prob_even_with_mult_l3868_386868


namespace NUMINAMATH_CALUDE_sequence_a_formula_l3868_386880

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 5
  | (n + 2) => (2 * (sequence_a (n + 1))^2 - 3 * sequence_a (n + 1) - 9) / (2 * sequence_a n)

theorem sequence_a_formula : ∀ n : ℕ, sequence_a n = 2^(n + 2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_formula_l3868_386880


namespace NUMINAMATH_CALUDE_max_min_x_squared_l3868_386832

def f (x : ℝ) : ℝ := x^2

theorem max_min_x_squared :
  ∃ (max min : ℝ), 
    (∀ x, -3 ≤ x ∧ x ≤ 1 → f x ≤ max) ∧
    (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = max) ∧
    (∀ x, -3 ≤ x ∧ x ≤ 1 → min ≤ f x) ∧
    (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = min) ∧
    max = 9 ∧ min = 0 := by
  sorry

end NUMINAMATH_CALUDE_max_min_x_squared_l3868_386832


namespace NUMINAMATH_CALUDE_dime_probability_l3868_386813

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℚ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℚ
  | _ => 1250

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℚ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℚ := coinCount Coin.Quarter + coinCount Coin.Dime + coinCount Coin.Penny

/-- The probability of selecting a dime from the jar -/
def probDime : ℚ := coinCount Coin.Dime / totalCoins

theorem dime_probability : probDime = 5 / 57 := by
  sorry

end NUMINAMATH_CALUDE_dime_probability_l3868_386813


namespace NUMINAMATH_CALUDE_project_bolts_boxes_l3868_386803

/-- The number of bolts in each box of bolts -/
def bolts_per_box : ℕ := 11

/-- The number of boxes of nuts purchased -/
def boxes_of_nuts : ℕ := 3

/-- The number of nuts in each box of nuts -/
def nuts_per_box : ℕ := 15

/-- The number of bolts left over -/
def bolts_leftover : ℕ := 3

/-- The number of nuts left over -/
def nuts_leftover : ℕ := 6

/-- The total number of bolts and nuts used for the project -/
def total_used : ℕ := 113

/-- The minimum number of boxes of bolts purchased -/
def min_boxes_of_bolts : ℕ := 7

theorem project_bolts_boxes :
  ∃ (boxes_of_bolts : ℕ),
    boxes_of_bolts * bolts_per_box ≥
      total_used - (boxes_of_nuts * nuts_per_box - nuts_leftover) + bolts_leftover ∧
    boxes_of_bolts = min_boxes_of_bolts :=
by sorry

end NUMINAMATH_CALUDE_project_bolts_boxes_l3868_386803


namespace NUMINAMATH_CALUDE_division_problem_l3868_386888

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 55053 → 
  divisor = 456 → 
  remainder = 333 → 
  dividend = divisor * quotient + remainder → 
  quotient = 120 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3868_386888


namespace NUMINAMATH_CALUDE_evaluate_expression_l3868_386826

theorem evaluate_expression (x : ℝ) (h : x = -3) :
  (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3868_386826


namespace NUMINAMATH_CALUDE_cos_a2_a12_equals_half_l3868_386850

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem cos_a2_a12_equals_half
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_condition : a 1 * a 13 + 2 * (a 7)^2 = 5 * Real.pi) :
  Real.cos (a 2 * a 12) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cos_a2_a12_equals_half_l3868_386850


namespace NUMINAMATH_CALUDE_triangle_squares_area_sum_l3868_386827

/-- Given a right triangle EAB with BE = 12 and another right triangle EAH with AH = 5,
    the sum of the areas of squares ABCD, AEFG, and AHIJ is equal to 169 square units. -/
theorem triangle_squares_area_sum : 
  ∀ (A B C D E F G H I J : ℝ × ℝ),
  let ab := dist A B
  let ae := dist A E
  let ah := dist A H
  let be := dist B E
  -- Angle EAB is a right angle
  (ab ^ 2 + ae ^ 2 = be ^ 2) →
  -- BE = 12 units
  (be = 12) →
  -- Triangle EAH is a right triangle
  (ae ^ 2 + ah ^ 2 = (dist E H) ^ 2) →
  -- AH = 5 units
  (ah = 5) →
  -- The sum of the areas of squares ABCD, AEFG, and AHIJ is 169
  (ab ^ 2 + ae ^ 2 + (dist E H) ^ 2 = 169) := by
  sorry


end NUMINAMATH_CALUDE_triangle_squares_area_sum_l3868_386827


namespace NUMINAMATH_CALUDE_remaining_flight_time_l3868_386809

def flight_duration : ℕ := 10 * 60  -- in minutes
def tv_episode_duration : ℕ := 25  -- in minutes
def num_tv_episodes : ℕ := 3
def sleep_duration : ℕ := 270  -- 4.5 hours in minutes
def movie_duration : ℕ := 105  -- 1 hour 45 minutes in minutes
def num_movies : ℕ := 2

theorem remaining_flight_time :
  flight_duration - (num_tv_episodes * tv_episode_duration + sleep_duration + num_movies * movie_duration) = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_flight_time_l3868_386809


namespace NUMINAMATH_CALUDE_not_all_equilateral_triangles_congruent_l3868_386862

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for equilateral triangles
def congruent (t1 t2 : EquilateralTriangle) : Prop :=
  t1.side_length = t2.side_length

-- Theorem statement
theorem not_all_equilateral_triangles_congruent :
  ∃ (t1 t2 : EquilateralTriangle), ¬(congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_not_all_equilateral_triangles_congruent_l3868_386862


namespace NUMINAMATH_CALUDE_g_neg_two_equals_neg_seventeen_l3868_386863

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 1

-- State the theorem
theorem g_neg_two_equals_neg_seventeen
  (h1 : ∀ x, f x + 2 * x^2 = -(f (-x) + 2 * (-x)^2)) -- y = f(x) + 2x^2 is odd
  (h2 : f 2 = 2) -- f(2) = 2
  : g f (-2) = -17 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_two_equals_neg_seventeen_l3868_386863


namespace NUMINAMATH_CALUDE_negative_half_times_negative_two_l3868_386819

theorem negative_half_times_negative_two : (-1/2 : ℚ) * (-2 : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_half_times_negative_two_l3868_386819


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l3868_386815

theorem consecutive_integers_sum_of_cubes (n : ℤ) : 
  n > 0 ∧ (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 11534 →
  (n - 1)^3 + n^3 + (n + 1)^3 + (n + 2)^3 = 74836 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l3868_386815


namespace NUMINAMATH_CALUDE_inequality_proof_l3868_386893

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) :
  (c - a < c - b) ∧ (a⁻¹ * c > b⁻¹ * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3868_386893


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3868_386861

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x * Real.exp x > 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ * Real.exp x₀ ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3868_386861


namespace NUMINAMATH_CALUDE_technicians_in_exchange_group_and_expectation_l3868_386807

/-- Represents the distribution of job certificates --/
structure JobCertificates where
  junior : Nat
  intermediate : Nat
  senior : Nat
  technician : Nat
  seniorTechnician : Nat

/-- The total number of apprentices --/
def totalApprentices : Nat := 200

/-- The distribution of job certificates --/
def certificateDistribution : JobCertificates :=
  { junior := 20
  , intermediate := 60
  , senior := 60
  , technician := 40
  , seniorTechnician := 20 }

/-- The number of people selected for the exchange group --/
def exchangeGroupSize : Nat := 10

/-- The number of people chosen as representatives to speak --/
def speakersSize : Nat := 3

/-- Theorem stating the number of technicians in the exchange group and the expected number of technicians among speakers --/
theorem technicians_in_exchange_group_and_expectation :
  let totalTechnicians := certificateDistribution.technician + certificateDistribution.seniorTechnician
  let techniciansInExchangeGroup := (totalTechnicians * exchangeGroupSize) / totalApprentices
  let expectationOfTechnicians : Rat := 9 / 10
  techniciansInExchangeGroup = 3 ∧ 
  expectationOfTechnicians = (0 * (7 / 24 : Rat) + 1 * (21 / 40 : Rat) + 2 * (7 / 40 : Rat) + 3 * (1 / 120 : Rat)) := by
  sorry

end NUMINAMATH_CALUDE_technicians_in_exchange_group_and_expectation_l3868_386807


namespace NUMINAMATH_CALUDE_data_instances_eq_720_l3868_386847

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The interval between recordings in seconds -/
def recording_interval : ℕ := 5

/-- The number of data instances recorded in one hour by a device that records every 5 seconds -/
def data_instances : ℕ := 
  (seconds_per_minute * minutes_per_hour) / recording_interval

/-- Theorem: The number of data instances recorded in one hour is 720 -/
theorem data_instances_eq_720 : data_instances = 720 := by
  sorry

end NUMINAMATH_CALUDE_data_instances_eq_720_l3868_386847


namespace NUMINAMATH_CALUDE_probability_red_then_blue_l3868_386829

def red_marbles : ℕ := 4
def white_marbles : ℕ := 5
def blue_marbles : ℕ := 3

def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

theorem probability_red_then_blue : 
  (red_marbles : ℚ) / total_marbles * blue_marbles / (total_marbles - 1) = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_then_blue_l3868_386829


namespace NUMINAMATH_CALUDE_no_real_solutions_for_abs_equation_l3868_386834

theorem no_real_solutions_for_abs_equation : 
  ¬ ∃ x : ℝ, |x^2 - 3| = 2*x + 6 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_abs_equation_l3868_386834


namespace NUMINAMATH_CALUDE_no_primes_in_factorial_range_l3868_386848

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 1) :
  ∀ k ∈ Set.Ioo (n! - (n - 1)) n!, ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_factorial_range_l3868_386848


namespace NUMINAMATH_CALUDE_fast_food_cost_correct_l3868_386873

/-- The cost of fast food given the number of servings of each type -/
def fast_food_cost (a b : ℕ) : ℕ := 30 * a + 20 * b

/-- Theorem stating that the cost of fast food is calculated correctly -/
theorem fast_food_cost_correct (a b : ℕ) : 
  fast_food_cost a b = 30 * a + 20 * b := by
  sorry

end NUMINAMATH_CALUDE_fast_food_cost_correct_l3868_386873


namespace NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_for_q_l3868_386874

theorem p_neither_sufficient_nor_necessary_for_q :
  ¬(∀ x y : ℝ, x + y ≠ -2 → (x ≠ -1 ∧ y ≠ -1)) ∧
  ¬(∀ x y : ℝ, (x ≠ -1 ∧ y ≠ -1) → x + y ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_for_q_l3868_386874


namespace NUMINAMATH_CALUDE_max_value_of_4x_plus_3y_l3868_386866

theorem max_value_of_4x_plus_3y (x y : ℝ) : 
  x^2 + y^2 = 16*x + 8*y + 10 → (4*x + 3*y ≤ 42) ∧ ∃ x y, x^2 + y^2 = 16*x + 8*y + 10 ∧ 4*x + 3*y = 42 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_4x_plus_3y_l3868_386866


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3868_386884

/-- Given a real number m, proves that if the solution to the system of linear inequalities
    (2x - 1 > 3(x - 2) and x < m) is x < 5, then m ≥ 5. -/
theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (2*x - 1 > 3*(x - 2) ∧ x < m) ↔ x < 5) → m ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3868_386884


namespace NUMINAMATH_CALUDE_second_prime_range_l3868_386896

theorem second_prime_range (p q : ℕ) (hp : Prime p) (hq : Prime q) : 
  15 < p * q ∧ p * q ≤ 70 ∧ 2 < p ∧ p < 6 ∧ p * q = 69 → q = 23 := by
  sorry

end NUMINAMATH_CALUDE_second_prime_range_l3868_386896


namespace NUMINAMATH_CALUDE_roots_sum_powers_l3868_386820

theorem roots_sum_powers (γ δ : ℝ) : 
  γ^2 - 5*γ + 6 = 0 → δ^2 - 5*δ + 6 = 0 → 8*γ^5 + 15*δ^4 = 8425 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l3868_386820


namespace NUMINAMATH_CALUDE_initial_phone_count_prove_initial_phone_count_l3868_386840

theorem initial_phone_count : ℕ → Prop :=
  fun initial_count =>
    let defective_count : ℕ := 5
    let customer_a_bought : ℕ := 3
    let customer_b_bought : ℕ := 5
    let customer_c_bought : ℕ := 7
    let total_sold := customer_a_bought + customer_b_bought + customer_c_bought
    initial_count - defective_count = total_sold ∧ initial_count = 20

theorem prove_initial_phone_count :
  ∃ (x : ℕ), initial_phone_count x :=
sorry

end NUMINAMATH_CALUDE_initial_phone_count_prove_initial_phone_count_l3868_386840


namespace NUMINAMATH_CALUDE_cosine_domain_range_l3868_386889

theorem cosine_domain_range (a b : Real) (f : Real → Real) : 
  (∀ x ∈ Set.Icc a b, f x = Real.cos x) →
  (∀ x ∈ Set.Icc a b, -1/2 ≤ f x ∧ f x ≤ 1) →
  b - a ≠ π/3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_domain_range_l3868_386889


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3868_386885

theorem power_fraction_simplification :
  (5^2022)^2 - (5^2020)^2 / (5^2021)^2 - (5^2019)^2 = 5^2 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3868_386885


namespace NUMINAMATH_CALUDE_some_number_value_l3868_386818

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * n * 49) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3868_386818


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3868_386890

theorem scientific_notation_equivalence : 11.3 * (10 ^ 3) = 1.13 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3868_386890


namespace NUMINAMATH_CALUDE_spending_fraction_is_three_fourths_l3868_386817

/-- Represents a person's monthly savings and spending habits -/
structure SavingsHabit where
  monthly_salary : ℝ
  savings_fraction : ℝ
  spending_fraction : ℝ
  savings_fraction_nonneg : 0 ≤ savings_fraction
  spending_fraction_nonneg : 0 ≤ spending_fraction
  fractions_sum_to_one : savings_fraction + spending_fraction = 1

/-- The theorem stating that if yearly savings are 4 times monthly spending, 
    then the spending fraction is 3/4 -/
theorem spending_fraction_is_three_fourths 
  (habit : SavingsHabit) 
  (yearly_savings_eq_four_times_monthly_spending : 
    12 * habit.savings_fraction * habit.monthly_salary = 
    4 * habit.spending_fraction * habit.monthly_salary) :
  habit.spending_fraction = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_spending_fraction_is_three_fourths_l3868_386817


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3868_386853

theorem polynomial_expansion (t : ℝ) : 
  (3 * t^3 - 4 * t + 1) * (4 * t^2 - 5 * t + 3) = 
  12 * t^5 - 15 * t^4 - 7 * t^3 + 24 * t^2 - 17 * t + 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3868_386853


namespace NUMINAMATH_CALUDE_calculation_result_l3868_386823

theorem calculation_result : 1 + 0.1 - 0.1 + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l3868_386823


namespace NUMINAMATH_CALUDE_berrys_friday_temperature_l3868_386845

/-- Given Berry's temperatures for 6 days and the average for a week, 
    prove that his temperature on Friday was 99 degrees. -/
theorem berrys_friday_temperature 
  (temps : List ℝ) 
  (h_temps : temps = [99.1, 98.2, 98.7, 99.3, 99.8, 98.9]) 
  (h_avg : (temps.sum + x) / 7 = 99) : x = 99 := by
  sorry

end NUMINAMATH_CALUDE_berrys_friday_temperature_l3868_386845


namespace NUMINAMATH_CALUDE_tangent_circles_ratio_l3868_386806

/-- Two circles touching internally with specific tangent properties -/
structure TangentCircles where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  touch_internally : R > r  -- Circles touch internally
  radii_angle : ℝ  -- Angle between the two radii of the larger circle
  radii_tangent : Bool  -- The two radii are tangent to the smaller circle

/-- Theorem stating the ratio of radii for circles with specific tangent properties -/
theorem tangent_circles_ratio 
  (c : TangentCircles) 
  (h1 : c.radii_angle = 60) 
  (h2 : c.radii_tangent = true) : 
  c.R / c.r = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_circles_ratio_l3868_386806


namespace NUMINAMATH_CALUDE_moss_pollen_diameter_scientific_notation_l3868_386872

/-- Expresses a given decimal number in scientific notation -/
def scientificNotation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem moss_pollen_diameter_scientific_notation :
  scientificNotation 0.0000084 = (8.4, -6) := by sorry

end NUMINAMATH_CALUDE_moss_pollen_diameter_scientific_notation_l3868_386872


namespace NUMINAMATH_CALUDE_stadium_ratio_l3868_386843

theorem stadium_ratio (initial_total : ℕ) (initial_girls : ℕ) (final_total : ℕ) 
  (h1 : initial_total = 600)
  (h2 : initial_girls = 240)
  (h3 : final_total = 480)
  (h4 : (initial_total - initial_girls) / 4 + (initial_girls - (initial_total - final_total - (initial_total - initial_girls) / 4)) = initial_girls) :
  (initial_total - final_total - (initial_total - initial_girls) / 4) / initial_girls = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_stadium_ratio_l3868_386843


namespace NUMINAMATH_CALUDE_root_product_theorem_l3868_386852

theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ : ℂ) : 
  (x₁^5 + x₁^2 + 1 = 0) → 
  (x₂^5 + x₂^2 + 1 = 0) → 
  (x₃^5 + x₃^2 + 1 = 0) → 
  (x₄^5 + x₄^2 + 1 = 0) → 
  (x₅^5 + x₅^2 + 1 = 0) → 
  (x₁^3 - 2) * (x₂^3 - 2) * (x₃^3 - 2) * (x₄^3 - 2) * (x₅^3 - 2) = -243 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3868_386852


namespace NUMINAMATH_CALUDE_new_average_weight_l3868_386883

theorem new_average_weight 
  (A B C D E : ℝ)
  (h1 : (A + B + C) / 3 = 70)
  (h2 : (A + B + C + D) / 4 = 70)
  (h3 : E = D + 3)
  (h4 : A = 81) :
  (B + C + D + E) / 4 = 68 := by
sorry

end NUMINAMATH_CALUDE_new_average_weight_l3868_386883


namespace NUMINAMATH_CALUDE_pens_sold_correct_solve_paul_pens_problem_l3868_386839

/-- Calculates the number of pens sold given the initial and final counts -/
def pens_sold (initial : ℕ) (final : ℕ) : ℕ :=
  initial - final

theorem pens_sold_correct (initial final : ℕ) (h : initial ≥ final) :
  pens_sold initial final = initial - final :=
by sorry

/-- The specific problem instance -/
def paul_pens_problem : Prop :=
  pens_sold 106 14 = 92

theorem solve_paul_pens_problem : paul_pens_problem :=
by sorry

end NUMINAMATH_CALUDE_pens_sold_correct_solve_paul_pens_problem_l3868_386839


namespace NUMINAMATH_CALUDE_negation_of_squared_plus_one_geq_one_l3868_386835

theorem negation_of_squared_plus_one_geq_one :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_squared_plus_one_geq_one_l3868_386835


namespace NUMINAMATH_CALUDE_train_A_speed_l3868_386887

/-- The speed of Train A in miles per hour -/
def speed_A : ℝ := 30

/-- The speed of Train B in miles per hour -/
def speed_B : ℝ := 38

/-- The time difference between Train A and Train B departures in hours -/
def time_diff : ℝ := 2

/-- The distance from the station where Train B overtakes Train A in miles -/
def overtake_distance : ℝ := 285

/-- Theorem stating that the speed of Train A is 30 miles per hour -/
theorem train_A_speed :
  speed_A = 30 ∧
  speed_A * (overtake_distance / speed_B + time_diff) = overtake_distance :=
sorry

end NUMINAMATH_CALUDE_train_A_speed_l3868_386887


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l3868_386802

theorem boys_to_girls_ratio (T : ℚ) (G : ℚ) (h : T > 0) (h1 : G > 0) (h2 : (1/2) * G = (1/6) * T) :
  (T - G) / G = 2 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l3868_386802


namespace NUMINAMATH_CALUDE_inequality_proof_l3868_386867

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) :
  x^2 * y^2 + |x^2 - y^2| ≤ π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3868_386867


namespace NUMINAMATH_CALUDE_sum_of_squares_l3868_386870

theorem sum_of_squares : 17^2 + 19^2 + 23^2 + 29^2 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3868_386870


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3868_386836

theorem min_distance_to_line (x y : ℝ) (h : 2 * x + y + 5 = 0) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 5 ∧
  ∀ (a b : ℝ), 2 * a + b + 5 = 0 → Real.sqrt (a^2 + b^2) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3868_386836


namespace NUMINAMATH_CALUDE_trig_identity_quadratic_equation_solution_l3868_386844

-- Problem 1
theorem trig_identity : 
  Real.sin (π / 4) - 3 * Real.tan (π / 6) + Real.sqrt 2 * Real.cos (π / 3) = Real.sqrt 2 - Real.sqrt 3 := by
  sorry

-- Problem 2
theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x + 8
  ∀ x : ℝ, f x = 0 ↔ x = 4 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_quadratic_equation_solution_l3868_386844


namespace NUMINAMATH_CALUDE_coin_difference_l3868_386825

/-- Represents the denominations of coins available -/
inductive Coin
  | fiveCent
  | twentyCent
  | fiftyCent

/-- The value of each coin in cents -/
def coinValue : Coin → Nat
  | Coin.fiveCent => 5
  | Coin.twentyCent => 20
  | Coin.fiftyCent => 50

/-- The amount to be paid in cents -/
def amountToPay : Nat := 50

/-- A function that calculates the minimum number of coins needed -/
def minCoins : Nat := sorry

/-- A function that calculates the maximum number of coins needed -/
def maxCoins : Nat := sorry

/-- Theorem stating the difference between max and min number of coins -/
theorem coin_difference : maxCoins - minCoins = 9 := by sorry

end NUMINAMATH_CALUDE_coin_difference_l3868_386825


namespace NUMINAMATH_CALUDE_fixed_point_on_circle_l3868_386822

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*m*y + 6*m - 2 = 0

-- Theorem statement
theorem fixed_point_on_circle :
  ∀ m : ℝ, circle_equation 1 1 m ∨ circle_equation (1/5) (7/5) m :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_on_circle_l3868_386822


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3868_386899

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x + y > 2 → max x y > 1) ∧
  ¬(max x y > 1 → x + y > 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3868_386899


namespace NUMINAMATH_CALUDE_seventeen_sided_polygon_diagonals_l3868_386849

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 17 sides has 119 diagonals -/
theorem seventeen_sided_polygon_diagonals :
  num_diagonals 17 = 119 := by sorry

end NUMINAMATH_CALUDE_seventeen_sided_polygon_diagonals_l3868_386849


namespace NUMINAMATH_CALUDE_function_inequality_l3868_386814

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x > 0, x * (f' x) + x^2 < f x) :
  2 * f 1 > f 2 + 2 ∧ 3 * f 1 > f 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3868_386814
