import Mathlib

namespace NUMINAMATH_CALUDE_sum_ends_in_zero_squares_same_last_digit_l2971_297125

theorem sum_ends_in_zero_squares_same_last_digit (a b : ℤ) :
  (a + b) % 10 = 0 → (a ^ 2) % 10 = (b ^ 2) % 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_ends_in_zero_squares_same_last_digit_l2971_297125


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2971_297104

def divisible_by_2_or_5_ends_with_0 (n : ℕ) : Prop :=
  (n % 2 = 0 ∨ n % 5 = 0) → n % 10 = 0

def last_digit_not_0_not_divisible_by_2_and_5 (n : ℕ) : Prop :=
  n % 10 ≠ 0 → (n % 2 ≠ 0 ∧ n % 5 ≠ 0)

theorem contrapositive_equivalence :
  ∀ n : ℕ, divisible_by_2_or_5_ends_with_0 n ↔ last_digit_not_0_not_divisible_by_2_and_5 n :=
by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2971_297104


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2971_297101

theorem opposite_of_negative_two (a : ℝ) : a = -(- 2) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2971_297101


namespace NUMINAMATH_CALUDE_base5_division_l2971_297129

-- Define a function to convert from base 5 to base 10
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define the dividend and divisor in base 5
def dividend : List Nat := [4, 0, 2, 3, 1]  -- 13204₅
def divisor : List Nat := [3, 2]  -- 23₅

-- Define the expected quotient and remainder in base 5
def expectedQuotient : List Nat := [1, 1, 3]  -- 311₅
def expectedRemainder : Nat := 1  -- 1₅

-- Theorem statement
theorem base5_division :
  let dividend10 := base5ToBase10 dividend
  let divisor10 := base5ToBase10 divisor
  let quotient10 := dividend10 / divisor10
  let remainder10 := dividend10 % divisor10
  base5ToBase10 expectedQuotient = quotient10 ∧
  expectedRemainder = remainder10 := by
  sorry


end NUMINAMATH_CALUDE_base5_division_l2971_297129


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l2971_297118

theorem paper_clip_distribution (total_clips : ℕ) (num_boxes : ℕ) (clips_per_box : ℕ) :
  total_clips = 81 →
  num_boxes = 9 →
  total_clips = num_boxes * clips_per_box →
  clips_per_box = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l2971_297118


namespace NUMINAMATH_CALUDE_unit_vector_d_l2971_297139

def d : ℝ × ℝ := (12, -5)

theorem unit_vector_d :
  let magnitude := Real.sqrt (d.1 ^ 2 + d.2 ^ 2)
  (d.1 / magnitude, d.2 / magnitude) = (12 / 13, -5 / 13) := by
  sorry

end NUMINAMATH_CALUDE_unit_vector_d_l2971_297139


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2971_297100

-- Define the ratio of quantities for models A, B, and C
def ratio_A : ℕ := 2
def ratio_B : ℕ := 3
def ratio_C : ℕ := 4

-- Define the number of units of model A in the sample
def units_A : ℕ := 16

-- Define the total sample size
def sample_size : ℕ := units_A + (ratio_B * units_A / ratio_A) + (ratio_C * units_A / ratio_A)

-- Theorem statement
theorem stratified_sample_size :
  sample_size = 72 :=
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2971_297100


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2971_297194

theorem polynomial_division_theorem (x : ℝ) :
  x^5 - x^4 + x^3 - 9 = (x - 1) * (x^4 - x^3 + x^2 - x + 1) + (-9) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2971_297194


namespace NUMINAMATH_CALUDE_distance_range_for_intersecting_circles_l2971_297191

-- Define the radii of the two circles
def r₁ : ℝ := 3
def r₂ : ℝ := 5

-- Define the property of intersection
def intersecting (d : ℝ) : Prop := d < r₁ + r₂ ∧ d > abs (r₁ - r₂)

-- Theorem statement
theorem distance_range_for_intersecting_circles (d : ℝ) 
  (h : intersecting d) : 2 < d ∧ d < 8 := by sorry

end NUMINAMATH_CALUDE_distance_range_for_intersecting_circles_l2971_297191


namespace NUMINAMATH_CALUDE_two_incorrect_statements_l2971_297165

/-- Represents the coefficients of a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the roots of a quadratic equation -/
structure QuadraticRoots where
  x₁ : ℝ
  x₂ : ℝ

/-- Predicate to check if both roots are positive -/
def both_roots_positive (roots : QuadraticRoots) : Prop :=
  roots.x₁ > 0 ∧ roots.x₂ > 0

/-- Predicate to check if the given coefficients satisfy Vieta's formulas for the given roots -/
def satisfies_vieta (coeff : QuadraticCoefficients) (roots : QuadraticRoots) : Prop :=
  roots.x₁ + roots.x₂ = -coeff.b / coeff.a ∧ roots.x₁ * roots.x₂ = coeff.c / coeff.a

/-- The four statements about the signs of coefficients -/
def statement_1 (coeff : QuadraticCoefficients) : Prop := coeff.a > 0 ∧ coeff.b > 0 ∧ coeff.c > 0
def statement_2 (coeff : QuadraticCoefficients) : Prop := coeff.a < 0 ∧ coeff.b < 0 ∧ coeff.c < 0
def statement_3 (coeff : QuadraticCoefficients) : Prop := coeff.a > 0 ∧ coeff.b < 0 ∧ coeff.c < 0
def statement_4 (coeff : QuadraticCoefficients) : Prop := coeff.a < 0 ∧ coeff.b > 0 ∧ coeff.c > 0

/-- Main theorem: Exactly 2 out of 4 statements are incorrect for a quadratic equation with two positive roots -/
theorem two_incorrect_statements
  (coeff : QuadraticCoefficients)
  (roots : QuadraticRoots)
  (h_positive : both_roots_positive roots)
  (h_vieta : satisfies_vieta coeff roots) :
  (¬statement_1 coeff ∧ ¬statement_2 coeff ∧ statement_3 coeff ∧ statement_4 coeff) ∨
  (¬statement_1 coeff ∧ ¬statement_2 coeff ∧ statement_3 coeff ∧ ¬statement_4 coeff) ∨
  (¬statement_1 coeff ∧ ¬statement_2 coeff ∧ ¬statement_3 coeff ∧ statement_4 coeff) :=
by sorry

end NUMINAMATH_CALUDE_two_incorrect_statements_l2971_297165


namespace NUMINAMATH_CALUDE_job_completion_time_l2971_297114

/-- Given the work rates of machines A, B, and C, prove that 15 type A machines and 7 type B machines complete the job in 4 hours. -/
theorem job_completion_time 
  (A B C : ℝ) -- Work rates of machines A, B, and C in jobs per hour
  (h1 : 15 * A + 7 * B = 1 / 4) -- 15 type A and 7 type B machines complete the job in x hours
  (h2 : 8 * B + 15 * C = 1 / 11) -- 8 type B and 15 type C machines complete the job in 11 hours
  (h3 : A + B + C = 1 / 44) -- 1 of each machine type completes the job in 44 hours
  : 15 * A + 7 * B = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2971_297114


namespace NUMINAMATH_CALUDE_parabola_symmetry_l2971_297115

/-- A parabola with vertex form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is on a parabola -/
def IsOnParabola (p : Point) (para : Parabola) : Prop :=
  p.y = para.a * (p.x - para.h)^2 + para.k

theorem parabola_symmetry (para : Parabola) (m : ℝ) :
  let A : Point := { x := -1, y := 4 }
  let B : Point := { x := m, y := 4 }
  (IsOnParabola A para ∧ IsOnParabola B para) → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l2971_297115


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l2971_297136

theorem greatest_value_quadratic_inequality :
  ∀ a : ℝ, a^2 - 12*a + 35 ≤ 0 → a ≤ 7 ∧
  ∃ a : ℝ, a^2 - 12*a + 35 ≤ 0 ∧ a = 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l2971_297136


namespace NUMINAMATH_CALUDE_work_completion_theorem_l2971_297146

/-- Represents the number of days required to complete the work -/
def original_days : ℕ := 11

/-- Represents the number of additional men who joined -/
def additional_men : ℕ := 10

/-- Represents the number of days saved after additional men joined -/
def days_saved : ℕ := 3

/-- Calculates the original number of men required to complete the work -/
def original_men : ℕ := 27

theorem work_completion_theorem :
  ∃ (work_rate : ℚ),
    (original_men * work_rate * original_days : ℚ) =
    ((original_men + additional_men) * work_rate * (original_days - days_saved) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l2971_297146


namespace NUMINAMATH_CALUDE_problem_solution_l2971_297105

theorem problem_solution (a b : ℝ) 
  (h1 : Real.log a + b = -2)
  (h2 : a ^ b = 10) : 
  a = (1 : ℝ) / 10 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2971_297105


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2971_297164

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 55) :
  1 / x + 1 / y = 16 / 55 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2971_297164


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l2971_297192

theorem ratio_of_sum_and_difference (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 5 * (a - b)) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l2971_297192


namespace NUMINAMATH_CALUDE_quadruplet_solution_l2971_297132

theorem quadruplet_solution (a b c d : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_product : a * b * c * d = 1)
  (h_eq1 : a^2012 + 2012 * b = 2012 * c + d^2012)
  (h_eq2 : 2012 * a + b^2012 = c^2012 + 2012 * d) :
  ∃ t : ℝ, t > 0 ∧ a = t ∧ b = 1/t ∧ c = 1/t ∧ d = t :=
sorry

end NUMINAMATH_CALUDE_quadruplet_solution_l2971_297132


namespace NUMINAMATH_CALUDE_bus_average_speed_with_stoppages_l2971_297179

/-- Calculates the average speed of a bus including stoppages -/
theorem bus_average_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (total_time : ℝ) :
  speed_without_stoppages = 50 →
  stoppage_time = 12 →
  total_time = 60 →
  (speed_without_stoppages * (total_time - stoppage_time) / total_time) = 40 :=
by sorry

end NUMINAMATH_CALUDE_bus_average_speed_with_stoppages_l2971_297179


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2971_297137

/-- The standard equation of a hyperbola with given foci and asymptotes -/
theorem hyperbola_equation (c : ℝ) (m : ℝ) :
  c = Real.sqrt 10 →
  m = 1 / 2 →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1 ↔
      ((x + c)^2 + y^2)^(1/2) - ((x - c)^2 + y^2)^(1/2) = 2*a)) ∧
    (∀ (x : ℝ), y = m*x ∨ y = -m*x ↔ x^2 / a^2 - y^2 / b^2 = 0) ∧
    a^2 = 8 ∧ b^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2971_297137


namespace NUMINAMATH_CALUDE_students_per_classroom_l2971_297161

/-- Given a school trip scenario, calculate the number of students per classroom. -/
theorem students_per_classroom
  (num_classrooms : ℕ)
  (seats_per_bus : ℕ)
  (num_buses : ℕ)
  (h1 : num_classrooms = 67)
  (h2 : seats_per_bus = 6)
  (h3 : num_buses = 737) :
  (num_buses * seats_per_bus) / num_classrooms = 66 :=
by sorry

end NUMINAMATH_CALUDE_students_per_classroom_l2971_297161


namespace NUMINAMATH_CALUDE_fish_ratio_calculation_l2971_297117

/-- The ratio of tagged fish to total fish in a second catch -/
def fish_ratio (tagged_first : ℕ) (total_second : ℕ) (tagged_second : ℕ) (total_pond : ℕ) : ℚ :=
  tagged_second / total_second

/-- Theorem stating the ratio of tagged fish to total fish in the second catch -/
theorem fish_ratio_calculation :
  let tagged_first : ℕ := 40
  let total_second : ℕ := 50
  let tagged_second : ℕ := 2
  let total_pond : ℕ := 1000
  fish_ratio tagged_first total_second tagged_second total_pond = 1 / 25 := by
  sorry


end NUMINAMATH_CALUDE_fish_ratio_calculation_l2971_297117


namespace NUMINAMATH_CALUDE_triangle_property_l2971_297154

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.A = 2 * t.B) 
  (h2 : t.A = 3 * t.B) : 
  t.a^2 = t.b * (t.b + t.c) ∧ 
  t.c^2 = (1 / t.b) * (t.a - t.b) * (t.a^2 - t.b^2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_property_l2971_297154


namespace NUMINAMATH_CALUDE_sum_base6_100_l2971_297190

/-- Converts a number from base 10 to base 6 -/
def toBase6 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 6 to base 10 -/
def fromBase6 (n : ℕ) : ℕ := sorry

/-- Sum of numbers from 1 to n in base 6 -/
def sumBase6 (n : ℕ) : ℕ := sorry

theorem sum_base6_100 : sumBase6 100 = toBase6 (fromBase6 6110) := by sorry

end NUMINAMATH_CALUDE_sum_base6_100_l2971_297190


namespace NUMINAMATH_CALUDE_inequality_solution_l2971_297124

theorem inequality_solution (x : ℝ) :
  (∃ a : ℝ, a ∈ Set.Icc (-1) 2 ∧ (2 - a) * x^3 + (1 - 2*a) * x^2 - 6*x + 5 + 4*a - a^2 < 0) ↔
  (x < -2 ∨ (0 < x ∧ x < 1) ∨ x > 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2971_297124


namespace NUMINAMATH_CALUDE_min_value_sqrt_inequality_l2971_297186

theorem min_value_sqrt_inequality :
  ∃ (a : ℝ), (∀ (x y : ℝ), x > 0 ∧ y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ∧
  (∀ (b : ℝ), (∀ (x y : ℝ), x > 0 ∧ y > 0 → Real.sqrt x + Real.sqrt y ≤ b * Real.sqrt (x + y)) → a ≤ b) ∧
  a = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_inequality_l2971_297186


namespace NUMINAMATH_CALUDE_complement_of_44_36_l2971_297145

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- Calculates the complement of an angle -/
def complement (α : Angle) : Angle :=
  let total_minutes := 90 * 60 - (α.degrees * 60 + α.minutes)
  { degrees := total_minutes / 60, minutes := total_minutes % 60 }

theorem complement_of_44_36 :
  let α : Angle := { degrees := 44, minutes := 36 }
  complement α = { degrees := 45, minutes := 24 } := by
  sorry

end NUMINAMATH_CALUDE_complement_of_44_36_l2971_297145


namespace NUMINAMATH_CALUDE_least_integer_with_deletion_property_l2971_297112

theorem least_integer_with_deletion_property : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x = 950) ∧ 
  (∀ y : ℕ, y > 0 ∧ y < x → ¬(y / 10 = y / 19)) ∧
  (x / 10 = x / 19) := by
  sorry

end NUMINAMATH_CALUDE_least_integer_with_deletion_property_l2971_297112


namespace NUMINAMATH_CALUDE_eggs_per_basket_l2971_297116

theorem eggs_per_basket (blue_eggs : Nat) (yellow_eggs : Nat) (min_eggs : Nat)
  (h1 : blue_eggs = 30)
  (h2 : yellow_eggs = 42)
  (h3 : min_eggs = 6) :
  ∃ (x : Nat), x ≥ min_eggs ∧ x ∣ blue_eggs ∧ x ∣ yellow_eggs ∧
    ∀ (y : Nat), y > x → ¬(y ∣ blue_eggs ∧ y ∣ yellow_eggs) :=
by
  sorry

end NUMINAMATH_CALUDE_eggs_per_basket_l2971_297116


namespace NUMINAMATH_CALUDE_martin_initial_fruits_l2971_297133

/-- The number of fruits Martin initially had --/
def initial_fruits : ℕ := 288

/-- The number of oranges Martin has after eating half his fruits --/
def oranges_after : ℕ := 50

/-- The number of apples Martin has after eating half his fruits --/
def apples_after : ℕ := 72

/-- The number of limes Martin has after eating half his fruits --/
def limes_after : ℕ := 24

theorem martin_initial_fruits :
  (initial_fruits / 2 = oranges_after + apples_after + limes_after) ∧
  (oranges_after = 2 * limes_after) ∧
  (apples_after = 3 * limes_after) ∧
  (oranges_after = 50) ∧
  (apples_after = 72) :=
by sorry

end NUMINAMATH_CALUDE_martin_initial_fruits_l2971_297133


namespace NUMINAMATH_CALUDE_exists_valid_pairs_l2971_297157

def digits_ge_6 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≥ 6

def a (k : ℕ) : ℕ :=
  (10^k - 3) * 10^2 + 97

theorem exists_valid_pairs :
  ∃ n : ℕ, ∀ k ≥ n, 
    digits_ge_6 (a k) ∧ 
    digits_ge_6 7 ∧ 
    digits_ge_6 (a k * 7) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_pairs_l2971_297157


namespace NUMINAMATH_CALUDE_triangles_in_4x6_grid_l2971_297119

/-- Represents a grid with vertical and horizontal sections -/
structure Grid :=
  (vertical_sections : ℕ)
  (horizontal_sections : ℕ)

/-- Calculates the number of triangles in a grid with diagonal lines -/
def count_triangles (g : Grid) : ℕ :=
  let small_right_triangles := g.vertical_sections * g.horizontal_sections
  let medium_right_triangles := 2 * (g.vertical_sections - 1) * (g.horizontal_sections - 1)
  let large_isosceles_triangles := g.horizontal_sections - 1
  small_right_triangles + medium_right_triangles + large_isosceles_triangles

/-- Theorem: The number of triangles in a 4x6 grid is 90 -/
theorem triangles_in_4x6_grid :
  count_triangles { vertical_sections := 4, horizontal_sections := 6 } = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_4x6_grid_l2971_297119


namespace NUMINAMATH_CALUDE_last_number_proof_l2971_297159

theorem last_number_proof (a b c d : ℝ) : 
  (a + b + c) / 3 = 6 →
  (b + c + d) / 3 = 3 →
  a + d = 13 →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_last_number_proof_l2971_297159


namespace NUMINAMATH_CALUDE_soda_price_proof_l2971_297177

/-- Given a regular price per can of soda, prove that it equals $0.55 under the given conditions -/
theorem soda_price_proof (P : ℝ) : 
  (∃ (discounted_price : ℝ), 
    discounted_price = 0.75 * P ∧ 
    70 * discounted_price = 28.875) → 
  P = 0.55 := by
sorry

end NUMINAMATH_CALUDE_soda_price_proof_l2971_297177


namespace NUMINAMATH_CALUDE_volunteer_selection_theorem_l2971_297108

/-- The number of volunteers --/
def n : ℕ := 20

/-- The number of volunteers to be selected --/
def k : ℕ := 4

/-- The number of the first specific volunteer that must be selected --/
def a : ℕ := 5

/-- The number of the second specific volunteer that must be selected --/
def b : ℕ := 14

/-- The number of volunteers with numbers less than the first specific volunteer --/
def m : ℕ := a - 1

/-- The number of volunteers with numbers greater than the second specific volunteer --/
def p : ℕ := n - b

/-- The total number of ways to select the volunteers under the given conditions --/
def total_ways : ℕ := Nat.choose m 2 + Nat.choose p 2

theorem volunteer_selection_theorem :
  total_ways = 21 := by sorry

end NUMINAMATH_CALUDE_volunteer_selection_theorem_l2971_297108


namespace NUMINAMATH_CALUDE_weight_swap_l2971_297138

structure Weight :=
  (value : ℝ)

def WeighingScale (W X Y Z : Weight) : Prop :=
  (Z.value > Y.value) ∧
  (X.value > W.value) ∧
  (Y.value + Z.value > W.value + X.value) ∧
  (Z.value > W.value)

theorem weight_swap (W X Y Z : Weight) 
  (h : WeighingScale W X Y Z) : 
  (W.value + X.value > Y.value + Z.value) → 
  (Z.value + X.value > Y.value + W.value) :=
sorry

end NUMINAMATH_CALUDE_weight_swap_l2971_297138


namespace NUMINAMATH_CALUDE_james_brother_age_l2971_297181

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- The current year -/
def currentYear : ℕ := 2023

/-- Calculates a person's age in a given year -/
def ageInYear (p : Person) (year : ℕ) : ℕ :=
  if year ≥ currentYear then p.age + (year - currentYear)
  else p.age - (currentYear - year)

theorem james_brother_age :
  let john : Person := ⟨39⟩
  let james : Person := ⟨(ageInYear john (currentYear - 3) / 2) - 6⟩
  let james_brother : Person := ⟨james.age + 4⟩
  james_brother.age = 16 := by sorry

end NUMINAMATH_CALUDE_james_brother_age_l2971_297181


namespace NUMINAMATH_CALUDE_cubic_sum_from_sixth_power_l2971_297130

theorem cubic_sum_from_sixth_power (x : ℝ) (h : 34 = x^6 + 1/x^6) :
  x^3 + 1/x^3 = 6 ∨ x^3 + 1/x^3 = -6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_from_sixth_power_l2971_297130


namespace NUMINAMATH_CALUDE_distance_between_trees_l2971_297196

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : 
  yard_length = 414 → num_trees = 24 → 
  (yard_length : ℚ) / (num_trees - 1 : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l2971_297196


namespace NUMINAMATH_CALUDE_mold_cost_is_250_l2971_297134

/-- The cost of a mold for handmade shoes --/
def mold_cost (hourly_rate : ℝ) (hours : ℝ) (work_percentage : ℝ) (total_paid : ℝ) : ℝ :=
  total_paid - work_percentage * hourly_rate * hours

/-- Proves that the cost of the mold is $250 given the problem conditions --/
theorem mold_cost_is_250 :
  mold_cost 75 8 0.8 730 = 250 := by
  sorry

end NUMINAMATH_CALUDE_mold_cost_is_250_l2971_297134


namespace NUMINAMATH_CALUDE_arithmetic_operations_l2971_297123

theorem arithmetic_operations :
  (-10 + 2 = -8) ∧
  (-6 - 3 = -9) ∧
  ((-4) * 6 = -24) ∧
  ((-15) / 5 = -3) ∧
  ((-4)^2 / 2 = 8) ∧
  (|(-2)| - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l2971_297123


namespace NUMINAMATH_CALUDE_inequality_solution_l2971_297189

-- Define the inequality
def inequality (x : ℝ) : Prop :=
  9.280 * (Real.log x / Real.log 7) - Real.log 7 * (Real.log x / Real.log 3) > Real.log 0.25 / Real.log 2

-- Define the solution interval
def solution_interval : Set ℝ :=
  {x | x > 0 ∧ x < Real.exp ((2 * Real.log 3) / (Real.log 7 / Real.log 3 - Real.log 3 / Real.log 7))}

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, inequality x ↔ x ∈ solution_interval :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2971_297189


namespace NUMINAMATH_CALUDE_tie_in_may_l2971_297141

structure Player where
  january : ℕ
  february : ℕ
  march : ℕ
  april : ℕ
  may : ℕ

def johnson : Player := ⟨2, 12, 20, 15, 9⟩
def martinez : Player := ⟨5, 9, 15, 20, 9⟩

def cumulative_score (p : Player) (month : ℕ) : ℕ :=
  match month with
  | 1 => p.january
  | 2 => p.january + p.february
  | 3 => p.january + p.february + p.march
  | 4 => p.january + p.february + p.march + p.april
  | 5 => p.january + p.february + p.march + p.april + p.may
  | _ => 0

def first_tie_month : ℕ :=
  [1, 2, 3, 4, 5].find? (λ m => cumulative_score johnson m = cumulative_score martinez m)
    |>.getD 0

theorem tie_in_may :
  first_tie_month = 5 := by sorry

end NUMINAMATH_CALUDE_tie_in_may_l2971_297141


namespace NUMINAMATH_CALUDE_problem_statement_l2971_297156

theorem problem_statement (x y : ℝ) 
  (h1 : (x - y) / (x + y) = 9)
  (h2 : x * y / (x + y) = -60) :
  (x + y) + (x - y) + x * y = -150 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2971_297156


namespace NUMINAMATH_CALUDE_squares_below_line_l2971_297150

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of integer points strictly below a line in the first quadrant -/
def countPointsBelowLine (l : Line) : ℕ :=
  sorry

/-- The specific line from the problem -/
def problemLine : Line :=
  { a := 12, b := 180, c := 2160 }

/-- The theorem statement -/
theorem squares_below_line :
  countPointsBelowLine problemLine = 1969 := by
  sorry

end NUMINAMATH_CALUDE_squares_below_line_l2971_297150


namespace NUMINAMATH_CALUDE_greatest_partition_number_l2971_297187

/-- A partition of the positive integers into k subsets -/
def Partition (k : ℕ) := Fin k → Set ℕ

/-- The property that a partition satisfies the sum condition for all n ≥ 15 -/
def SatisfiesSumCondition (p : Partition k) : Prop :=
  ∀ (n : ℕ) (i : Fin k), n ≥ 15 →
    ∃ (x y : ℕ), x ∈ p i ∧ y ∈ p i ∧ x ≠ y ∧ x + y = n

/-- The main theorem stating that 3 is the greatest k satisfying the conditions -/
theorem greatest_partition_number :
  (∃ (p : Partition 3), SatisfiesSumCondition p) ∧
  (∀ k > 3, ¬∃ (p : Partition k), SatisfiesSumCondition p) :=
sorry

end NUMINAMATH_CALUDE_greatest_partition_number_l2971_297187


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2971_297169

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2971_297169


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l2971_297121

theorem product_mod_seventeen : (1520 * 1521 * 1522) % 17 = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l2971_297121


namespace NUMINAMATH_CALUDE_sales_growth_rate_is_ten_percent_l2971_297167

/-- The average monthly growth rate of total sales from February to April -/
def average_monthly_growth_rate : ℝ := 0.1

/-- Total sales in February (in yuan) -/
def february_sales : ℝ := 240000

/-- Total sales in April (in yuan) -/
def april_sales : ℝ := 290400

/-- Number of months between February and April -/
def months_between : ℕ := 2

theorem sales_growth_rate_is_ten_percent :
  april_sales = february_sales * (1 + average_monthly_growth_rate) ^ months_between := by
  sorry

end NUMINAMATH_CALUDE_sales_growth_rate_is_ten_percent_l2971_297167


namespace NUMINAMATH_CALUDE_direct_proportion_through_point_decreasing_l2971_297185

-- Define the direct proportion function
def direct_proportion (m : ℝ) (x : ℝ) : ℝ := m * x

-- Define the theorem
theorem direct_proportion_through_point_decreasing (m : ℝ) :
  (direct_proportion m m = 4) →
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → direct_proportion m x₁ > direct_proportion m x₂) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_through_point_decreasing_l2971_297185


namespace NUMINAMATH_CALUDE_exists_line_with_three_colors_l2971_297143

/-- A color type with four possible values -/
inductive Color
  | One
  | Two
  | Three
  | Four

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A coloring function that assigns a color to each point in the plane -/
def Coloring := Point → Color

/-- A predicate that checks if a coloring uses all four colors -/
def uses_all_colors (f : Coloring) : Prop :=
  (∃ p : Point, f p = Color.One) ∧
  (∃ p : Point, f p = Color.Two) ∧
  (∃ p : Point, f p = Color.Three) ∧
  (∃ p : Point, f p = Color.Four)

/-- A predicate that checks if a point is on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem -/
theorem exists_line_with_three_colors (f : Coloring) (h : uses_all_colors f) :
  ∃ l : Line, ∃ p₁ p₂ p₃ : Point,
    on_line p₁ l ∧ on_line p₂ l ∧ on_line p₃ l ∧
    f p₁ ≠ f p₂ ∧ f p₁ ≠ f p₃ ∧ f p₂ ≠ f p₃ :=
sorry

end NUMINAMATH_CALUDE_exists_line_with_three_colors_l2971_297143


namespace NUMINAMATH_CALUDE_position_after_four_steps_l2971_297158

/-- Given a number line where the distance from 0 to 30 is divided into 6 equal steps,
    the position reached after 4 steps is 20. -/
theorem position_after_four_steps :
  ∀ (total_distance : ℝ) (total_steps : ℕ) (steps_taken : ℕ),
    total_distance = 30 →
    total_steps = 6 →
    steps_taken = 4 →
    (total_distance / total_steps) * steps_taken = 20 :=
by
  sorry

#check position_after_four_steps

end NUMINAMATH_CALUDE_position_after_four_steps_l2971_297158


namespace NUMINAMATH_CALUDE_parabola_b_value_l2971_297135

/-- A parabola passing through two points -/
structure Parabola where
  a : ℝ
  b : ℝ
  passes_through_1_2 : 2 = 1^2 + a * 1 + b
  passes_through_3_2 : 2 = 3^2 + a * 3 + b

/-- The value of b for the parabola passing through (1,2) and (3,2) is 5 -/
theorem parabola_b_value (p : Parabola) : p.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_b_value_l2971_297135


namespace NUMINAMATH_CALUDE_expression_factorization_l2971_297152

theorem expression_factorization (a b c d : ℝ) : 
  a^3 * (b^2 - d^2) + b^3 * (c^2 - a^2) + c^3 * (d^2 - b^2) + d^3 * (a^2 - c^2) = 
  (a - b) * (b - c) * (c - d) * (d - a) * 
  (a^2 + a*b + a*c + a*d + b^2 + b*c + b*d + c^2 + c*d + d^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2971_297152


namespace NUMINAMATH_CALUDE_palindrome_product_sum_theorem_l2971_297195

/-- A positive three-digit palindrome is a number between 100 and 999 inclusive,
    where the first and third digits are the same. -/
def IsPositiveThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- There exist two positive three-digit palindromes whose product is 589185 and whose sum is 1534. -/
theorem palindrome_product_sum_theorem :
  ∃ (a b : ℕ), IsPositiveThreeDigitPalindrome a ∧
                IsPositiveThreeDigitPalindrome b ∧
                a * b = 589185 ∧
                a + b = 1534 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_product_sum_theorem_l2971_297195


namespace NUMINAMATH_CALUDE_car_wash_group_composition_l2971_297178

theorem car_wash_group_composition (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls : ℚ) / total = 2 / 5 →
  ((initial_girls : ℚ) - 2) / total = 3 / 10 →
  initial_girls = 8 := by
sorry

end NUMINAMATH_CALUDE_car_wash_group_composition_l2971_297178


namespace NUMINAMATH_CALUDE_digit_arrangement_count_l2971_297106

theorem digit_arrangement_count : 
  let digits : List ℕ := [4, 7, 5, 2, 0]
  let n : ℕ := digits.length
  let non_zero_digits : List ℕ := digits.filter (· ≠ 0)
  96 = (n - 1) * Nat.factorial (non_zero_digits.length) := by
  sorry

end NUMINAMATH_CALUDE_digit_arrangement_count_l2971_297106


namespace NUMINAMATH_CALUDE_fraction_simplification_l2971_297175

theorem fraction_simplification (d : ℝ) : 
  (5 + 4*d) / 9 - 3 + 1/3 = (4*d - 19) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2971_297175


namespace NUMINAMATH_CALUDE_distribute_five_into_three_l2971_297184

/-- The number of ways to distribute n distinct objects into k distinct non-empty groups --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 150 ways to distribute 5 distinct objects into 3 distinct non-empty groups --/
theorem distribute_five_into_three : distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_into_three_l2971_297184


namespace NUMINAMATH_CALUDE_survey_result_l2971_297166

theorem survey_result (total : ℕ) (tv_dislike_percent : ℚ) (both_dislike_percent : ℚ) 
  (h1 : total = 1500)
  (h2 : tv_dislike_percent = 25 / 100)
  (h3 : both_dislike_percent = 20 / 100) :
  ↑⌊both_dislike_percent * (tv_dislike_percent * total)⌋ = 75 := by
  sorry

end NUMINAMATH_CALUDE_survey_result_l2971_297166


namespace NUMINAMATH_CALUDE_complex_sum_powers_l2971_297109

theorem complex_sum_powers (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^101 + z^102 + z^103 + z^104 + z^105 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l2971_297109


namespace NUMINAMATH_CALUDE_utility_expense_increase_l2971_297182

theorem utility_expense_increase
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0) :
  let june_total := a + b + c
  let july_increase := 0.1 * a + 0.3 * b + 0.2 * c
  july_increase / june_total = (0.1 * a + 0.3 * b + 0.2 * c) / (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_utility_expense_increase_l2971_297182


namespace NUMINAMATH_CALUDE_total_books_collected_l2971_297173

def north_america : ℕ := 581
def south_america : ℕ := 435
def africa : ℕ := 524
def europe : ℕ := 688
def australia : ℕ := 319
def asia : ℕ := 526
def antarctica : ℕ := 276

theorem total_books_collected :
  north_america + south_america + africa + europe + australia + asia + antarctica = 3349 := by
  sorry

end NUMINAMATH_CALUDE_total_books_collected_l2971_297173


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l2971_297180

/-- The perimeter of a semicircle with radius 11 is approximately 56.56 -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 11
  let π_approx : ℝ := 3.14159
  let semicircle_perimeter := π_approx * r + 2 * r
  ∃ ε > 0, abs (semicircle_perimeter - 56.56) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l2971_297180


namespace NUMINAMATH_CALUDE_dollar_neg_three_four_l2971_297155

-- Define the $ operation
def dollar (x y : ℤ) : ℤ := x * (y + 2) + x * y

-- Theorem statement
theorem dollar_neg_three_four : dollar (-3) 4 = -30 := by
  sorry

end NUMINAMATH_CALUDE_dollar_neg_three_four_l2971_297155


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2971_297103

theorem quadratic_roots_relation (u v : ℝ) (m n : ℝ) : 
  (3 * u^2 + 4 * u + 5 = 0) →
  (3 * v^2 + 4 * v + 5 = 0) →
  ((u^2 + 1)^2 + m * (u^2 + 1) + n = 0) →
  ((v^2 + 1)^2 + m * (v^2 + 1) + n = 0) →
  m = -4/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2971_297103


namespace NUMINAMATH_CALUDE_final_score_eq_initial_minus_one_l2971_297113

/-- A scoring system where 1 point is deducted for a missing answer -/
structure ScoringSystem where
  initial_score : ℝ
  missing_answer_deduction : ℝ := 1

/-- The final score after deducting for a missing answer -/
def final_score (s : ScoringSystem) : ℝ :=
  s.initial_score - s.missing_answer_deduction

/-- Theorem stating that the final score is equal to the initial score minus 1 -/
theorem final_score_eq_initial_minus_one (s : ScoringSystem) :
  final_score s = s.initial_score - 1 := by
  sorry

#check final_score_eq_initial_minus_one

end NUMINAMATH_CALUDE_final_score_eq_initial_minus_one_l2971_297113


namespace NUMINAMATH_CALUDE_all_statements_imply_negation_l2971_297128

theorem all_statements_imply_negation (p q r : Prop) : 
  -- Statement 1
  ((p ∧ q ∧ r) → (¬p ∨ ¬q ∨ r)) ∧
  -- Statement 2
  ((p ∧ ¬q ∧ r) → (¬p ∨ ¬q ∨ r)) ∧
  -- Statement 3
  ((¬p ∧ q ∧ ¬r) → (¬p ∨ ¬q ∨ r)) ∧
  -- Statement 4
  ((¬p ∧ q ∧ ¬r) → (¬p ∨ ¬q ∨ r)) := by
  sorry

#check all_statements_imply_negation

end NUMINAMATH_CALUDE_all_statements_imply_negation_l2971_297128


namespace NUMINAMATH_CALUDE_line_slope_and_intercept_l2971_297199

/-- Given a line with equation 3x + 4y + 5 = 0, prove its slope is -3/4 and y-intercept is -5/4 -/
theorem line_slope_and_intercept :
  let line := {(x, y) : ℝ × ℝ | 3 * x + 4 * y + 5 = 0}
  ∃ m b : ℝ, m = -3/4 ∧ b = -5/4 ∧ ∀ x y : ℝ, (x, y) ∈ line ↔ y = m * x + b :=
sorry

end NUMINAMATH_CALUDE_line_slope_and_intercept_l2971_297199


namespace NUMINAMATH_CALUDE_court_cases_dismissed_l2971_297170

theorem court_cases_dismissed (total_cases : ℕ) 
  (remaining_cases : ℕ) (innocent_cases : ℕ) (delayed_cases : ℕ) (guilty_cases : ℕ) :
  total_cases = 17 →
  remaining_cases = innocent_cases + delayed_cases + guilty_cases →
  innocent_cases = 2 * (remaining_cases / 3) →
  delayed_cases = 1 →
  guilty_cases = 4 →
  total_cases - remaining_cases = 2 := by
sorry

end NUMINAMATH_CALUDE_court_cases_dismissed_l2971_297170


namespace NUMINAMATH_CALUDE_inequality_solution_and_function_property_l2971_297102

def f (x : ℝ) := |x - 2|

theorem inequality_solution_and_function_property :
  (∀ x : ℝ, (|x - 2| + |x| ≤ 4) ↔ (x ∈ Set.Icc (-1) 3)) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a ≠ b → a * f b + b * f a ≥ 2 * |a - b|) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_and_function_property_l2971_297102


namespace NUMINAMATH_CALUDE_unreasonable_milk_volume_l2971_297111

/-- Represents the volume of milk in liters --/
def milk_volume : ℝ := 250

/-- Represents a reasonable maximum volume of milk a person can drink in a day (in liters) --/
def max_reasonable_volume : ℝ := 10

/-- Theorem stating that the given milk volume is unreasonable for a person to drink in a day --/
theorem unreasonable_milk_volume : milk_volume > max_reasonable_volume := by
  sorry

end NUMINAMATH_CALUDE_unreasonable_milk_volume_l2971_297111


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2971_297131

def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2*x > 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2971_297131


namespace NUMINAMATH_CALUDE_line_perpendicular_to_parallel_plane_l2971_297142

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_plane
  (m : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : parallel α β) :
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_parallel_plane_l2971_297142


namespace NUMINAMATH_CALUDE_ice_cream_frozen_yoghurt_cost_difference_l2971_297162

/-- Calculates the difference in cost between ice cream and frozen yoghurt purchases -/
theorem ice_cream_frozen_yoghurt_cost_difference :
  let ice_cream_cartons : ℕ := 10
  let frozen_yoghurt_cartons : ℕ := 4
  let ice_cream_price : ℕ := 4
  let frozen_yoghurt_price : ℕ := 1
  let ice_cream_total_cost := ice_cream_cartons * ice_cream_price
  let frozen_yoghurt_total_cost := frozen_yoghurt_cartons * frozen_yoghurt_price
  ice_cream_total_cost - frozen_yoghurt_total_cost = 36 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_frozen_yoghurt_cost_difference_l2971_297162


namespace NUMINAMATH_CALUDE_ratio_p_to_r_l2971_297163

theorem ratio_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4)
  (h2 : r / s = 3 / 2)
  (h3 : s / q = 1 / 5) :
  p / r = 25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_p_to_r_l2971_297163


namespace NUMINAMATH_CALUDE_sqrt_2_minus_x_real_range_l2971_297144

theorem sqrt_2_minus_x_real_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2 - x) ↔ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_2_minus_x_real_range_l2971_297144


namespace NUMINAMATH_CALUDE_theo_daily_consumption_l2971_297188

/-- Represents the daily water consumption of the three siblings -/
structure SiblingWaterConsumption where
  theo : ℕ
  mason : ℕ
  roxy : ℕ

/-- The total water consumption of the siblings over a week -/
def weeklyConsumption (swc : SiblingWaterConsumption) : ℕ :=
  7 * (swc.theo + swc.mason + swc.roxy)

/-- Theorem stating Theo's daily water consumption -/
theorem theo_daily_consumption :
  ∃ (swc : SiblingWaterConsumption),
    swc.mason = 7 ∧
    swc.roxy = 9 ∧
    weeklyConsumption swc = 168 ∧
    swc.theo = 8 := by
  sorry

end NUMINAMATH_CALUDE_theo_daily_consumption_l2971_297188


namespace NUMINAMATH_CALUDE_quadrilateral_count_l2971_297107

/-- The number of points on the circumference of the circle -/
def n : ℕ := 15

/-- The number of vertices required from the circumference -/
def k : ℕ := 3

/-- The number of different convex quadrilaterals that can be formed -/
def num_quadrilaterals : ℕ := Nat.choose n k

theorem quadrilateral_count :
  num_quadrilaterals = 455 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_count_l2971_297107


namespace NUMINAMATH_CALUDE_addition_puzzle_solution_l2971_297120

/-- A digit is a natural number from 0 to 9 -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- Function to convert a four-digit number to its decimal representation -/
def toDecimal (a b c d : Digit) : ℕ := 1000 * a.val + 100 * b.val + 10 * c.val + d.val

/-- Predicate to check if three digits are distinct -/
def areDistinct (a b c : Digit) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem addition_puzzle_solution :
  ∃ (possibleD : Finset Digit),
    (∀ a b c d : Digit,
      areDistinct a b c →
      toDecimal a b a c + toDecimal c a b a = toDecimal d c d d →
      d ∈ possibleD) ∧
    possibleD.card = 7 := by sorry

end NUMINAMATH_CALUDE_addition_puzzle_solution_l2971_297120


namespace NUMINAMATH_CALUDE_fourth_term_of_progression_l2971_297198

-- Define the geometric progression
def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

-- Define our specific progression
def our_progression (n : ℕ) : ℝ := 5^(1 / (5 * 2^(n - 1)))

-- Theorem statement
theorem fourth_term_of_progression :
  our_progression 4 = 5^(1/10) := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_progression_l2971_297198


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l2971_297197

theorem chess_game_draw_probability 
  (p_a_wins : ℝ) 
  (p_a_not_lose : ℝ) 
  (h1 : p_a_wins = 0.4) 
  (h2 : p_a_not_lose = 0.9) : 
  p_a_not_lose - p_a_wins = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l2971_297197


namespace NUMINAMATH_CALUDE_tornado_distance_ratio_l2971_297127

/-- Given the distances traveled by various objects in a tornado, prove the ratio of
    the lawn chair's distance to the car's distance. -/
theorem tornado_distance_ratio :
  ∀ (car_distance lawn_chair_distance birdhouse_distance : ℝ),
  car_distance = 200 →
  birdhouse_distance = 1200 →
  birdhouse_distance = 3 * lawn_chair_distance →
  lawn_chair_distance / car_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_tornado_distance_ratio_l2971_297127


namespace NUMINAMATH_CALUDE_range_of_x0_l2971_297151

/-- The circle C: x^2 + y^2 = 3 -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 3

/-- The line l: x + 3y - 6 = 0 -/
def Line (x y : ℝ) : Prop := x + 3*y - 6 = 0

/-- The angle between two vectors is 60 degrees -/
def AngleSixtyDegrees (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1*x2 + y1*y2) / (Real.sqrt (x1^2 + y1^2) * Real.sqrt (x2^2 + y2^2)) = 1/2

theorem range_of_x0 (x0 y0 : ℝ) :
  Line x0 y0 →
  (∃ x y, Circle x y ∧ AngleSixtyDegrees x0 y0 x y) →
  0 ≤ x0 ∧ x0 ≤ 6/5 := by sorry

end NUMINAMATH_CALUDE_range_of_x0_l2971_297151


namespace NUMINAMATH_CALUDE_point_A_x_range_l2971_297148

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 6 = 0

-- Define a point on the circle
def point_on_circle (x y : ℝ) : Prop := circle_M x y

-- Define a point on the line
def point_on_line (x y : ℝ) : Prop := line_l x y

-- Define the angle between three points
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem point_A_x_range :
  ∀ (A B C : ℝ × ℝ),
    point_on_line A.1 A.2 →
    point_on_circle B.1 B.2 →
    point_on_circle C.1 C.2 →
    angle A B C = 60 →
    1 ≤ A.1 ∧ A.1 ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_point_A_x_range_l2971_297148


namespace NUMINAMATH_CALUDE_johns_book_expense_l2971_297140

def earnings : ℕ := 10 * 26

theorem johns_book_expense (money_left : ℕ) (book_expense : ℕ) : 
  money_left = 160 → 
  earnings = money_left + 2 * book_expense → 
  book_expense = 50 :=
by sorry

end NUMINAMATH_CALUDE_johns_book_expense_l2971_297140


namespace NUMINAMATH_CALUDE_jimmy_pens_purchase_l2971_297168

theorem jimmy_pens_purchase (pen_cost : ℕ) (notebook_cost : ℕ) (folder_cost : ℕ)
  (notebooks_bought : ℕ) (folders_bought : ℕ) (paid : ℕ) (change : ℕ) :
  pen_cost = 1 →
  notebook_cost = 3 →
  folder_cost = 5 →
  notebooks_bought = 4 →
  folders_bought = 2 →
  paid = 50 →
  change = 25 →
  (paid - change - (notebooks_bought * notebook_cost + folders_bought * folder_cost)) / pen_cost = 3 :=
by sorry

end NUMINAMATH_CALUDE_jimmy_pens_purchase_l2971_297168


namespace NUMINAMATH_CALUDE_max_numbers_summing_to_1000_with_distinct_digit_sums_l2971_297153

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a list of natural numbers has pairwise distinct sums of digits -/
def hasPairwiseDistinctDigitSums (list : List ℕ) : Prop := sorry

/-- The maximum number of natural numbers summing to 1000 with pairwise distinct digit sums -/
theorem max_numbers_summing_to_1000_with_distinct_digit_sums :
  ∃ (list : List ℕ),
    list.sum = 1000 ∧
    hasPairwiseDistinctDigitSums list ∧
    list.length = 19 ∧
    ∀ (other_list : List ℕ),
      other_list.sum = 1000 →
      hasPairwiseDistinctDigitSums other_list →
      other_list.length ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_max_numbers_summing_to_1000_with_distinct_digit_sums_l2971_297153


namespace NUMINAMATH_CALUDE_line_circle_intersection_point_on_line_point_inside_circle_l2971_297160

/-- The line y = kx + 1 intersects the circle x^2 + y^2 = 2 but doesn't pass through its center -/
theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), y = k * x + 1 ∧ x^2 + y^2 = 2 ∧ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

/-- The point (0, 1) is always on the line y = kx + 1 -/
theorem point_on_line (k : ℝ) : k * 0 + 1 = 1 := by
  sorry

/-- The point (0, 1) is inside the circle x^2 + y^2 = 2 -/
theorem point_inside_circle : 0^2 + 1^2 < 2 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_point_on_line_point_inside_circle_l2971_297160


namespace NUMINAMATH_CALUDE_sixth_term_of_sequence_l2971_297110

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- State the theorem
theorem sixth_term_of_sequence (a d : ℝ) :
  arithmetic_sequence a d 2 = 14 ∧
  arithmetic_sequence a d 4 = 32 →
  arithmetic_sequence a d 6 = 50 := by
  sorry


end NUMINAMATH_CALUDE_sixth_term_of_sequence_l2971_297110


namespace NUMINAMATH_CALUDE_partnership_profit_l2971_297193

/-- Represents the profit distribution in a partnership --/
structure Partnership where
  a_investment : ℕ  -- A's investment
  b_investment : ℕ  -- B's investment
  a_period : ℕ      -- A's investment period
  b_period : ℕ      -- B's investment period
  b_profit : ℕ      -- B's profit

/-- Calculates the total profit of the partnership --/
def total_profit (p : Partnership) : ℕ :=
  let ratio := (p.a_investment * p.a_period) / (p.b_investment * p.b_period)
  p.b_profit * (ratio + 1)

/-- Theorem stating the total profit for the given partnership conditions --/
theorem partnership_profit (p : Partnership) 
  (h1 : p.a_investment = 3 * p.b_investment) 
  (h2 : p.a_period = 2 * p.b_period)
  (h3 : p.b_profit = 3000) : 
  total_profit p = 21000 := by
  sorry

#eval total_profit { a_investment := 3, b_investment := 1, a_period := 2, b_period := 1, b_profit := 3000 }

end NUMINAMATH_CALUDE_partnership_profit_l2971_297193


namespace NUMINAMATH_CALUDE_black_balls_count_l2971_297176

theorem black_balls_count (white_balls : ℕ) (prob_white : ℚ) : 
  white_balls = 5 →
  prob_white = 5 / 11 →
  ∃ (total_balls : ℕ), 
    (prob_white = white_balls / total_balls) ∧
    (total_balls - white_balls = 6) :=
by sorry

end NUMINAMATH_CALUDE_black_balls_count_l2971_297176


namespace NUMINAMATH_CALUDE_game_probability_l2971_297147

theorem game_probability : 
  let total_rounds : ℕ := 6
  let alex_wins : ℕ := 3
  let mel_wins : ℕ := 2
  let chelsea_wins : ℕ := 1
  let p_alex : ℚ := 1/2
  let p_mel : ℚ := 1/3
  let p_chelsea : ℚ := 1/6
  (p_alex + p_mel + p_chelsea = 1) →
  (p_mel = 2 * p_chelsea) →
  (Nat.choose total_rounds alex_wins * 
   Nat.choose (total_rounds - alex_wins) mel_wins * 
   p_alex ^ alex_wins * p_mel ^ mel_wins * p_chelsea ^ chelsea_wins : ℚ) = 5/36 :=
by sorry

end NUMINAMATH_CALUDE_game_probability_l2971_297147


namespace NUMINAMATH_CALUDE_parallelogram_height_l2971_297122

theorem parallelogram_height (area base height : ℝ) : 
  area = 364 ∧ base = 26 ∧ area = base * height → height = 14 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2971_297122


namespace NUMINAMATH_CALUDE_earnings_exceed_goal_l2971_297174

/-- Represents the berry-picking job scenario --/
structure BerryPicking where
  lingonberry_rate : ℝ
  cloudberry_rate : ℝ
  blueberry_rate : ℝ
  monday_lingonberry : ℝ
  monday_cloudberry : ℝ
  monday_blueberry : ℝ
  tuesday_lingonberry_factor : ℝ
  tuesday_cloudberry_factor : ℝ
  tuesday_blueberry : ℝ
  goal : ℝ

/-- Calculates the total earnings for Monday and Tuesday --/
def total_earnings (job : BerryPicking) : ℝ :=
  let monday_earnings := 
    job.lingonberry_rate * job.monday_lingonberry +
    job.cloudberry_rate * job.monday_cloudberry +
    job.blueberry_rate * job.monday_blueberry
  let tuesday_earnings := 
    job.lingonberry_rate * (job.tuesday_lingonberry_factor * job.monday_lingonberry) +
    job.cloudberry_rate * (job.tuesday_cloudberry_factor * job.monday_cloudberry) +
    job.blueberry_rate * job.tuesday_blueberry
  monday_earnings + tuesday_earnings

/-- Theorem: Steve's earnings exceed his goal after two days --/
theorem earnings_exceed_goal (job : BerryPicking) 
  (h1 : job.lingonberry_rate = 2)
  (h2 : job.cloudberry_rate = 3)
  (h3 : job.blueberry_rate = 5)
  (h4 : job.monday_lingonberry = 8)
  (h5 : job.monday_cloudberry = 10)
  (h6 : job.monday_blueberry = 0)
  (h7 : job.tuesday_lingonberry_factor = 3)
  (h8 : job.tuesday_cloudberry_factor = 2)
  (h9 : job.tuesday_blueberry = 5)
  (h10 : job.goal = 150) :
  total_earnings job > job.goal := by
  sorry

end NUMINAMATH_CALUDE_earnings_exceed_goal_l2971_297174


namespace NUMINAMATH_CALUDE_competition_scores_l2971_297126

theorem competition_scores (x y z w : ℝ) 
  (hA : x = (y + z + w) / 3 + 2)
  (hB : y = (x + z + w) / 3 - 3)
  (hC : z = (x + y + w) / 3 + 3) :
  (x + y + z) / 3 - w = 2 := by sorry

end NUMINAMATH_CALUDE_competition_scores_l2971_297126


namespace NUMINAMATH_CALUDE_expression_evaluation_l2971_297171

theorem expression_evaluation : (10 + 1/3) + (-11.5) + (-10 - 1/3) - 4.5 = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2971_297171


namespace NUMINAMATH_CALUDE_graphing_calculator_theorem_l2971_297149

/-- Represents the number of students who brought graphing calculators -/
def graphing_calculator_count : ℕ := 10

/-- Represents the total number of boys in the class -/
def total_boys : ℕ := 20

/-- Represents the total number of girls in the class -/
def total_girls : ℕ := 18

/-- Represents the number of students who brought scientific calculators -/
def scientific_calculator_count : ℕ := 30

/-- Represents the number of girls who brought scientific calculators -/
def girls_with_scientific_calculators : ℕ := 15

theorem graphing_calculator_theorem :
  graphing_calculator_count = 10 ∧
  total_boys + total_girls = scientific_calculator_count + graphing_calculator_count :=
by sorry

end NUMINAMATH_CALUDE_graphing_calculator_theorem_l2971_297149


namespace NUMINAMATH_CALUDE_function_bounds_l2971_297172

theorem function_bounds (x : ℝ) : 
  (1 : ℝ) / 2 ≤ (x^2 + x + 1) / (x^2 + 1) ∧ (x^2 + x + 1) / (x^2 + 1) ≤ (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_bounds_l2971_297172


namespace NUMINAMATH_CALUDE_ac_over_b_value_l2971_297183

noncomputable def f (x : ℝ) : ℝ := |2 - Real.log x / Real.log 3|

theorem ac_over_b_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c) (h_f : f a = 2 * f b ∧ f a = 2 * f c) :
  a * c / b = 243 := by
  sorry

end NUMINAMATH_CALUDE_ac_over_b_value_l2971_297183
