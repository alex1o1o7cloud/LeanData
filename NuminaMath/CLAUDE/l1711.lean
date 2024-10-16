import Mathlib

namespace NUMINAMATH_CALUDE_base4_calculation_l1711_171132

/-- Represents a number in base 4 --/
def Base4 : Type := ℕ

/-- Multiplication operation for base 4 numbers --/
def mul_base4 : Base4 → Base4 → Base4 := sorry

/-- Division operation for base 4 numbers --/
def div_base4 : Base4 → Base4 → Base4 := sorry

/-- Conversion from decimal to base 4 --/
def to_base4 (n : ℕ) : Base4 := sorry

/-- Conversion from base 4 to decimal --/
def from_base4 (n : Base4) : ℕ := sorry

theorem base4_calculation :
  let a := to_base4 203
  let b := to_base4 21
  let c := to_base4 3
  let result := to_base4 110320
  mul_base4 (div_base4 a c) b = result := by sorry

end NUMINAMATH_CALUDE_base4_calculation_l1711_171132


namespace NUMINAMATH_CALUDE_required_run_rate_calculation_l1711_171199

/-- Represents a cricket game situation -/
structure CricketGame where
  totalOvers : ℕ
  firstInningOvers : ℕ
  firstInningRunRate : ℚ
  wicketsLost : ℕ
  targetScore : ℕ
  remainingRunsNeeded : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstInningOvers
  let runsScored := game.firstInningRunRate * game.firstInningOvers
  let actualRemainingRuns := game.targetScore - runsScored
  actualRemainingRuns / remainingOvers

/-- Theorem stating the required run rate for the given game situation -/
theorem required_run_rate_calculation (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstInningOvers = 20)
  (h3 : game.firstInningRunRate = 4.2)
  (h4 : game.wicketsLost = 5)
  (h5 : game.targetScore = 250)
  (h6 : game.remainingRunsNeeded = 195) :
  requiredRunRate game = 5.53 := by
  sorry

#eval requiredRunRate {
  totalOvers := 50,
  firstInningOvers := 20,
  firstInningRunRate := 4.2,
  wicketsLost := 5,
  targetScore := 250,
  remainingRunsNeeded := 195
}

end NUMINAMATH_CALUDE_required_run_rate_calculation_l1711_171199


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l1711_171142

theorem acute_triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (∀ x : ℝ, x^2 - 2 * Real.sqrt 3 * x + 2 = 0 → (x = a ∨ x = b)) →
  2 * Real.sin (A + B) - Real.sqrt 3 = 0 →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  c = Real.sqrt 6 ∧
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l1711_171142


namespace NUMINAMATH_CALUDE_mexico_city_car_restriction_l1711_171150

/-- The minimum number of cars needed for a family in Mexico City -/
def min_cars : ℕ := 14

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of restricted days per car per week -/
def restricted_days_per_car : ℕ := 2

/-- The minimum number of cars that must be available each day -/
def min_available_cars : ℕ := 10

theorem mexico_city_car_restriction :
  ∀ n : ℕ,
  n ≥ min_cars →
  (∀ d : ℕ, d < days_in_week →
    n - (n * restricted_days_per_car / days_in_week) ≥ min_available_cars) ∧
  (∀ m : ℕ, m < min_cars →
    ∃ d : ℕ, d < days_in_week ∧
      m - (m * restricted_days_per_car / days_in_week) < min_available_cars) :=
by sorry


end NUMINAMATH_CALUDE_mexico_city_car_restriction_l1711_171150


namespace NUMINAMATH_CALUDE_students_liking_both_sports_l1711_171104

/-- Given a class of students with information about their sports preferences,
    prove the number of students who like both basketball and table tennis. -/
theorem students_liking_both_sports
  (total : ℕ)
  (basketball : ℕ)
  (table_tennis : ℕ)
  (neither : ℕ)
  (h1 : total = 40)
  (h2 : basketball = 20)
  (h3 : table_tennis = 15)
  (h4 : neither = 8)
  : ∃ x : ℕ, x = 3 ∧ basketball + table_tennis - x + neither = total :=
by sorry

end NUMINAMATH_CALUDE_students_liking_both_sports_l1711_171104


namespace NUMINAMATH_CALUDE_quadratic_properties_l1711_171156

-- Define the quadratic function
def quadratic (a b t : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + t - 1

theorem quadratic_properties :
  ∀ (a b t : ℝ), t < 0 →
  -- Part 1
  (quadratic a b (-2) 1 = -4 ∧ quadratic a b (-2) (-1) = 0) → (a = 1 ∧ b = -2) ∧
  -- Part 2
  (2 * a - b = 1) → 
    ∃ (k p : ℝ), k ≠ 0 ∧ 
      ∀ (x : ℝ), (quadratic a b (-2) x = k * x + p) → 
        ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic a b (-2) x1 = k * x1 + p ∧ quadratic a b (-2) x2 = k * x2 + p ∧
  -- Part 3
  ∀ (m n : ℝ), m > 0 ∧ n > 0 →
    quadratic a b t (-1) = t ∧ quadratic a b t m = t - n ∧
    ((1/2) * n - 2 * t = (1/2) * (m + 1) * (quadratic a b t m - quadratic a b t (-1))) →
    (∀ (x : ℝ), -1 ≤ x ∧ x ≤ m → quadratic a b t x ≤ quadratic a b t (-1)) →
    ((0 < a ∧ a ≤ 1/3) ∨ (-1 ≤ a ∧ a < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1711_171156


namespace NUMINAMATH_CALUDE_matrix_power_calculation_l1711_171155

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 2, 1]

theorem matrix_power_calculation :
  (2 • A)^10 = !![1024, 0; 20480, 1024] := by sorry

end NUMINAMATH_CALUDE_matrix_power_calculation_l1711_171155


namespace NUMINAMATH_CALUDE_intersection_A_B_l1711_171198

open Set

def A : Set ℝ := {x | (x - 2) / x ≤ 0 ∧ x ≠ 0}
def B : Set ℝ := Icc (-1 : ℝ) 1

theorem intersection_A_B : A ∩ B = Ioc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1711_171198


namespace NUMINAMATH_CALUDE_power_of_power_l1711_171129

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1711_171129


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l1711_171106

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let z : ℂ := (3 * a + 2 * Complex.I) * (b - Complex.I)
  (z.re = 4) →
  (∀ x y : ℝ, x > 0 → y > 0 →
    let w : ℂ := (3 * x + 2 * Complex.I) * (y - Complex.I)
    w.re = 4 → 2 * x + y ≥ 2 * a + b) →
  2 * a + b = 4 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l1711_171106


namespace NUMINAMATH_CALUDE_farmer_apples_l1711_171127

/-- The number of apples given away by the farmer -/
def apples_given : ℕ := 88

/-- The number of apples left after giving some away -/
def apples_left : ℕ := 39

/-- The initial number of apples the farmer had -/
def initial_apples : ℕ := apples_given + apples_left

theorem farmer_apples : initial_apples = 127 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l1711_171127


namespace NUMINAMATH_CALUDE_complex_equation_proof_l1711_171107

theorem complex_equation_proof (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^2005 + (y / (x + y))^2005 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l1711_171107


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1711_171165

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 59405 / 30958 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1711_171165


namespace NUMINAMATH_CALUDE_fraction_problem_l1711_171103

theorem fraction_problem (a b c d e f : ℚ) :
  (∃ (k : ℚ), a = k * 1 ∧ b = k * 2 ∧ c = k * 5) →
  (∃ (m : ℚ), d = m * 1 ∧ e = m * 3 ∧ f = m * 7) →
  (a / d + b / e + c / f) / 3 = 200 / 441 →
  a / d = 4 / 7 ∧ b / e = 8 / 21 ∧ c / f = 20 / 49 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1711_171103


namespace NUMINAMATH_CALUDE_football_match_end_time_l1711_171146

-- Define a custom time type
structure Time where
  hour : Nat
  minute : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + m
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

-- State the theorem
theorem football_match_end_time :
  let start_time : Time := { hour := 15, minute := 30 }
  let duration : Nat := 145
  let end_time : Time := addMinutes start_time duration
  end_time = { hour := 17, minute := 55 } := by
  sorry

end NUMINAMATH_CALUDE_football_match_end_time_l1711_171146


namespace NUMINAMATH_CALUDE_expression_equals_six_l1711_171122

theorem expression_equals_six :
  2 - Real.sqrt 3 + (2 - Real.sqrt 3)⁻¹ + (Real.sqrt 3 + 2)⁻¹ = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_six_l1711_171122


namespace NUMINAMATH_CALUDE_remainder_theorem_l1711_171184

theorem remainder_theorem (r : ℝ) : (r^15 + 1) % (r + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1711_171184


namespace NUMINAMATH_CALUDE_parabola_circle_tangent_l1711_171124

/-- The value of p for a parabola y^2 = 2px (p > 0) whose directrix is tangent to the circle (x-3)^2 + y^2 = 16 -/
theorem parabola_circle_tangent (p : ℝ) : 
  p > 0 ∧ 
  (∃ (x y : ℝ), y^2 = 2*p*x) ∧ 
  (∃ (x y : ℝ), (x - 3)^2 + y^2 = 16) ∧
  (∃ (x : ℝ), x = -p/2 ∧ (x - 3)^2 + (2*p*x) = 16) →
  p = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangent_l1711_171124


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1711_171197

theorem arithmetic_calculation : (30 / (10 + 2 - 5) + 4) * 7 = 58 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1711_171197


namespace NUMINAMATH_CALUDE_circle_equation_l1711_171167

/-- A circle with center on the line x + y = 0 and intersecting the x-axis at (-3, 0) and (1, 0) -/
structure Circle where
  center : ℝ × ℝ
  center_on_line : center.1 + center.2 = 0
  intersects_x_axis : ∃ (t : ℝ), t^2 = (center.1 + 3)^2 + center.2^2 ∧ t^2 = (center.1 - 1)^2 + center.2^2

/-- The equation of the circle is (x+1)² + (y-1)² = 5 -/
theorem circle_equation (c : Circle) : 
  ∀ (x y : ℝ), (x + 1)^2 + (y - 1)^2 = 5 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = ((c.center.1 + 3)^2 + c.center.2^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1711_171167


namespace NUMINAMATH_CALUDE_roof_length_width_difference_l1711_171112

/-- Represents a rectangular roof -/
structure RectangularRoof where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Theorem: For a rectangular roof with length 4 times the width and area 784 sq ft,
    the difference between length and width is 42 ft -/
theorem roof_length_width_difference 
  (roof : RectangularRoof)
  (h1 : roof.length = 4 * roof.width)
  (h2 : roof.area = 784)
  (h3 : roof.area = roof.length * roof.width) :
  roof.length - roof.width = 42 := by
  sorry


end NUMINAMATH_CALUDE_roof_length_width_difference_l1711_171112


namespace NUMINAMATH_CALUDE_g_of_neg_three_eq_one_l1711_171134

/-- Given a function g(x) = (3x + 4) / (x - 2), prove that g(-3) = 1 -/
theorem g_of_neg_three_eq_one (g : ℝ → ℝ) (h : ∀ x, x ≠ 2 → g x = (3 * x + 4) / (x - 2)) : 
  g (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_of_neg_three_eq_one_l1711_171134


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l1711_171145

theorem divisible_by_eleven (n : ℕ) : ∃ k : ℤ, 5^(2*n) + 3^(n+2) + 3^n = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l1711_171145


namespace NUMINAMATH_CALUDE_tom_rides_11860_miles_l1711_171131

/-- Tom's daily bike riding distance for the first part of the year -/
def first_part_distance : ℕ := 30

/-- Number of days in the first part of the year -/
def first_part_days : ℕ := 183

/-- Tom's daily bike riding distance for the second part of the year -/
def second_part_distance : ℕ := 35

/-- Total number of days in a year -/
def total_days : ℕ := 365

/-- Calculate the total miles Tom rides in a year -/
def total_miles : ℕ := first_part_distance * first_part_days + 
                        second_part_distance * (total_days - first_part_days)

theorem tom_rides_11860_miles : total_miles = 11860 := by
  sorry

end NUMINAMATH_CALUDE_tom_rides_11860_miles_l1711_171131


namespace NUMINAMATH_CALUDE_jane_rejection_proof_l1711_171139

/-- Represents the percentage of products Jane rejected -/
def jane_rejection_rate : ℝ := 0.9

/-- Represents the percentage of products John rejected -/
def john_rejection_rate : ℝ := 0.5

/-- Represents the fraction of total products Jane inspected -/
def jane_inspection_fraction : ℝ := 0.625

/-- Represents the total percentage of rejected products -/
def total_rejection_rate : ℝ := 0.75

/-- Theorem stating that given the conditions, Jane's rejection rate is 0.9% -/
theorem jane_rejection_proof :
  john_rejection_rate * (1 - jane_inspection_fraction) +
  jane_rejection_rate * jane_inspection_fraction / 100 =
  total_rejection_rate / 100 := by
  sorry

#check jane_rejection_proof

end NUMINAMATH_CALUDE_jane_rejection_proof_l1711_171139


namespace NUMINAMATH_CALUDE_james_and_david_probability_l1711_171162

def total_workers : ℕ := 14
def workers_to_choose : ℕ := 2

theorem james_and_david_probability :
  (1 : ℚ) / (Nat.choose total_workers workers_to_choose) = 1 / 91 := by
  sorry

end NUMINAMATH_CALUDE_james_and_david_probability_l1711_171162


namespace NUMINAMATH_CALUDE_x1_x2_range_l1711_171113

noncomputable section

def f (x : ℝ) : ℝ := if x ≥ 1 then Real.log x else 1 - x / 2

def F (m : ℝ) (x : ℝ) : ℝ := f (f x + 1) + m

theorem x1_x2_range (m : ℝ) (x₁ x₂ : ℝ) (h₁ : F m x₁ = 0) (h₂ : F m x₂ = 0) (h₃ : x₁ ≠ x₂) :
  x₁ * x₂ < Real.sqrt (Real.exp 1) ∧ ∀ y : ℝ, ∃ m : ℝ, ∃ x₁ x₂ : ℝ, 
    F m x₁ = 0 ∧ F m x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁ * x₂ < y :=
by sorry

end

end NUMINAMATH_CALUDE_x1_x2_range_l1711_171113


namespace NUMINAMATH_CALUDE_continuous_at_3_l1711_171100

def f (x : ℝ) : ℝ := -3 * x^2 - 9

theorem continuous_at_3 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x - f 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuous_at_3_l1711_171100


namespace NUMINAMATH_CALUDE_function_properties_and_triangle_l1711_171170

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin x + Real.cos x

theorem function_properties_and_triangle (m : ℝ) (A : ℝ) :
  f m (π / 2) = 1 →
  f m (π / 12) = Real.sqrt 2 * Real.sin A →
  0 < A ∧ A < π / 2 →
  (3 * Real.sqrt 3) / 2 = 1 / 2 * 2 * 3 * Real.sin A →
  m = 1 ∧
  (∀ x : ℝ, f m (x + 2 * π) = f m x) ∧
  (∀ x : ℝ, f m x ≤ Real.sqrt 2) ∧
  (∀ x : ℝ, f m x ≥ -Real.sqrt 2) ∧
  A = π / 3 ∧
  Real.sqrt 7 = Real.sqrt (3^2 + 2^2 - 2 * 2 * 3 * Real.cos A) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_and_triangle_l1711_171170


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1711_171123

/-- Given an arithmetic sequence with a non-zero common difference,
    if its second, third, and sixth terms form a geometric sequence,
    then the common ratio of this geometric sequence is 3. -/
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) -- The arithmetic sequence
  (d : ℝ) -- The common difference of the arithmetic sequence
  (h1 : d ≠ 0) -- The common difference is non-zero
  (h2 : ∀ n, a (n + 1) = a n + d) -- Definition of arithmetic sequence
  (h3 : ∃ r, r ≠ 0 ∧ a 3 = r * a 2 ∧ a 6 = r * a 3) -- Second, third, and sixth terms form a geometric sequence
  : ∃ r, r = 3 ∧ a 3 = r * a 2 ∧ a 6 = r * a 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1711_171123


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l1711_171174

theorem smallest_x_for_equation : 
  ∃ (x : ℕ+), x = 4 ∧ 
  (∀ (y : ℕ+), (3 : ℚ) / 4 = (y : ℚ) / (200 + x)) ∧
  (∀ (x' : ℕ+), x' < x → 
    ¬∃ (y : ℕ+), (3 : ℚ) / 4 = (y : ℚ) / (200 + x')) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l1711_171174


namespace NUMINAMATH_CALUDE_house_number_painting_cost_l1711_171133

/-- Calculates the sum of digits for numbers in an arithmetic sequence --/
def sumOfDigits (start : ℕ) (diff : ℕ) (count : ℕ) : ℕ :=
  sorry

/-- Calculates the total cost of painting house numbers --/
def totalCost (southStart southDiff northStart northDiff housesPerSide : ℕ) : ℕ :=
  sorry

theorem house_number_painting_cost :
  totalCost 5 7 7 8 25 = 125 :=
sorry

end NUMINAMATH_CALUDE_house_number_painting_cost_l1711_171133


namespace NUMINAMATH_CALUDE_unique_triangle_with_perimeter_8_l1711_171164

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of an integer triangle -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- Two triangles are congruent if they have the same side lengths (up to permutation) -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.a ∧ t1.b = t2.c ∧ t1.c = t2.b) ∨
  (t1.a = t2.b ∧ t1.b = t2.a ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b) ∨
  (t1.a = t2.c ∧ t1.b = t2.b ∧ t1.c = t2.a)

theorem unique_triangle_with_perimeter_8 :
  ∃! (t : IntTriangle), perimeter t = 8 ∧
  ∀ (t' : IntTriangle), perimeter t' = 8 → congruent t t' := by
  sorry

end NUMINAMATH_CALUDE_unique_triangle_with_perimeter_8_l1711_171164


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l1711_171136

theorem least_n_satisfying_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → m < n → ¬(1 / m - 1 / (m + 1) < 1 / 10)) ∧ 
  (1 / n - 1 / (n + 1) < 1 / 10) ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l1711_171136


namespace NUMINAMATH_CALUDE_bug_path_tiles_l1711_171130

/-- The number of tiles a bug visits when walking diagonally across a rectangular floor -/
def tiles_visited (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- Theorem stating that a bug walking diagonally across an 18x24 rectangular floor visits 36 tiles -/
theorem bug_path_tiles : tiles_visited 18 24 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l1711_171130


namespace NUMINAMATH_CALUDE_rhombus_circumcircle_radii_l1711_171194

/-- A rhombus with circumcircles of two triangles formed by its sides -/
structure RhombusWithCircumcircles where
  /-- Side length of the rhombus -/
  side_length : ℝ
  /-- Distance between centers of circumcircles -/
  center_distance : ℝ
  /-- Radius of the circumcircle of triangle ABC -/
  radius_ABC : ℝ
  /-- Radius of the circumcircle of triangle BCD -/
  radius_BCD : ℝ
  /-- The side length is positive -/
  side_length_pos : 0 < side_length
  /-- The center distance is positive -/
  center_distance_pos : 0 < center_distance

/-- Theorem about the radii of circumcircles in a specific rhombus configuration -/
theorem rhombus_circumcircle_radii
  (r : RhombusWithCircumcircles)
  (h1 : r.side_length = 6)
  (h2 : r.center_distance = 8) :
  r.radius_ABC = 3 * Real.sqrt 10 ∧ r.radius_BCD = 3 * Real.sqrt 10 := by
  sorry

#check rhombus_circumcircle_radii

end NUMINAMATH_CALUDE_rhombus_circumcircle_radii_l1711_171194


namespace NUMINAMATH_CALUDE_m_range_l1711_171166

-- Define the propositions p and q
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem m_range (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1711_171166


namespace NUMINAMATH_CALUDE_min_sum_of_digits_f_l1711_171193

def f (n : ℕ) : ℕ := 17 * n^2 - 11 * n + 1

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem min_sum_of_digits_f :
  ∀ n : ℕ, sum_of_digits (f n) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_digits_f_l1711_171193


namespace NUMINAMATH_CALUDE_guest_speaker_payment_l1711_171137

theorem guest_speaker_payment (B : Nat) : 
  B < 10 → (100 * 2 + 10 * B + 5) % 13 = 0 → B = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_guest_speaker_payment_l1711_171137


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1711_171114

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

theorem solution_set_of_inequality 
  (f : ℝ → ℝ) 
  (h_decreasing : is_decreasing f) 
  (h_point1 : f 0 = 1) 
  (h_point2 : f 3 = -1) :
  {x : ℝ | |f (x + 1)| < 1} = Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1711_171114


namespace NUMINAMATH_CALUDE_motorcycle_cyclist_meeting_times_l1711_171149

theorem motorcycle_cyclist_meeting_times 
  (angle : Real) 
  (cyclist_speed : Real) 
  (motorcyclist_speed : Real) 
  (t : Real) : 
  angle = π / 3 →
  cyclist_speed = 36 →
  motorcyclist_speed = 72 →
  (motorcyclist_speed^2 * t^2 + cyclist_speed^2 * (t - 1)^2 - 
   2 * motorcyclist_speed * cyclist_speed * |t| * |t - 1| * (1/2) = 252^2) →
  (t = 4 ∨ t = -4) :=
by sorry

end NUMINAMATH_CALUDE_motorcycle_cyclist_meeting_times_l1711_171149


namespace NUMINAMATH_CALUDE_straight_line_probability_l1711_171157

/-- The number of dots in each row or column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots required to form a line -/
def dotsInLine : ℕ := 4

/-- The number of possible straight lines containing four dots in a 5x5 grid -/
def numStraightLines : ℕ := 16

/-- The total number of ways to choose 4 dots from 25 dots -/
def totalWaysToChoose : ℕ := Nat.choose totalDots dotsInLine

/-- The probability of selecting four dots that form a straight line -/
def probabilityOfStraightLine : ℚ := numStraightLines / totalWaysToChoose

theorem straight_line_probability :
  probabilityOfStraightLine = 16 / 12650 := by sorry

end NUMINAMATH_CALUDE_straight_line_probability_l1711_171157


namespace NUMINAMATH_CALUDE_square_wood_weight_l1711_171117

/-- Represents the properties of a piece of wood -/
structure Wood where
  length : ℝ
  width : ℝ
  weight : ℝ

/-- Calculates the area of a piece of wood -/
def area (w : Wood) : ℝ := w.length * w.width

/-- Theorem stating the weight of the square piece of wood -/
theorem square_wood_weight (rect : Wood) (square : Wood) :
  rect.length = 4 ∧ 
  rect.width = 6 ∧ 
  rect.weight = 24 ∧
  square.length = 5 ∧
  square.width = 5 →
  square.weight = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_wood_weight_l1711_171117


namespace NUMINAMATH_CALUDE_f_minimum_value_l1711_171160

noncomputable def f (x : ℝ) : ℝ :=
  x + (2*x)/(x^2 + 1) + (x*(x + 5))/(x^2 + 3) + (3*(x + 3))/(x*(x^2 + 3))

theorem f_minimum_value (x : ℝ) (h : x > 0) : f x ≥ 5.5 := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l1711_171160


namespace NUMINAMATH_CALUDE_circle_radius_l1711_171105

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the axis of symmetry
def axis_of_symmetry (x : ℝ) : Prop := x = -1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 3 = 0

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- State the theorem
theorem circle_radius :
  ∀ C : Circle,
  (C.center.1 = -1) →  -- Center is on the axis of symmetry
  (C.center.2 ≠ 0) →  -- Center is not on the x-axis
  (C.center.1 + C.radius)^2 + C.center.2^2 = C.radius^2 →  -- Circle passes through the focus
  (∃ (x y : ℝ), tangent_line x y ∧ 
    ((x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2)) →  -- Circle is tangent to the line
  C.radius = 14 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l1711_171105


namespace NUMINAMATH_CALUDE_probability_sum_6_is_5_36_l1711_171168

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := faces * faces

/-- The number of combinations that result in a sum of 6 -/
def favorable_outcomes : ℕ := 5

/-- The probability of rolling a sum of 6 with two dice -/
def probability_sum_6 : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_6_is_5_36 : probability_sum_6 = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_6_is_5_36_l1711_171168


namespace NUMINAMATH_CALUDE_integral_of_polynomial_l1711_171121

theorem integral_of_polynomial : ∫ (x : ℝ) in (0)..(2), (3*x^2 + 4*x^3) = 24 := by sorry

end NUMINAMATH_CALUDE_integral_of_polynomial_l1711_171121


namespace NUMINAMATH_CALUDE_lexie_age_l1711_171188

/-- Represents the ages of Lexie, her brother, and her sister -/
structure Family where
  lexie : ℕ
  brother : ℕ
  sister : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (f : Family) : Prop :=
  f.lexie = f.brother + 6 ∧
  f.sister = 2 * f.lexie ∧
  f.sister - f.brother = 14

/-- Theorem stating that if a family satisfies the given conditions, Lexie's age is 8 -/
theorem lexie_age (f : Family) (h : satisfiesConditions f) : f.lexie = 8 := by
  sorry

#check lexie_age

end NUMINAMATH_CALUDE_lexie_age_l1711_171188


namespace NUMINAMATH_CALUDE_min_value_of_sqrt_sums_l1711_171148

theorem min_value_of_sqrt_sums (a b c : ℝ) : 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c → 
  a * b + b * c + c * a = a + b + c → 
  0 < a + b + c → 
  2 ≤ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sqrt_sums_l1711_171148


namespace NUMINAMATH_CALUDE_initial_amount_proof_l1711_171152

/-- The initial amount of money Edward had -/
def initial_amount : ℝ := 17.80

/-- The cost of one toy car -/
def toy_car_cost : ℝ := 0.95

/-- The number of toy cars Edward bought -/
def num_toy_cars : ℕ := 4

/-- The cost of the race track -/
def race_track_cost : ℝ := 6.00

/-- The amount of money Edward has left -/
def remaining_money : ℝ := 8.00

/-- Theorem stating that the initial amount is equal to the sum of expenses and remaining money -/
theorem initial_amount_proof : 
  initial_amount = num_toy_cars * toy_car_cost + race_track_cost + remaining_money :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l1711_171152


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1711_171138

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℝ),
    (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 →
      5 * x^2 / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) ∧
    P = 20 ∧ Q = -15 ∧ R = -10 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1711_171138


namespace NUMINAMATH_CALUDE_max_largest_integer_l1711_171154

theorem max_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 70 →
  max a (max b (max c (max d e))) - min a (min b (min c (min d e))) = 10 →
  max a (max b (max c (max d e))) ≤ 340 :=
sorry

end NUMINAMATH_CALUDE_max_largest_integer_l1711_171154


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1711_171163

theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) : z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1711_171163


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_one_plus_reciprocal_product_l1711_171141

theorem sum_reciprocals_equals_one_plus_reciprocal_product 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y + 1) : 
  1 / x + 1 / y = 1 + 1 / (x * y) := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_one_plus_reciprocal_product_l1711_171141


namespace NUMINAMATH_CALUDE_pirate_treasure_l1711_171169

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_l1711_171169


namespace NUMINAMATH_CALUDE_baseball_league_games_played_l1711_171176

/-- Represents a baseball league with a given number of teams and games per pair of teams. -/
structure BaseballLeague where
  num_teams : ℕ
  games_per_pair : ℕ

/-- Calculates the total number of games played in a baseball league with one team forfeiting one game against each other team. -/
def games_actually_played (league : BaseballLeague) : ℕ :=
  let total_scheduled := (league.num_teams * (league.num_teams - 1) * league.games_per_pair) / 2
  let forfeited_games := league.num_teams - 1
  total_scheduled - forfeited_games

/-- Theorem stating that in a baseball league with 10 teams, 4 games per pair, and one team forfeiting one game against each other team, the total number of games actually played is 171. -/
theorem baseball_league_games_played :
  let league : BaseballLeague := { num_teams := 10, games_per_pair := 4 }
  games_actually_played league = 171 := by
  sorry

end NUMINAMATH_CALUDE_baseball_league_games_played_l1711_171176


namespace NUMINAMATH_CALUDE_one_incorrect_statement_l1711_171191

-- Define the structure for a statistical statement
structure StatStatement :=
  (id : Nat)
  (content : String)
  (isCorrect : Bool)

-- Define the list of statements
def statements : List StatStatement :=
  [
    { id := 1, content := "Residuals can be used to judge the effectiveness of model fitting", isCorrect := true },
    { id := 2, content := "Given a regression equation: ŷ=3-5x, when variable x increases by one unit, y increases by an average of 5 units", isCorrect := false },
    { id := 3, content := "The linear regression line: ŷ=b̂x+â must pass through the point (x̄, ȳ)", isCorrect := true },
    { id := 4, content := "In a 2×2 contingency table, it is calculated that χ²=13.079, thus there is a 99% confidence that there is a relationship between the two variables (where P(χ²≥10.828)=0.001)", isCorrect := true }
  ]

-- Theorem: Exactly one statement is incorrect
theorem one_incorrect_statement : 
  (statements.filter (fun s => !s.isCorrect)).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_incorrect_statement_l1711_171191


namespace NUMINAMATH_CALUDE_fence_width_is_ten_l1711_171171

/-- A rectangular fence with specific properties -/
structure RectangularFence where
  circumference : ℝ
  length : ℝ
  width : ℝ
  circ_eq : circumference = 2 * (length + width)
  width_eq : width = 2 * length

/-- The width of a rectangular fence with circumference 30m and width twice the length is 10m -/
theorem fence_width_is_ten (fence : RectangularFence) 
    (h_circ : fence.circumference = 30) : fence.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_fence_width_is_ten_l1711_171171


namespace NUMINAMATH_CALUDE_alex_cake_slices_l1711_171173

theorem alex_cake_slices (total_slices : ℕ) (cakes : ℕ) : 
  cakes = 2 →
  (total_slices / 4 : ℚ) + (3 * total_slices / 4 / 3 : ℚ) + 3 + 5 = total_slices →
  total_slices / cakes = 8 := by
sorry

end NUMINAMATH_CALUDE_alex_cake_slices_l1711_171173


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1711_171102

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 3 * a 5 = 4 →
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 = 128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1711_171102


namespace NUMINAMATH_CALUDE_sequence_sum_property_l1711_171101

theorem sequence_sum_property (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n : ℕ+, S n = n^2 - a n) →
  (∃ k : ℕ+, 1 < S k ∧ S k < 9) →
  (∃ k : ℕ+, k = 2 ∧ 1 < S k ∧ S k < 9) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l1711_171101


namespace NUMINAMATH_CALUDE_molecular_weight_aluminum_iodide_l1711_171109

/-- Given that the molecular weight of 7 moles of aluminum iodide is 2856 grams,
    prove that the molecular weight of one mole of aluminum iodide is 408 grams/mole. -/
theorem molecular_weight_aluminum_iodide :
  let total_weight : ℝ := 2856
  let num_moles : ℝ := 7
  total_weight / num_moles = 408 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_aluminum_iodide_l1711_171109


namespace NUMINAMATH_CALUDE_mean_temperature_l1711_171182

def temperatures : List ℤ := [-6, -3, -3, -4, 2, 4, 1]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -9/7 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l1711_171182


namespace NUMINAMATH_CALUDE_simplify_expressions_l1711_171195

theorem simplify_expressions : 
  (2 * Real.sqrt 12 + 3 * Real.sqrt 3 - Real.sqrt 27 = 4 * Real.sqrt 3) ∧
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l1711_171195


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1711_171190

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 5*x^4 + 8*x^2 - 4 = (x-1)*(x+1)*(x^2-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1711_171190


namespace NUMINAMATH_CALUDE_divide_into_three_unequal_groups_divide_into_three_equal_groups_divide_among_three_people_l1711_171115

-- Define the number of books
def n : ℕ := 6

-- Theorem for the first question
theorem divide_into_three_unequal_groups :
  (n.choose 1) * ((n - 1).choose 2) * ((n - 3).choose 3) = 60 := by sorry

-- Theorem for the second question
theorem divide_into_three_equal_groups :
  (n.choose 2 * (n - 2).choose 2 * (n - 4).choose 2) / 6 = 15 := by sorry

-- Theorem for the third question
theorem divide_among_three_people :
  n.choose 2 * (n - 2).choose 2 * (n - 4).choose 2 = 90 := by sorry

end NUMINAMATH_CALUDE_divide_into_three_unequal_groups_divide_into_three_equal_groups_divide_among_three_people_l1711_171115


namespace NUMINAMATH_CALUDE_third_month_sale_l1711_171161

def average_sale : ℝ := 6500
def num_months : ℕ := 6
def sale_month1 : ℝ := 6435
def sale_month2 : ℝ := 6927
def sale_month4 : ℝ := 7230
def sale_month5 : ℝ := 6562
def sale_month6 : ℝ := 4991

theorem third_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6
  total_sales - known_sales = 6855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l1711_171161


namespace NUMINAMATH_CALUDE_equation_solution_l1711_171111

theorem equation_solution (x y z : ℚ) 
  (eq1 : x - 4*y - 2*z = 0) 
  (eq2 : 3*x + 2*y - z = 0) 
  (z_neq_zero : z ≠ 0) : 
  (x^2 - 5*x*y) / (2*y^2 + z^2) = 164/147 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1711_171111


namespace NUMINAMATH_CALUDE_tan_half_angle_formula_22_5_degrees_l1711_171144

theorem tan_half_angle_formula_22_5_degrees : 
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_angle_formula_22_5_degrees_l1711_171144


namespace NUMINAMATH_CALUDE_trapezoid_base_lengths_l1711_171126

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  b : ℝ  -- smaller base
  h : ℝ  -- altitude
  B : ℝ  -- larger base
  d : ℝ  -- common difference
  arithmetic_progression : b = h - 2 * d ∧ B = h + 2 * d
  area : (b + B) * h / 2 = 48

/-- Theorem stating the base lengths of the trapezoid -/
theorem trapezoid_base_lengths (t : Trapezoid) : 
  t.b = Real.sqrt 48 - 2 * Real.sqrt 3 ∧ 
  t.B = Real.sqrt 48 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_base_lengths_l1711_171126


namespace NUMINAMATH_CALUDE_street_tree_count_l1711_171183

theorem street_tree_count (road_length : ℕ) (interval : ℕ) (h1 : road_length = 2575) (h2 : interval = 25) : 
  2 * (road_length / interval + 1) = 208 := by
  sorry

end NUMINAMATH_CALUDE_street_tree_count_l1711_171183


namespace NUMINAMATH_CALUDE_polynomial_identity_l1711_171159

theorem polynomial_identity (a b c : ℤ) (h_c_odd : Odd c) :
  let P : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c
  let p : ℕ → ℝ := fun i ↦ P i
  (p 1)^3 + (p 2)^3 + (p 3)^3 = 3*(p 1)*(p 2)*(p 3) →
  (p 2) + 2*(p 1) - 3*(p 0) = 18 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1711_171159


namespace NUMINAMATH_CALUDE_alex_jane_pen_difference_l1711_171153

def pens_after_n_weeks (initial_pens : ℕ) (n : ℕ) : ℕ :=
  initial_pens * (2 ^ n)

theorem alex_jane_pen_difference :
  let alex_initial_pens : ℕ := 4
  let weeks_in_month : ℕ := 4
  let jane_pens : ℕ := 16
  pens_after_n_weeks alex_initial_pens weeks_in_month - jane_pens = 16 :=
by
  sorry

#check alex_jane_pen_difference

end NUMINAMATH_CALUDE_alex_jane_pen_difference_l1711_171153


namespace NUMINAMATH_CALUDE_bound_difference_for_elements_in_A_l1711_171196

/-- The function f(x) = |x+2| + |x-2| -/
def f (x : ℝ) : ℝ := |x + 2| + |x - 2|

/-- The set A of all x such that f(x) ≤ 6 -/
def A : Set ℝ := {x | f x ≤ 6}

/-- Theorem stating that if m and n are in A, then |1/3 * m - 1/2 * n| ≤ 5/2 -/
theorem bound_difference_for_elements_in_A (m n : ℝ) (hm : m ∈ A) (hn : n ∈ A) :
  |1/3 * m - 1/2 * n| ≤ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_bound_difference_for_elements_in_A_l1711_171196


namespace NUMINAMATH_CALUDE_domino_placement_l1711_171140

/-- The maximum number of 1 × k dominos that can be placed on an n × n chessboard. -/
def max_dominos (n k : ℕ) : ℕ :=
  if n = k ∨ n = 2*k - 1 then n
  else if k < n ∧ n < 2*k - 1 then 2*n - 2*k + 2
  else 0

theorem domino_placement (n k : ℕ) (h1 : k ≤ n) (h2 : n < 2*k) :
  max_dominos n k = if n = k ∨ n = 2*k - 1 then n else 2*n - 2*k + 2 :=
by sorry

end NUMINAMATH_CALUDE_domino_placement_l1711_171140


namespace NUMINAMATH_CALUDE_factory_problem_l1711_171125

/-- Represents a factory with workers and production methods -/
structure Factory where
  total_workers : ℕ
  production_increase : ℝ → ℝ
  new_method_factor : ℝ

/-- The conditions and proof goals for the factory problem -/
theorem factory_problem (f : Factory) : 
  (f.production_increase (40 / f.total_workers) = 1.2) →
  (f.production_increase 0.6 = 2.5) →
  (f.total_workers = 500 ∧ f.new_method_factor = 3.5) := by
  sorry


end NUMINAMATH_CALUDE_factory_problem_l1711_171125


namespace NUMINAMATH_CALUDE_student_count_l1711_171116

theorem student_count : ∃ S : ℕ, 
  S = 92 ∧ 
  (3 / 8 : ℚ) * (S - 20 : ℚ) = 27 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1711_171116


namespace NUMINAMATH_CALUDE_iris_mall_spending_l1711_171120

/-- The total amount spent by Iris at the mall --/
def total_spent (jacket_price shorts_price pants_price : ℕ) 
                (jacket_count shorts_count pants_count : ℕ) : ℕ :=
  jacket_price * jacket_count + shorts_price * shorts_count + pants_price * pants_count

/-- Theorem stating that Iris spent $90 at the mall --/
theorem iris_mall_spending : 
  total_spent 10 6 12 3 2 4 = 90 := by
  sorry

end NUMINAMATH_CALUDE_iris_mall_spending_l1711_171120


namespace NUMINAMATH_CALUDE_lansing_elementary_schools_l1711_171192

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 6175 / 247

/-- Theorem: There are 25 elementary schools in Lansing -/
theorem lansing_elementary_schools : num_schools = 25 := by
  sorry

end NUMINAMATH_CALUDE_lansing_elementary_schools_l1711_171192


namespace NUMINAMATH_CALUDE_distribute_basketballs_count_l1711_171177

/-- The number of ways to distribute four labeled basketballs among three kids -/
def distribute_basketballs : ℕ :=
  30

/-- Each kid must get at least one basketball -/
axiom each_kid_gets_one : True

/-- Basketballs are labeled 1, 2, 3, and 4 -/
axiom basketballs_labeled : True

/-- Basketballs 1 and 2 cannot be given to the same kid -/
axiom one_and_two_separate : True

/-- The number of ways to distribute the basketballs satisfying all conditions is 30 -/
theorem distribute_basketballs_count :
  distribute_basketballs = 30 :=
by sorry

end NUMINAMATH_CALUDE_distribute_basketballs_count_l1711_171177


namespace NUMINAMATH_CALUDE_smallest_n_for_locker_one_open_l1711_171151

/-- Represents the state of lockers -/
def LockerState := List Bool

/-- The rule for closing lockers -/
def closeLockers (state : LockerState) (start : Nat) (count : Nat) : LockerState :=
  sorry

/-- Simulates Ansoon's process of closing lockers -/
def ansoonProcess (n : Nat) : Nat :=
  sorry

/-- Theorem stating that 2046 is the smallest N > 2021 where locker 1 is last open -/
theorem smallest_n_for_locker_one_open (n : Nat) :
  n > 2021 → (ansoonProcess n = 1 → n ≥ 2046) ∧ (ansoonProcess 2046 = 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_locker_one_open_l1711_171151


namespace NUMINAMATH_CALUDE_operation_equivalence_l1711_171143

-- Define the set of operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply the operation
def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_equivalence 
  (star mul : Operation) 
  (h_unique : star ≠ mul) 
  (h_eq : (apply_op star 16 4) / (apply_op mul 10 2) = 4) :
  (apply_op star 5 15) / (apply_op mul 8 12) = 30 := by
  sorry


end NUMINAMATH_CALUDE_operation_equivalence_l1711_171143


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1711_171179

theorem stratified_sample_size 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (sampled_male : ℕ) 
  (h1 : total_employees = 120) 
  (h2 : male_employees = 90) 
  (h3 : sampled_male = 27) 
  (h4 : male_employees < total_employees) :
  (sampled_male : ℚ) / (male_employees : ℚ) * (total_employees : ℚ) = 36 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l1711_171179


namespace NUMINAMATH_CALUDE_test_score_problem_l1711_171110

/-- Proves that given a test with 25 problems, a scoring system of 4 points for each correct answer
    and -1 point for each wrong answer, and a total score of 85, the number of wrong answers is 3. -/
theorem test_score_problem (total_problems : Nat) (right_points : Int) (wrong_points : Int) (total_score : Int)
    (h1 : total_problems = 25)
    (h2 : right_points = 4)
    (h3 : wrong_points = -1)
    (h4 : total_score = 85) :
    ∃ (right : Nat) (wrong : Nat),
      right + wrong = total_problems ∧
      right_points * right + wrong_points * wrong = total_score ∧
      wrong = 3 :=
by sorry

end NUMINAMATH_CALUDE_test_score_problem_l1711_171110


namespace NUMINAMATH_CALUDE_total_income_is_139_80_l1711_171172

/-- Represents a pastry item with its original price, discount rate, and quantity sold. -/
structure Pastry where
  name : String
  originalPrice : Float
  discountRate : Float
  quantitySold : Nat

/-- Calculates the total income generated from selling pastries after applying discounts. -/
def calculateTotalIncome (pastries : List Pastry) : Float :=
  pastries.foldl (fun acc pastry =>
    let discountedPrice := pastry.originalPrice * (1 - pastry.discountRate)
    acc + discountedPrice * pastry.quantitySold.toFloat
  ) 0

/-- Theorem stating that the total income from the given pastries is $139.80. -/
theorem total_income_is_139_80 : 
  let pastries : List Pastry := [
    { name := "Cupcakes", originalPrice := 3.00, discountRate := 0.30, quantitySold := 25 },
    { name := "Cookies", originalPrice := 2.00, discountRate := 0.45, quantitySold := 18 },
    { name := "Brownies", originalPrice := 4.00, discountRate := 0.25, quantitySold := 15 },
    { name := "Macarons", originalPrice := 1.50, discountRate := 0.50, quantitySold := 30 }
  ]
  calculateTotalIncome pastries = 139.80 := by
  sorry

end NUMINAMATH_CALUDE_total_income_is_139_80_l1711_171172


namespace NUMINAMATH_CALUDE_compute_expression_l1711_171186

theorem compute_expression : 75 * 1313 - 25 * 1313 = 65650 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1711_171186


namespace NUMINAMATH_CALUDE_technician_round_trip_l1711_171135

theorem technician_round_trip (D : ℝ) (h : D > 0) :
  let total_distance := 2 * D
  let distance_traveled := 0.55 * total_distance
  let return_distance := distance_traveled - D
  return_distance / D = 0.1 := by
sorry

end NUMINAMATH_CALUDE_technician_round_trip_l1711_171135


namespace NUMINAMATH_CALUDE_radii_and_circles_regions_l1711_171147

/-- The number of regions created by radii and concentric circles inside a larger circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem stating that 16 radii and 10 concentric circles create 176 regions -/
theorem radii_and_circles_regions :
  num_regions 16 10 = 176 := by
  sorry

end NUMINAMATH_CALUDE_radii_and_circles_regions_l1711_171147


namespace NUMINAMATH_CALUDE_bug_crawl_distance_l1711_171108

-- Define the positions of the bug
def start_pos : ℤ := 3
def pos1 : ℤ := -5
def pos2 : ℤ := 8
def end_pos : ℤ := 0

-- Define the function to calculate distance between two points
def distance (a b : ℤ) : ℕ := (a - b).natAbs

-- Define the total distance
def total_distance : ℕ := 
  distance start_pos pos1 + distance pos1 pos2 + distance pos2 end_pos

-- Theorem to prove
theorem bug_crawl_distance : total_distance = 29 := by
  sorry

end NUMINAMATH_CALUDE_bug_crawl_distance_l1711_171108


namespace NUMINAMATH_CALUDE_cube_roots_of_negative_one_l1711_171158

theorem cube_roots_of_negative_one :
  let z₁ : ℂ := -1
  let z₂ : ℂ := (1 + Complex.I * Real.sqrt 3) / 2
  let z₃ : ℂ := (1 - Complex.I * Real.sqrt 3) / 2
  (∀ z : ℂ, z^3 = -1 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end NUMINAMATH_CALUDE_cube_roots_of_negative_one_l1711_171158


namespace NUMINAMATH_CALUDE_remaining_customers_l1711_171128

/-- Proves that the number of customers remaining after some left is 5 -/
theorem remaining_customers (initial : ℕ) (remaining : ℕ) (new : ℕ) (final : ℕ)
  (h1 : initial = 8)
  (h2 : remaining < initial)
  (h3 : new = 99)
  (h4 : final = 104)
  (h5 : remaining + new = final) :
  remaining = 5 := by sorry

end NUMINAMATH_CALUDE_remaining_customers_l1711_171128


namespace NUMINAMATH_CALUDE_speed_ratio_l1711_171180

/-- Two people walk in opposite directions for 1 hour and swap destinations -/
structure WalkProblem where
  /-- Speed of person A in km/h -/
  v₁ : ℝ
  /-- Speed of person B in km/h -/
  v₂ : ℝ
  /-- Both speeds are positive -/
  h₁ : v₁ > 0
  h₂ : v₂ > 0
  /-- Person A reaches B's destination 35 minutes after B reaches A's destination -/
  h₃ : v₂ / v₁ - v₁ / v₂ = 35 / 60

/-- The ratio of speeds is 3:4 -/
theorem speed_ratio (w : WalkProblem) : w.v₁ / w.v₂ = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_l1711_171180


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1711_171175

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I)*z = 4 + 3*I) : z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1711_171175


namespace NUMINAMATH_CALUDE_intersection_sum_l1711_171185

/-- Given two functions f and g defined as:
    f(x) = -2|x-a| + b
    g(x) = 2|x-c| + d
    If f and g intersect at points (10, 15) and (18, 7),
    then a + c = 28 -/
theorem intersection_sum (a b c d : ℝ) : 
  (∀ x, -2 * |x - a| + b = 2 * |x - c| + d → x = 10 ∨ x = 18) →
  -2 * |10 - a| + b = 15 →
  -2 * |18 - a| + b = 7 →
  2 * |10 - c| + d = 15 →
  2 * |18 - c| + d = 7 →
  a + c = 28 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l1711_171185


namespace NUMINAMATH_CALUDE_alyssa_puppies_l1711_171189

/-- The number of puppies Alyssa has after breeding and giving some away -/
def remaining_puppies (initial : ℕ) (puppies_per_puppy : ℕ) (given_away : ℕ) : ℕ :=
  initial + initial * puppies_per_puppy - given_away

/-- Theorem stating that Alyssa has 20 puppies left -/
theorem alyssa_puppies : remaining_puppies 7 4 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_puppies_l1711_171189


namespace NUMINAMATH_CALUDE_integer_ratio_problem_l1711_171181

theorem integer_ratio_problem (a b : ℤ) :
  1996 * a + b / 96 = a + b →
  b / a = 2016 ∨ a / b = 1 / 2016 := by
sorry

end NUMINAMATH_CALUDE_integer_ratio_problem_l1711_171181


namespace NUMINAMATH_CALUDE_inscribed_polygon_perimeter_l1711_171187

/-- 
Given two regular polygons with perimeters a and b circumscribed around a circle, 
and a third regular polygon inscribed in the same circle, where the second and 
third polygons have twice as many sides as the first, the perimeter of the third 
polygon is equal to b√(a / (2a - b)).
-/
theorem inscribed_polygon_perimeter 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_circumscribed : ∃ (r : ℝ) (n : ℕ), a = 2 * n * r * Real.tan (2 * π / n) ∧ 
                                        b = 4 * n * r * Real.tan (π / n))
  (h_inscribed : ∃ (x : ℝ), x = b * Real.cos (π / n)) :
  ∃ (x : ℝ), x = b * Real.sqrt (a / (2 * a - b)) :=
sorry

end NUMINAMATH_CALUDE_inscribed_polygon_perimeter_l1711_171187


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l1711_171118

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (4 * x = x^2 - 8) ↔ (x^2 - 4*x - 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l1711_171118


namespace NUMINAMATH_CALUDE_john_spent_110_l1711_171178

/-- The amount of money John spent on wigs for his plays -/
def johnSpent (numPlays : ℕ) (numActs : ℕ) (wigsPerAct : ℕ) (wigCost : ℕ) (sellPrice : ℕ) : ℕ :=
  let totalWigs := numPlays * numActs * wigsPerAct
  let totalCost := totalWigs * wigCost
  let soldWigs := numActs * wigsPerAct
  let moneyBack := soldWigs * sellPrice
  totalCost - moneyBack

/-- Theorem stating that John spent $110 on wigs -/
theorem john_spent_110 :
  johnSpent 3 5 2 5 4 = 110 := by
  sorry

end NUMINAMATH_CALUDE_john_spent_110_l1711_171178


namespace NUMINAMATH_CALUDE_max_students_is_25_l1711_171119

/-- A graph representing friendships in a class of students. -/
structure FriendshipGraph (n : ℕ) where
  friends : Fin n → Fin n → Prop

/-- The property that among any six students, there are two that are not friends. -/
def hasTwoNonFriends (G : FriendshipGraph n) : Prop :=
  ∀ (s : Finset (Fin n)), s.card = 6 → ∃ (i j : Fin n), i ∈ s ∧ j ∈ s ∧ i ≠ j ∧ ¬G.friends i j

/-- The property that for any pair of non-friends, there is a student among the remaining four who is friends with both. -/
def hasCommonFriend (G : FriendshipGraph n) : Prop :=
  ∀ (i j : Fin n), i ≠ j → ¬G.friends i j →
    ∃ (k : Fin n), k ≠ i ∧ k ≠ j ∧ G.friends i k ∧ G.friends j k

/-- The main theorem: The maximum number of students satisfying the given conditions is 25. -/
theorem max_students_is_25 :
  (∃ (n : ℕ) (G : FriendshipGraph n), hasTwoNonFriends G ∧ hasCommonFriend G) ∧
  (∀ (n : ℕ) (G : FriendshipGraph n), hasTwoNonFriends G → hasCommonFriend G → n ≤ 25) :=
sorry

end NUMINAMATH_CALUDE_max_students_is_25_l1711_171119
