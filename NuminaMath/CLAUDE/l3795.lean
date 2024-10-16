import Mathlib

namespace NUMINAMATH_CALUDE_lowest_common_multiple_8_12_l3795_379596

theorem lowest_common_multiple_8_12 : ∃ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 12 ∣ n ∧ ∀ m : ℕ, m > 0 → 8 ∣ m → 12 ∣ m → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_lowest_common_multiple_8_12_l3795_379596


namespace NUMINAMATH_CALUDE_company_employees_l3795_379529

theorem company_employees (total : ℕ) 
  (h1 : (60 : ℚ) / 100 * total = (total : ℚ) - (40 : ℚ) / 100 * total)
  (h2 : (20 : ℚ) / 100 * total = (40 : ℚ) / 100 * total / 2)
  (h3 : (20 : ℚ) / 100 * total = 20) :
  total = 100 := by
sorry

end NUMINAMATH_CALUDE_company_employees_l3795_379529


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3795_379534

theorem intersection_of_lines :
  ∃! (x y : ℚ), (3 * y = -2 * x + 6) ∧ (-2 * y = 4 * x - 3) ∧ (x = 3/8) ∧ (y = 7/4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3795_379534


namespace NUMINAMATH_CALUDE_volleyball_team_combinations_l3795_379564

theorem volleyball_team_combinations (n : ℕ) (k : ℕ) (h1 : n = 14) (h2 : k = 6) :
  Nat.choose n k = 3003 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_combinations_l3795_379564


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3795_379557

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 8. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 2 ∧ b = 2 ∧ c = 4 →  -- Two sides are 2, one side is 4
  a + b > c →              -- Triangle inequality
  a = b →                  -- Isosceles condition
  a + b + c = 8 :=         -- Perimeter is 8
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3795_379557


namespace NUMINAMATH_CALUDE_probability_nine_correct_l3795_379550

/-- The number of English-Russian expression pairs to be matched -/
def total_pairs : ℕ := 10

/-- The number of correctly matched pairs we're interested in -/
def correct_matches : ℕ := 9

/-- Represents the probability of getting exactly 9 out of 10 matches correct when choosing randomly -/
def prob_nine_correct : ℝ := 0

/-- Theorem stating that the probability of getting exactly 9 out of 10 matches correct when choosing randomly is 0 -/
theorem probability_nine_correct :
  prob_nine_correct = 0 := by sorry

end NUMINAMATH_CALUDE_probability_nine_correct_l3795_379550


namespace NUMINAMATH_CALUDE_remaining_note_denomination_l3795_379590

theorem remaining_note_denomination 
  (total_amount : ℕ) 
  (total_notes : ℕ) 
  (fifty_notes : ℕ) 
  (h1 : total_amount = 10350)
  (h2 : total_notes = 108)
  (h3 : fifty_notes = 97) :
  (total_amount - 50 * fifty_notes) / (total_notes - fifty_notes) = 500 := by
sorry


end NUMINAMATH_CALUDE_remaining_note_denomination_l3795_379590


namespace NUMINAMATH_CALUDE_simplify_expression_l3795_379524

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^3 + b^3 = a + b - 1) : 
  a / b + b / a - 2 / (a * b) = -1 - 1 / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3795_379524


namespace NUMINAMATH_CALUDE_two_Z_six_l3795_379569

/-- Definition of the operation Z -/
def Z (a b : ℤ) : ℤ := b + 10 * a - a ^ 2

/-- Theorem stating that 2Z6 = 22 -/
theorem two_Z_six : Z 2 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_two_Z_six_l3795_379569


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3795_379526

-- Define a quadratic function
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

-- Define the inverse function property
def HasInverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, f (g x) = x) ∧ (∀ x, g (f x) = x)

theorem quadratic_function_property (f : ℝ → ℝ) 
  (h1 : QuadraticFunction f)
  (h2 : HasInverse f)
  (h3 : ∀ x, f x = 3 * (Classical.choose h2) x + 5)
  (h4 : f 1 = 5) :
  f 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3795_379526


namespace NUMINAMATH_CALUDE_min_value_expression_l3795_379582

theorem min_value_expression (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  1 / (x + y)^2 + 1 / (x - y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3795_379582


namespace NUMINAMATH_CALUDE_items_not_washed_l3795_379594

theorem items_not_washed (total_items : ℕ) (items_washed : ℕ) : 
  total_items = 129 → items_washed = 20 → total_items - items_washed = 109 := by
  sorry

end NUMINAMATH_CALUDE_items_not_washed_l3795_379594


namespace NUMINAMATH_CALUDE_tan_660_degrees_l3795_379570

theorem tan_660_degrees : Real.tan (660 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_660_degrees_l3795_379570


namespace NUMINAMATH_CALUDE_unique_pair_l3795_379535

/-- A function that returns the last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- A function that checks if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A theorem stating that the only pair of positive integers (a, b) satisfying
    all the given conditions is (9, 4) -/
theorem unique_pair : ∀ a b : ℕ+, 
  (lastDigit (a.val + b.val) = 3) →
  (∃ p : ℕ, Nat.Prime p ∧ a.val - b.val = p) →
  isPerfectSquare (a.val * b.val) →
  (a.val = 9 ∧ b.val = 4) ∨ (a.val = 4 ∧ b.val = 9) := by
  sorry

#check unique_pair

end NUMINAMATH_CALUDE_unique_pair_l3795_379535


namespace NUMINAMATH_CALUDE_count_numbers_with_three_700_l3795_379549

def contains_three (n : Nat) : Bool :=
  n.repr.any (· = '3')

def count_numbers_with_three (upper_bound : Nat) : Nat :=
  (List.range upper_bound).filter contains_three |>.length

theorem count_numbers_with_three_700 :
  count_numbers_with_three 700 = 214 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_three_700_l3795_379549


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3795_379532

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (6 - 5 * z) = 7 :=
by
  -- The unique solution is z = -43/5
  use -43/5
  constructor
  · -- Prove that -43/5 satisfies the equation
    sorry
  · -- Prove that any solution must equal -43/5
    sorry

#check sqrt_equation_solution

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3795_379532


namespace NUMINAMATH_CALUDE_sqrt_pattern_l3795_379595

theorem sqrt_pattern (a b : ℝ) : 
  (∀ n : ℕ, n ≥ 2 → n ≤ 4 → Real.sqrt (n + n / (n^2 - 1)) = n * Real.sqrt (n / (n^2 - 1))) →
  Real.sqrt (6 + a / b) = 6 * Real.sqrt (a / b) →
  a = 6 ∧ b = 35 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l3795_379595


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3795_379559

def A : Set ℝ := {x | x^2 - 2*x - 8 > 0}
def B : Set ℝ := {-3, -1, 1, 3, 5}

theorem intersection_of_A_and_B : A ∩ B = {-3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3795_379559


namespace NUMINAMATH_CALUDE_area_between_curves_l3795_379575

-- Define the functions for the curves
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x^3

-- Define the bounds of integration
def lower_bound : ℝ := 0
def upper_bound : ℝ := 1

-- State the theorem
theorem area_between_curves : 
  (∫ x in lower_bound..upper_bound, f x - g x) = 1/12 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l3795_379575


namespace NUMINAMATH_CALUDE_integral_x_plus_inverse_x_l3795_379583

open Real MeasureTheory

theorem integral_x_plus_inverse_x : ∫ x in (1 : ℝ)..2, (x + 1/x) = 3/2 + Real.log 2 := by sorry

end NUMINAMATH_CALUDE_integral_x_plus_inverse_x_l3795_379583


namespace NUMINAMATH_CALUDE_fraction_cube_multiply_l3795_379515

theorem fraction_cube_multiply (a b : ℚ) : (1 / 3 : ℚ)^3 * (1 / 5 : ℚ) = 1 / 135 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cube_multiply_l3795_379515


namespace NUMINAMATH_CALUDE_smallest_even_abundant_after_12_l3795_379574

def is_abundant (n : ℕ) : Prop :=
  n < (Finset.sum (Finset.filter (λ x => x < n ∧ n % x = 0) (Finset.range n)) id)

theorem smallest_even_abundant_after_12 :
  ∀ n : ℕ, n > 12 → n % 2 = 0 → is_abundant n → n ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_even_abundant_after_12_l3795_379574


namespace NUMINAMATH_CALUDE_convergence_trap_equivalence_l3795_379571

open Set Filter Topology Metric

variable {X : Type*} [MetricSpace X]
variable (x : ℕ → X) (a : X)

def is_trap (s : Set X) (x : ℕ → X) : Prop :=
  ∃ N, ∀ n ≥ N, x n ∈ s

theorem convergence_trap_equivalence :
  (Tendsto x atTop (𝓝 a)) ↔
  (∀ ε > 0, is_trap (ball a ε) x) :=
sorry

end NUMINAMATH_CALUDE_convergence_trap_equivalence_l3795_379571


namespace NUMINAMATH_CALUDE_sequence_problem_l3795_379568

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- An arithmetic sequence -/
def IsArithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The problem statement -/
theorem sequence_problem (a : Sequence) 
  (h_arith : IsArithmetic a)
  (h_geom : IsGeometric (fun n => a (n + 1)))
  (h_a5 : a 5 = 1) :
  a 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3795_379568


namespace NUMINAMATH_CALUDE_window_area_ratio_l3795_379588

theorem window_area_ratio :
  ∀ (ad ab : ℝ),
  ad / ab = 4 / 3 →
  ab = 36 →
  let r := ab / 2
  let rectangle_area := ad * ab
  let semicircles_area := π * r^2
  rectangle_area / semicircles_area = 16 / (3 * π) := by
sorry

end NUMINAMATH_CALUDE_window_area_ratio_l3795_379588


namespace NUMINAMATH_CALUDE_translation_result_l3795_379547

/-- Represents a point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally by a given distance -/
def translate_x (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- The initial point P -/
def P : Point :=
  { x := -2, y := 4 }

/-- The translation distance to the right -/
def translation_distance : ℝ := 1

theorem translation_result :
  translate_x P translation_distance = { x := -1, y := 4 } := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l3795_379547


namespace NUMINAMATH_CALUDE_no_integers_product_sum_20182017_l3795_379521

theorem no_integers_product_sum_20182017 : ¬∃ (a b : ℤ), a * b * (a + b) = 20182017 := by
  sorry

end NUMINAMATH_CALUDE_no_integers_product_sum_20182017_l3795_379521


namespace NUMINAMATH_CALUDE_set_operations_l3795_379507

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}

def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}

def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5/2}

theorem set_operations :
  (A ∩ B = {x | -1 < x ∧ x < 2}) ∧
  ((U \ B) ∪ P = {x | x ≤ 0 ∨ x ≥ 5/2}) ∧
  ((A ∩ B) ∩ (U \ P) = {x | 0 < x ∧ x < 2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3795_379507


namespace NUMINAMATH_CALUDE_expression_value_l3795_379585

theorem expression_value : 
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3795_379585


namespace NUMINAMATH_CALUDE_surrounding_circles_radius_l3795_379562

/-- The radius of the central circle -/
def central_radius : ℝ := 2

/-- The number of surrounding circles -/
def num_surrounding_circles : ℕ := 4

/-- Predicate that checks if all circles are touching each other -/
def circles_touching (r : ℝ) : Prop :=
  ∃ (centers : Fin num_surrounding_circles → ℝ × ℝ),
    ∀ (i j : Fin num_surrounding_circles),
      i ≠ j → ‖centers i - centers j‖ = 2 * r ∧
    ∀ (i : Fin num_surrounding_circles),
      ‖centers i‖ = central_radius + r

/-- Theorem stating that the radius of surrounding circles is 2 -/
theorem surrounding_circles_radius :
  ∃ (r : ℝ), r > 0 ∧ circles_touching r → r = 2 :=
sorry

end NUMINAMATH_CALUDE_surrounding_circles_radius_l3795_379562


namespace NUMINAMATH_CALUDE_no_three_distinct_reals_l3795_379592

theorem no_three_distinct_reals : ¬∃ (a b c p : ℝ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b * c = p ∧
  b + c * a = p ∧
  c + a * b = p := by
  sorry

end NUMINAMATH_CALUDE_no_three_distinct_reals_l3795_379592


namespace NUMINAMATH_CALUDE_waiter_customers_l3795_379541

theorem waiter_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) 
  (h1 : num_tables = 7)
  (h2 : women_per_table = 7)
  (h3 : men_per_table = 2) :
  num_tables * (women_per_table + men_per_table) = 63 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l3795_379541


namespace NUMINAMATH_CALUDE_x_coordinate_difference_at_y_20_l3795_379519

/-- A line in a 2D coordinate system --/
structure Line where
  slope : ℚ
  y_intercept : ℚ

def Line.through_points (x1 y1 x2 y2 : ℚ) : Line where
  slope := (y2 - y1) / (x2 - x1)
  y_intercept := y1 - ((y2 - y1) / (x2 - x1)) * x1

def Line.x_at_y (l : Line) (y : ℚ) : ℚ :=
  (y - l.y_intercept) / l.slope

theorem x_coordinate_difference_at_y_20 :
  let l := Line.through_points 0 6 3 0
  let m := Line.through_points 0 3 8 0
  let x_l := l.x_at_y 20
  let x_m := m.x_at_y 20
  |x_l - x_m| = 115 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_difference_at_y_20_l3795_379519


namespace NUMINAMATH_CALUDE_dividend_rate_is_14_percent_l3795_379558

/-- Calculates the rate of dividend given investment details and annual income -/
def rate_of_dividend (total_investment : ℚ) (share_face_value : ℚ) (share_quoted_price : ℚ) (annual_income : ℚ) : ℚ :=
  let number_of_shares := total_investment / share_quoted_price
  let dividend_per_share := annual_income / number_of_shares
  (dividend_per_share / share_face_value) * 100

/-- Theorem stating that given the specific investment details and annual income, the rate of dividend is 14% -/
theorem dividend_rate_is_14_percent :
  rate_of_dividend 4940 10 9.5 728 = 14 := by
  sorry

end NUMINAMATH_CALUDE_dividend_rate_is_14_percent_l3795_379558


namespace NUMINAMATH_CALUDE_cosine_vertical_shift_l3795_379586

theorem cosine_vertical_shift 
  (a b c d : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0) 
  (hmax : d + a = 7) 
  (hmin : d - a = 1) : 
  d = 4 := by
sorry

end NUMINAMATH_CALUDE_cosine_vertical_shift_l3795_379586


namespace NUMINAMATH_CALUDE_min_value_problem1_l3795_379522

theorem min_value_problem1 (x : ℝ) (h : x > 3) : 4 / (x - 3) + x ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem1_l3795_379522


namespace NUMINAMATH_CALUDE_larger_integer_problem_l3795_379553

theorem larger_integer_problem :
  ∃ (x : ℕ+) (y : ℕ+), (4 * x)^2 - 2 * x = 8100 ∧ x + 10 = 2 * y ∧ x = 22 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l3795_379553


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l3795_379527

theorem cow_chicken_problem (c h : ℕ) : 
  4 * c + 2 * h = 2 * (c + h) + 20 → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l3795_379527


namespace NUMINAMATH_CALUDE_expression_approximation_l3795_379545

theorem expression_approximation :
  let x := ((69.28 * 0.004)^3 * Real.sin (Real.pi/3)) / (0.03^2 * Real.log 0.58 * Real.cos (Real.pi/4))
  ∃ ε > 0, |x + 37.644| < ε ∧ ε < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_expression_approximation_l3795_379545


namespace NUMINAMATH_CALUDE_cab_driver_income_theorem_l3795_379512

/-- Represents the weather condition for a day --/
inductive Weather
  | Sunny
  | Rainy
  | Cloudy

/-- Represents a day's income data --/
structure DayData where
  income : ℝ
  weather : Weather
  isPeakHours : Bool

/-- Calculates the adjusted income for a day based on weather and peak hours --/
def adjustedIncome (day : DayData) : ℝ :=
  match day.weather with
  | Weather.Rainy => day.income * 1.1
  | Weather.Cloudy => day.income * 0.95
  | Weather.Sunny => 
    if day.isPeakHours then day.income * 1.2
    else day.income

/-- The income data for 12 days --/
def incomeData : List DayData := [
  ⟨200, Weather.Rainy, false⟩,
  ⟨150, Weather.Sunny, false⟩,
  ⟨750, Weather.Sunny, false⟩,
  ⟨400, Weather.Sunny, false⟩,
  ⟨500, Weather.Cloudy, false⟩,
  ⟨300, Weather.Rainy, false⟩,
  ⟨650, Weather.Sunny, false⟩,
  ⟨350, Weather.Cloudy, false⟩,
  ⟨600, Weather.Sunny, true⟩,
  ⟨450, Weather.Sunny, false⟩,
  ⟨530, Weather.Sunny, false⟩,
  ⟨480, Weather.Cloudy, false⟩
]

theorem cab_driver_income_theorem :
  let totalIncome := (incomeData.map adjustedIncome).sum
  let averageIncome := totalIncome / incomeData.length
  totalIncome = 4963.5 ∧ averageIncome = 413.625 := by
  sorry


end NUMINAMATH_CALUDE_cab_driver_income_theorem_l3795_379512


namespace NUMINAMATH_CALUDE_odd_function_property_l3795_379554

-- Define an odd function on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f 2 = 2) :
  f (-2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3795_379554


namespace NUMINAMATH_CALUDE_triangle_equilateral_l3795_379584

/-- A triangle with sides a, b, c corresponding to angles A, B, C is equilateral if
    a * cos(C) = c * cos(A) and a, b, c are in geometric progression. -/
theorem triangle_equilateral (a b c : ℝ) (A B C : Real) :
  a > 0 → b > 0 → c > 0 →
  a * Real.cos C = c * Real.cos A →
  ∃ r : ℝ, r > 0 ∧ a = b / r ∧ b = c / r →
  a = b ∧ b = c := by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l3795_379584


namespace NUMINAMATH_CALUDE_second_week_rainfall_l3795_379563

/-- Rainfall during the first two weeks of January in Springdale -/
def total_rainfall : ℝ := 20

/-- Ratio of second week's rainfall to first week's rainfall -/
def rainfall_ratio : ℝ := 1.5

/-- Theorem: The rainfall during the second week was 12 inches -/
theorem second_week_rainfall : 
  ∃ (first_week second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = rainfall_ratio * first_week ∧
    second_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_week_rainfall_l3795_379563


namespace NUMINAMATH_CALUDE_cube_intersection_length_l3795_379546

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with edge length a -/
structure Cube (a : ℝ) where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E : Point3D
  F : Point3D
  G : Point3D
  H : Point3D

/-- The theorem to be proved -/
theorem cube_intersection_length (a : ℝ) (cube : Cube a) 
  (M : Point3D) (N : Point3D) (P : Point3D) (T : Point3D)
  (h_a : a > 0)
  (h_M : M.x = a ∧ M.y = a ∧ M.z = a/2)
  (h_N : N.x = a ∧ N.y = a/3 ∧ N.z = a)
  (h_P : P.x = 0 ∧ P.y = 0 ∧ P.z = 3*a/4)
  (h_T : T.x = 0 ∧ T.y = a ∧ 0 ≤ T.z ∧ T.z ≤ a)
  (h_plane : ∃ (k : ℝ), k * (M.x - P.x) * (N.y - P.y) * (T.z - P.z) = 
                         k * (N.x - P.x) * (M.y - P.y) * (T.z - P.z) + 
                         k * (T.x - P.x) * (M.y - P.y) * (N.z - P.z)) :
  ∃ (DT : ℝ), DT = 5*a/6 ∧ DT = Real.sqrt ((T.x - cube.D.x)^2 + (T.y - cube.D.y)^2 + (T.z - cube.D.z)^2) :=
sorry

end NUMINAMATH_CALUDE_cube_intersection_length_l3795_379546


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3795_379514

theorem completing_square_equivalence (x : ℝ) :
  x^2 - 6*x + 1 = 0 ↔ (x - 3)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3795_379514


namespace NUMINAMATH_CALUDE_exponential_system_solution_l3795_379518

theorem exponential_system_solution (x y : ℝ) : 
  (4 : ℝ)^x = 256^(y + 1) → (27 : ℝ)^y = 3^(x - 2) → x = -4 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_system_solution_l3795_379518


namespace NUMINAMATH_CALUDE_clock_malfunction_l3795_379517

/-- Represents a time in HH:MM format -/
structure Time where
  hours : Nat
  minutes : Nat
  hh_valid : hours < 24
  mm_valid : minutes < 60

/-- Represents a malfunctioning clock where each digit either increases or decreases by 1 -/
def is_malfunctioned (original : Time) (displayed : Time) : Prop :=
  (displayed.hours / 10 = original.hours / 10 + 1 ∨ displayed.hours / 10 = original.hours / 10 - 1) ∧
  (displayed.hours % 10 = (original.hours % 10 + 1) % 10 ∨ displayed.hours % 10 = (original.hours % 10 - 1 + 10) % 10) ∧
  (displayed.minutes / 10 = original.minutes / 10 + 1 ∨ displayed.minutes / 10 = original.minutes / 10 - 1) ∧
  (displayed.minutes % 10 = (original.minutes % 10 + 1) % 10 ∨ displayed.minutes % 10 = (original.minutes % 10 - 1 + 10) % 10)

theorem clock_malfunction (displayed : Time) (h_displayed : displayed.hours = 20 ∧ displayed.minutes = 9) :
  ∃ (original : Time), is_malfunctioned original displayed ∧ original.hours = 11 ∧ original.minutes = 18 := by
  sorry

end NUMINAMATH_CALUDE_clock_malfunction_l3795_379517


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l3795_379556

/-- Proves that the initial alcohol percentage is 25% given the problem conditions -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (final_percentage : ℝ)
  (h_initial_volume : initial_volume = 6)
  (h_added_alcohol : added_alcohol = 3)
  (h_final_percentage : final_percentage = 50)
  (h_alcohol_balance : initial_volume * (initial_percentage / 100) + added_alcohol = 
                       (initial_volume + added_alcohol) * (final_percentage / 100)) :
  initial_percentage = 25 :=
by
  sorry

#check initial_alcohol_percentage

end NUMINAMATH_CALUDE_initial_alcohol_percentage_l3795_379556


namespace NUMINAMATH_CALUDE_sqrt_224_range_l3795_379538

theorem sqrt_224_range : 14 < Real.sqrt 224 ∧ Real.sqrt 224 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_224_range_l3795_379538


namespace NUMINAMATH_CALUDE_expression_result_l3795_379523

theorem expression_result : 
  (0.66 : ℝ)^3 - (0.1 : ℝ)^3 / (0.66 : ℝ)^2 + 0.066 + (0.1 : ℝ)^2 = 0.3612 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l3795_379523


namespace NUMINAMATH_CALUDE_reflect_center_l3795_379505

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflect_center :
  let original_center : ℝ × ℝ := (8, -3)
  let reflected_center : ℝ × ℝ := reflect_about_y_eq_neg_x original_center
  reflected_center = (3, -8) := by sorry

end NUMINAMATH_CALUDE_reflect_center_l3795_379505


namespace NUMINAMATH_CALUDE_vector_magnitude_l3795_379580

theorem vector_magnitude (a b : ℝ × ℝ) :
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  a - b = (Real.sqrt 3, Real.sqrt 2) →
  ‖a + 2 • b‖ = Real.sqrt 17 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3795_379580


namespace NUMINAMATH_CALUDE_juice_theorem_l3795_379510

def juice_problem (tom_initial jerry_initial : ℚ) 
  (drink_fraction transfer_fraction : ℚ) (final_transfer : ℚ) : Prop :=
  let tom_after_drinking := tom_initial * (1 - drink_fraction)
  let jerry_after_drinking := jerry_initial * (1 - drink_fraction)
  let jerry_transfer := jerry_after_drinking * transfer_fraction
  let tom_before_final := tom_after_drinking + jerry_transfer
  let jerry_before_final := jerry_after_drinking - jerry_transfer
  let tom_final := tom_before_final - final_transfer
  let jerry_final := jerry_before_final + final_transfer
  (jerry_initial = 2 * tom_initial) ∧
  (tom_final = jerry_final + 4) ∧
  (tom_initial + jerry_initial - (tom_final + jerry_final) = 80)

theorem juice_theorem : 
  juice_problem 40 80 (2/3) (1/4) 5 := by sorry

end NUMINAMATH_CALUDE_juice_theorem_l3795_379510


namespace NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l3795_379536

/-- Curve C defined by the equation x²/a + y²/b = 1 -/
structure CurveC (a b : ℝ) where
  equation : ∀ (x y : ℝ), x^2 / a + y^2 / b = 1

/-- Predicate for C being an ellipse with foci on the x-axis -/
def is_ellipse_x_foci (a b : ℝ) : Prop :=
  ∃ (c : ℝ), a > b ∧ b > 0 ∧ c^2 = a^2 - b^2

/-- Main theorem: "a > b" is necessary but not sufficient for C to be an ellipse with foci on x-axis -/
theorem a_gt_b_necessary_not_sufficient (a b : ℝ) :
  (is_ellipse_x_foci a b → a > b) ∧
  ¬(a > b → is_ellipse_x_foci a b) :=
sorry

end NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l3795_379536


namespace NUMINAMATH_CALUDE_cos_theta_equals_sqrt2_over_2_l3795_379539

/-- Given vectors a and b with an angle θ between them, 
    if a = (1,1) and b - a = (-1,1), then cos θ = √2/2 -/
theorem cos_theta_equals_sqrt2_over_2 (a b : ℝ × ℝ) (θ : ℝ) :
  a = (1, 1) →
  b - a = (-1, 1) →
  let cos_theta := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  cos_theta = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_theta_equals_sqrt2_over_2_l3795_379539


namespace NUMINAMATH_CALUDE_inequality_solution_l3795_379593

theorem inequality_solution (x : ℝ) : 
  (x^2 - 6*x + 8) / (x^2 - 9) > 0 ↔ x < -3 ∨ (2 < x ∧ x < 3) ∨ x > 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3795_379593


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l3795_379548

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℝ) 
  (error1 error2 error3 error4 error5 : ℝ) : 
  n = 70 →
  initial_mean = 350 →
  error1 = 215.5 - 195.5 →
  error2 = -30 - 30 →
  error3 = 720.8 - 670.8 →
  error4 = -95.4 - (-45.4) →
  error5 = 124.2 - 114.2 →
  (n : ℝ) * initial_mean + (error1 + error2 + error3 + error4 + error5) = n * 349.57 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l3795_379548


namespace NUMINAMATH_CALUDE_triangle_preserving_characterization_l3795_379509

/-- A function satisfying the triangle property -/
def TrianglePreserving (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c →
    (a + b > c ∧ b + c > a ∧ c + a > b ↔ f a + f b > f c ∧ f b + f c > f a ∧ f c + f a > f b)

/-- Main theorem: Characterization of triangle-preserving functions -/
theorem triangle_preserving_characterization (f : ℝ → ℝ) 
    (h₁ : ∀ x, 0 < x → 0 < f x) 
    (h₂ : TrianglePreserving f) :
    ∃ c : ℝ, c > 0 ∧ ∀ x, 0 < x → f x = c * x :=
  sorry

end NUMINAMATH_CALUDE_triangle_preserving_characterization_l3795_379509


namespace NUMINAMATH_CALUDE_total_interest_percentage_l3795_379598

def total_investment : ℝ := 100000
def interest_rate_1 : ℝ := 0.09
def interest_rate_2 : ℝ := 0.11
def amount_at_rate_2 : ℝ := 24999.999999999996

def amount_at_rate_1 : ℝ := total_investment - amount_at_rate_2

def interest_1 : ℝ := amount_at_rate_1 * interest_rate_1
def interest_2 : ℝ := amount_at_rate_2 * interest_rate_2

def total_interest : ℝ := interest_1 + interest_2

theorem total_interest_percentage : 
  (total_interest / total_investment) * 100 = 9.5 := by sorry

end NUMINAMATH_CALUDE_total_interest_percentage_l3795_379598


namespace NUMINAMATH_CALUDE_test_mean_score_l3795_379587

theorem test_mean_score (mean : ℝ) (std_dev : ℝ) (lowest_score : ℝ) : 
  std_dev = 10 →
  lowest_score = mean - 2 * std_dev →
  lowest_score = 20 →
  mean = 40 := by
sorry

end NUMINAMATH_CALUDE_test_mean_score_l3795_379587


namespace NUMINAMATH_CALUDE_speed_increase_ratio_l3795_379579

theorem speed_increase_ratio (v : ℝ) (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end NUMINAMATH_CALUDE_speed_increase_ratio_l3795_379579


namespace NUMINAMATH_CALUDE_rectangle_width_equals_circle_area_l3795_379561

theorem rectangle_width_equals_circle_area (r : ℝ) (l w : ℝ) : 
  r = Real.sqrt 12 → 
  l = 3 * Real.sqrt 2 → 
  π * r^2 = l * w → 
  w = 2 * Real.sqrt 2 * π := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_equals_circle_area_l3795_379561


namespace NUMINAMATH_CALUDE_prob_two_twos_given_sum7_is_one_fourth_l3795_379573

/-- Represents the outcome of three draws from an urn containing balls numbered 1 to 4 -/
structure ThreeDraws where
  first : Fin 4
  second : Fin 4
  third : Fin 4

/-- The set of all possible outcomes of three draws -/
def allOutcomes : Finset ThreeDraws := sorry

/-- The probability of each individual outcome, assuming uniform distribution -/
def probOfOutcome (outcome : ThreeDraws) : ℚ := 1 / 64

/-- The sum of the numbers drawn in a given outcome -/
def sumOfDraws (outcome : ThreeDraws) : Nat :=
  outcome.first.val + 1 + outcome.second.val + 1 + outcome.third.val + 1

/-- The set of outcomes where the sum of draws is 7 -/
def outcomesWithSum7 : Finset ThreeDraws :=
  allOutcomes.filter (λ o => sumOfDraws o = 7)

/-- The number of times 2 is drawn in a given outcome -/
def countTwos (outcome : ThreeDraws) : Nat :=
  (if outcome.first = 1 then 1 else 0) +
  (if outcome.second = 1 then 1 else 0) +
  (if outcome.third = 1 then 1 else 0)

/-- The set of outcomes where 2 is drawn at least twice and the sum is 7 -/
def outcomesWithTwoTwosAndSum7 : Finset ThreeDraws :=
  outcomesWithSum7.filter (λ o => countTwos o ≥ 2)

/-- The probability of drawing 2 at least twice given that the sum is 7 -/
def probTwoTwosGivenSum7 : ℚ :=
  (outcomesWithTwoTwosAndSum7.sum probOfOutcome) /
  (outcomesWithSum7.sum probOfOutcome)

theorem prob_two_twos_given_sum7_is_one_fourth :
  probTwoTwosGivenSum7 = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_prob_two_twos_given_sum7_is_one_fourth_l3795_379573


namespace NUMINAMATH_CALUDE_cookies_eaten_by_adults_l3795_379520

/-- Proves that the number of cookies eaten by adults is 40 --/
theorem cookies_eaten_by_adults (total_cookies : ℕ) (num_children : ℕ) (child_cookies : ℕ) : 
  total_cookies = 120 →
  num_children = 4 →
  child_cookies = 20 →
  (total_cookies - num_children * child_cookies : ℚ) = (1/3 : ℚ) * total_cookies :=
by
  sorry

#check cookies_eaten_by_adults

end NUMINAMATH_CALUDE_cookies_eaten_by_adults_l3795_379520


namespace NUMINAMATH_CALUDE_work_completion_time_l3795_379591

theorem work_completion_time 
  (work_rate_b : ℝ) 
  (work_rate_combined : ℝ) 
  (days_b : ℝ) 
  (days_combined : ℝ) :
  work_rate_b = 1 / days_b →
  work_rate_combined = 1 / days_combined →
  days_b = 6 →
  days_combined = 3.75 →
  work_rate_combined = work_rate_b + 1 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3795_379591


namespace NUMINAMATH_CALUDE_pet_shop_dogs_l3795_379533

/-- Given a pet shop with dogs, cats, and bunnies, where the ratio of dogs to cats to bunnies
    is 3:7:12 and the total number of dogs and bunnies is 375, prove that there are 75 dogs. -/
theorem pet_shop_dogs (dogs cats bunnies : ℕ) : 
  dogs + cats + bunnies > 0 →
  dogs * 7 = cats * 3 →
  dogs * 12 = bunnies * 3 →
  dogs + bunnies = 375 →
  dogs = 75 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_dogs_l3795_379533


namespace NUMINAMATH_CALUDE_chocolate_bars_in_box_l3795_379531

/-- The weight of a single chocolate bar in grams -/
def bar_weight : ℕ := 125

/-- The weight of the box in kilograms -/
def box_weight : ℕ := 2

/-- The number of chocolate bars in the box -/
def num_bars : ℕ := (box_weight * 1000) / bar_weight

/-- Theorem stating that the number of chocolate bars in the box is 16 -/
theorem chocolate_bars_in_box : num_bars = 16 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_in_box_l3795_379531


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l3795_379503

/-- Represents a rectangular plot with specific properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ
  area_eq : area = 21 * breadth
  length_eq : length = breadth + 10

/-- Theorem stating that a rectangular plot with the given properties has a breadth of 11 meters -/
theorem rectangular_plot_breadth (plot : RectangularPlot) : plot.breadth = 11 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l3795_379503


namespace NUMINAMATH_CALUDE_wall_bricks_l3795_379501

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 360

/-- Represents Brenda's time to build the wall alone (in hours) -/
def brenda_time : ℕ := 8

/-- Represents Brandon's time to build the wall alone (in hours) -/
def brandon_time : ℕ := 12

/-- Represents the decrease in combined output (in bricks per hour) -/
def output_decrease : ℕ := 15

/-- Represents the time taken to build the wall together (in hours) -/
def combined_time : ℕ := 6

/-- Theorem stating that the number of bricks in the wall is 360 -/
theorem wall_bricks : 
  (combined_time : ℚ) * ((total_bricks / brenda_time + total_bricks / brandon_time) - output_decrease) = total_bricks := by
  sorry

#check wall_bricks

end NUMINAMATH_CALUDE_wall_bricks_l3795_379501


namespace NUMINAMATH_CALUDE_hannahs_tshirts_l3795_379508

theorem hannahs_tshirts (sweatshirt_count : ℕ) (sweatshirt_price : ℕ) (tshirt_price : ℕ) (total_spent : ℕ) :
  sweatshirt_count = 3 →
  sweatshirt_price = 15 →
  tshirt_price = 10 →
  total_spent = 65 →
  (total_spent - sweatshirt_count * sweatshirt_price) / tshirt_price = 2 := by
sorry

end NUMINAMATH_CALUDE_hannahs_tshirts_l3795_379508


namespace NUMINAMATH_CALUDE_towels_to_wash_l3795_379597

/-- Represents the number of guests entering the gym each hour -/
def guests_per_hour : List ℕ := [40, 48, 60, 80, 68, 68, 48, 24]

/-- The number of towels used by staff -/
def staff_towels : ℕ := 20

/-- Calculates the total number of guests -/
def total_guests : ℕ := guests_per_hour.sum

/-- Calculates the number of towels used by guests -/
def guest_towels (total : ℕ) : ℕ :=
  (total * 10 / 100 * 3) + (total * 60 / 100 * 2) + (total * 30 / 100 * 1)

/-- The main theorem stating the total number of towels to be washed -/
theorem towels_to_wash : 
  guest_towels total_guests + staff_towels = 807 := by
  sorry

end NUMINAMATH_CALUDE_towels_to_wash_l3795_379597


namespace NUMINAMATH_CALUDE_increasing_linear_function_k_range_l3795_379560

theorem increasing_linear_function_k_range (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((k + 2) * x₁ + 1) < ((k + 2) * x₂ + 1)) →
  k > -2 :=
by sorry

end NUMINAMATH_CALUDE_increasing_linear_function_k_range_l3795_379560


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_of_2_1_l3795_379578

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The operation of finding the symmetric point with respect to the y-axis -/
def symmetricPointYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Theorem stating that the symmetric point of (2,1) with respect to the y-axis is (-2,1) -/
theorem symmetric_point_y_axis_of_2_1 :
  let P : Point := { x := 2, y := 1 }
  symmetricPointYAxis P = { x := -2, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_of_2_1_l3795_379578


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l3795_379502

/-- A hyperbola with equation x^2 + (k-1)y^2 = k+1 and foci on the x-axis -/
structure Hyperbola (k : ℝ) where
  eq : ∀ (x y : ℝ), x^2 + (k-1)*y^2 = k+1
  foci_on_x : True  -- This is a placeholder for the foci condition

/-- The range of k values for which the hyperbola is well-defined -/
def valid_k_range : Set ℝ := {k | ∃ h : Hyperbola k, True}

theorem hyperbola_k_range :
  valid_k_range = Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l3795_379502


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tangent_sum_l3795_379542

theorem arithmetic_sequence_tangent_sum (x y z : Real) 
  (h1 : y - x = π/3) 
  (h2 : z - y = π/3) : 
  Real.tan x * Real.tan y + Real.tan y * Real.tan z + Real.tan z * Real.tan x = -3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tangent_sum_l3795_379542


namespace NUMINAMATH_CALUDE_millet_majority_day_four_l3795_379511

/-- Amount of millet in the feeder on day n -/
def millet_amount (n : ℕ) : ℝ :=
  if n = 0 then 0
  else 0.4 + 0.7 * millet_amount (n - 1)

/-- Total amount of seeds in the feeder on day n -/
def total_seeds (n : ℕ) : ℝ := 1

theorem millet_majority_day_four :
  (∀ k < 4, millet_amount k ≤ 0.5) ∧ millet_amount 4 > 0.5 := by sorry

end NUMINAMATH_CALUDE_millet_majority_day_four_l3795_379511


namespace NUMINAMATH_CALUDE_mismatching_socks_count_l3795_379572

def total_socks : ℕ := 65
def ankle_sock_pairs : ℕ := 13
def crew_sock_pairs : ℕ := 10

theorem mismatching_socks_count :
  total_socks - 2 * (ankle_sock_pairs + crew_sock_pairs) = 19 :=
by sorry

end NUMINAMATH_CALUDE_mismatching_socks_count_l3795_379572


namespace NUMINAMATH_CALUDE_quadratic_intersects_twice_iff_k_condition_l3795_379551

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The discriminant of a quadratic function ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

/-- Predicate for a quadratic function intersecting x-axis at two points -/
def intersects_twice (a b c : ℝ) : Prop :=
  discriminant a b c > 0 ∧ a ≠ 0

theorem quadratic_intersects_twice_iff_k_condition (k : ℝ) :
  intersects_twice (k - 2) (-(2 * k - 1)) k ↔ k > -1/4 ∧ k ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersects_twice_iff_k_condition_l3795_379551


namespace NUMINAMATH_CALUDE_f_properties_l3795_379500

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem stating the properties of f and the inequality
theorem f_properties :
  (∃ (t : ℝ), t = 3 ∧ ∀ x, f x ≤ t) ∧
  (∀ x, x ≥ 2 → f x = 3) ∧
  (∀ a b : ℝ, a^2 + 2*b = 1 → 2*a^2 + b^2 ≥ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3795_379500


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_16_l3795_379544

theorem sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 ∨ Real.sqrt (Real.sqrt 16) = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_16_l3795_379544


namespace NUMINAMATH_CALUDE_train_length_l3795_379555

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 36 * (5/18) → time = 40 → speed * time = 400 := by sorry

end NUMINAMATH_CALUDE_train_length_l3795_379555


namespace NUMINAMATH_CALUDE_cube_edge_length_l3795_379566

/-- Given a cube with volume V, surface area S, and edge length a, 
    where V = S + 1, prove that a satisfies a³ - 6a² - 1 = 0 
    and the solution is closest to 6 -/
theorem cube_edge_length (V S a : ℝ) (hV : V = a^3) (hS : S = 6*a^2) (hVS : V = S + 1) :
  a^3 - 6*a^2 - 1 = 0 ∧ ∃ ε > 0, ∀ x : ℝ, x ≠ a → |x - 6| > |a - 6| - ε :=
sorry

end NUMINAMATH_CALUDE_cube_edge_length_l3795_379566


namespace NUMINAMATH_CALUDE_total_notes_l3795_379513

/-- Calculates the total number of notes on a communal board -/
theorem total_notes (red_rows : Nat) (red_per_row : Nat) (blue_per_red : Nat) (extra_blue : Nat) :
  red_rows = 5 →
  red_per_row = 6 →
  blue_per_red = 2 →
  extra_blue = 10 →
  red_rows * red_per_row + red_rows * red_per_row * blue_per_red + extra_blue = 100 := by
  sorry


end NUMINAMATH_CALUDE_total_notes_l3795_379513


namespace NUMINAMATH_CALUDE_every_algorithm_relies_on_sequential_structure_l3795_379599

/-- Represents the basic structures used in algorithms -/
inductive AlgorithmStructure
  | Logical
  | Conditional
  | Loop
  | Sequential

/-- Represents an algorithm with its characteristics -/
structure Algorithm where
  input : Nat
  output : Nat
  steps : List AlgorithmStructure
  isDefinite : Bool
  isFinite : Bool
  isEffective : Bool

/-- Theorem stating that every algorithm relies on the Sequential structure -/
theorem every_algorithm_relies_on_sequential_structure (a : Algorithm) :
  AlgorithmStructure.Sequential ∈ a.steps :=
sorry

end NUMINAMATH_CALUDE_every_algorithm_relies_on_sequential_structure_l3795_379599


namespace NUMINAMATH_CALUDE_intersection_sum_l3795_379525

/-- Given two lines y = 2x + c and y = -x + d intersecting at (4, 12), prove that c + d = 20 -/
theorem intersection_sum (c d : ℝ) : 
  (∀ x y, y = 2*x + c → y = -x + d → (x = 4 ∧ y = 12)) → 
  c + d = 20 := by sorry

end NUMINAMATH_CALUDE_intersection_sum_l3795_379525


namespace NUMINAMATH_CALUDE_intercept_sum_l3795_379530

/-- The modulus of the congruence -/
def m : ℕ := 17

/-- The congruence relation -/
def congruence (x y : ℕ) : Prop :=
  (7 * x) % m = (3 * y + 2) % m

/-- The x-intercept of the congruence -/
def x_intercept : ℕ := 10

/-- The y-intercept of the congruence -/
def y_intercept : ℕ := 5

/-- Theorem stating that the sum of x and y intercepts is 15 -/
theorem intercept_sum :
  x_intercept + y_intercept = 15 ∧
  congruence x_intercept 0 ∧
  congruence 0 y_intercept ∧
  x_intercept < m ∧
  y_intercept < m :=
sorry

end NUMINAMATH_CALUDE_intercept_sum_l3795_379530


namespace NUMINAMATH_CALUDE_quadratic_equation_in_y_l3795_379581

theorem quadratic_equation_in_y : 
  ∀ x y : ℝ, 
  (3 * x^2 - 4 * x + 7 * y + 3 = 0) → 
  (3 * x - 5 * y + 6 = 0) → 
  (25 * y^2 - 39 * y + 69 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_in_y_l3795_379581


namespace NUMINAMATH_CALUDE_circle_properties_l3795_379516

/-- A circle with center on the line y = -4x and tangent to x + y - 1 = 0 at (3, -2) -/
def special_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 4)^2 = 8

/-- The line y = -4x -/
def center_line (x y : ℝ) : Prop := y = -4 * x

/-- The line x + y - 1 = 0 -/
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

/-- The point P(3, -2) -/
def point_P : ℝ × ℝ := (3, -2)

theorem circle_properties :
  ∃ (cx cy : ℝ),
    center_line cx cy ∧
    special_circle cx cy ∧
    tangent_line (point_P.1) (point_P.2) ∧
    (∀ (x y : ℝ), tangent_line x y → ((x - cx)^2 + (y - cy)^2 ≥ 8)) ∧
    ((point_P.1 - cx)^2 + (point_P.2 - cy)^2 = 8) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3795_379516


namespace NUMINAMATH_CALUDE_product_of_specific_numbers_l3795_379540

theorem product_of_specific_numbers : 469158 * 9999 = 4690872842 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_numbers_l3795_379540


namespace NUMINAMATH_CALUDE_undefined_fraction_l3795_379543

theorem undefined_fraction (a b : ℝ) (h1 : a = 4) (h2 : b = -4) :
  ¬∃x : ℝ, x = 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_undefined_fraction_l3795_379543


namespace NUMINAMATH_CALUDE_simplify_expression_l3795_379565

theorem simplify_expression (x y z : ℚ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  18 * x^3 * y^2 * z^2 / (9 * x^2 * y * z^3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3795_379565


namespace NUMINAMATH_CALUDE_smallest_possible_a_l3795_379576

theorem smallest_possible_a (a b : ℤ) (x : ℝ) (h1 : a > x) (h2 : a < 41)
  (h3 : b > 39) (h4 : b < 51)
  (h5 : (↑40 / ↑40 : ℚ) - (↑a / ↑50 : ℚ) = 2/5) : a ≥ 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l3795_379576


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l3795_379528

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_bounds : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number we want to express in scientific notation -/
def target_number : ℝ := 318000000

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 3.18
    exponent := 8
    coeff_bounds := by sorry }

/-- Theorem stating that the proposed notation correctly represents the target number -/
theorem scientific_notation_correct :
  target_number = proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l3795_379528


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3795_379589

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α β : Plane) 
  (h1 : parallel m n) 
  (h2 : plane_perpendicular α β) 
  (h3 : perpendicular m α) : 
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3795_379589


namespace NUMINAMATH_CALUDE_equation_solution_system_solution_l3795_379506

-- Equation 1
theorem equation_solution (x : ℚ) : 
  (3 * x + 1) / 5 = 1 - (4 * x + 3) / 2 ↔ x = -7 / 26 := by sorry

-- System of equations
theorem system_solution (x y : ℚ) : 
  (3 * x - 4 * y = 14 ∧ 5 * x + 4 * y = 2) ↔ (x = 2 ∧ y = -2) := by sorry

end NUMINAMATH_CALUDE_equation_solution_system_solution_l3795_379506


namespace NUMINAMATH_CALUDE_system_solution_l3795_379552

theorem system_solution :
  ∃ (x y : ℤ), 
    (x + 9773 = 13200) ∧
    (2 * x - 3 * y = 1544) ∧
    (x = 3427) ∧
    (y = 1770) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3795_379552


namespace NUMINAMATH_CALUDE_triangle_parallelogram_altitude_l3795_379537

theorem triangle_parallelogram_altitude (base : ℝ) (triangle_altitude parallelogram_altitude : ℝ) :
  base > 0 →
  parallelogram_altitude > 0 →
  parallelogram_altitude = 100 →
  (1 / 2 * base * triangle_altitude) = (base * parallelogram_altitude) →
  triangle_altitude = 200 := by
  sorry

end NUMINAMATH_CALUDE_triangle_parallelogram_altitude_l3795_379537


namespace NUMINAMATH_CALUDE_janous_inequality_l3795_379567

theorem janous_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_janous_inequality_l3795_379567


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l3795_379504

open Real

theorem perpendicular_tangents_intersection (a : ℝ) :
  ∃ x ∈ Set.Ioo 0 (π/2),
    (2 * sin x = a * cos x) ∧
    (2 * cos x) * (-a * sin x) = -1 →
  a = 2 * sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l3795_379504


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l3795_379577

/-- If the terminal side of angle α passes through point (-2, 4), then sin α = (2√5) / 5 -/
theorem sin_alpha_for_point (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * (Real.cos α) = -2 ∧ r * (Real.sin α) = 4) →
  Real.sin α = (2 * Real.sqrt 5) / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l3795_379577
