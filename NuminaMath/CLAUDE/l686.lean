import Mathlib

namespace NUMINAMATH_CALUDE_average_transformation_l686_68673

theorem average_transformation (x₁ x₂ x₃ x₄ : ℝ) 
  (h : (x₁ + x₂ + x₃ + x₄) / 4 = 3) :
  ((2*x₁ - 3) + (2*x₂ - 3) + (2*x₃ - 3) + (2*x₄ - 3)) / 4 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_transformation_l686_68673


namespace NUMINAMATH_CALUDE_median_salary_is_24000_l686_68628

structure SalaryGroup where
  title : String
  count : Nat
  salary : Nat

def company_data : List SalaryGroup := [
  ⟨"President", 1, 140000⟩,
  ⟨"Vice-President", 4, 92000⟩,
  ⟨"Director", 12, 75000⟩,
  ⟨"Associate Director", 8, 55000⟩,
  ⟨"Administrative Specialist", 38, 24000⟩
]

def total_employees : Nat := (company_data.map (λ g => g.count)).sum

theorem median_salary_is_24000 :
  total_employees = 63 →
  (∃ median_index : Nat, median_index = (total_employees + 1) / 2) →
  (∃ median_salary : Nat, 
    (company_data.map (λ g => List.replicate g.count g.salary)).join.get! (median_index - 1) = median_salary ∧
    median_salary = 24000) :=
by sorry

end NUMINAMATH_CALUDE_median_salary_is_24000_l686_68628


namespace NUMINAMATH_CALUDE_sum_nine_is_negative_fiftyfour_l686_68606

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = 2
  fifth_term : a 5 = 3 * a 3
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

/-- Theorem: The sum of the first 9 terms of the given arithmetic sequence is -54 -/
theorem sum_nine_is_negative_fiftyfour (seq : ArithmeticSequence) : sum_n seq 9 = -54 := by
  sorry

end NUMINAMATH_CALUDE_sum_nine_is_negative_fiftyfour_l686_68606


namespace NUMINAMATH_CALUDE_ivan_milkshake_cost_l686_68670

/-- The cost of Ivan's milkshake -/
def milkshake_cost (initial_amount : ℚ) (cupcake_fraction : ℚ) (final_amount : ℚ) : ℚ :=
  initial_amount - initial_amount * cupcake_fraction - final_amount

/-- Theorem: The cost of Ivan's milkshake is $5 -/
theorem ivan_milkshake_cost :
  milkshake_cost 10 (1/5) 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ivan_milkshake_cost_l686_68670


namespace NUMINAMATH_CALUDE_statement_equivalence_l686_68655

-- Define the statement as a function that takes a real number y
def statementIsTrue (y : ℝ) : Prop := (1/2 * y + 5) > 0

-- Define the theorem
theorem statement_equivalence :
  ∀ y : ℝ, statementIsTrue y ↔ (1/2 * y + 5 > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_statement_equivalence_l686_68655


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_primes_l686_68689

def primes : List Nat := [11, 17, 19, 23, 29, 37, 41]

def is_divisible_by_all (n : Nat) (lst : List Nat) : Prop :=
  ∀ p ∈ lst, (n % p = 0)

theorem smallest_number_divisible_by_primes :
  ∀ n : Nat,
    (n < 3075837206 →
      ¬(is_divisible_by_all (n - 27) primes)) ∧
    (is_divisible_by_all (3075837206 - 27) primes) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_primes_l686_68689


namespace NUMINAMATH_CALUDE_scientific_notation_of_goat_wool_fineness_l686_68633

theorem scientific_notation_of_goat_wool_fineness :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 0.000015 = a * (10 : ℝ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_goat_wool_fineness_l686_68633


namespace NUMINAMATH_CALUDE_candy_bar_count_l686_68624

theorem candy_bar_count (num_bags : ℕ) (bars_per_bag : ℕ) (h1 : num_bags = 5) (h2 : bars_per_bag = 3) :
  num_bags * bars_per_bag = 15 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_count_l686_68624


namespace NUMINAMATH_CALUDE_mix_paint_intensity_theorem_l686_68685

/-- Calculates the intensity of a paint mixture after replacing a portion of the original paint with a different intensity paint. -/
def mixPaintIntensity (originalIntensity replacementIntensity fractionReplaced : ℚ) : ℚ :=
  (1 - fractionReplaced) * originalIntensity + fractionReplaced * replacementIntensity

/-- Theorem stating that mixing 10% intensity paint with 20% intensity paint in equal proportions results in 15% intensity. -/
theorem mix_paint_intensity_theorem :
  mixPaintIntensity (1/10) (1/5) (1/2) = (3/20) := by
  sorry

#eval mixPaintIntensity (1/10) (1/5) (1/2)

end NUMINAMATH_CALUDE_mix_paint_intensity_theorem_l686_68685


namespace NUMINAMATH_CALUDE_equivalent_expression_l686_68677

theorem equivalent_expression (x : ℝ) (h : x < 0) :
  Real.sqrt (x / (1 - (x^2 - 1) / x)) = -x / Real.sqrt (x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_expression_l686_68677


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l686_68651

/-- The ellipse C in standard form -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line l that intersects the ellipse -/
def line_l (k m x y : ℝ) : Prop := y = k * x + m

/-- The condition for the intersection points A and B -/
def intersection_condition (xA yA xB yB : ℝ) : Prop :=
  (2 * xA + xB)^2 + (2 * yA + yB)^2 = (2 * xA - xB)^2 + (2 * yA - yB)^2

/-- The main theorem -/
theorem ellipse_intersection_theorem :
  ∀ (k m : ℝ),
    (∃ (xA yA xB yB : ℝ),
      ellipse_C xA yA ∧ ellipse_C xB yB ∧
      line_l k m xA yA ∧ line_l k m xB yB ∧
      intersection_condition xA yA xB yB) ↔
    (m < -Real.sqrt 3 / 2 ∨ m > Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l686_68651


namespace NUMINAMATH_CALUDE_sequence_property_l686_68688

/-- A sequence where all terms are distinct starting from index 2 -/
def DistinctSequence (x : ℕ → ℝ) : Prop :=
  ∀ i j, i ≥ 2 → j ≥ 2 → i ≠ j → x i ≠ x j

/-- The recurrence relation for the sequence -/
def SatisfiesRecurrence (x : ℕ → ℝ) : Prop :=
  ∀ n, x n = (x (n - 1) + 98 * x n + x (n + 1)) / 100

theorem sequence_property (x : ℕ → ℝ) 
    (h1 : DistinctSequence x) 
    (h2 : SatisfiesRecurrence x) : 
  Real.sqrt ((x 2023 - x 1) / 2022 * (2021 / (x 2023 - x 2))) + 2021 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l686_68688


namespace NUMINAMATH_CALUDE_band_formation_max_members_l686_68687

theorem band_formation_max_members :
  ∀ (r x : ℕ),
    r * x + 2 < 100 →
    (r - 2) * (x + 1) = r * x + 2 →
    ∀ (m : ℕ),
      m < 100 →
      ∃ (r' x' : ℕ),
        r' * x' + 2 = m →
        (r' - 2) * (x' + 1) = m →
        m ≤ 98 :=
by sorry

end NUMINAMATH_CALUDE_band_formation_max_members_l686_68687


namespace NUMINAMATH_CALUDE_max_volume_rectangular_prism_l686_68627

/-- Represents the volume of a rectangular prism as a function of the shorter base edge length -/
def prism_volume (x : ℝ) : ℝ := x * (x + 0.5) * (3.2 - 2 * x)

/-- The theorem stating the maximum volume and corresponding height of the rectangular prism -/
theorem max_volume_rectangular_prism :
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < 1.6 ∧
    prism_volume x = 1.8 ∧
    3.2 - 2 * x = 1.2 ∧
    ∀ (y : ℝ), 0 < y ∧ y < 1.6 → prism_volume y ≤ prism_volume x :=
sorry


end NUMINAMATH_CALUDE_max_volume_rectangular_prism_l686_68627


namespace NUMINAMATH_CALUDE_tangerine_sum_l686_68658

theorem tangerine_sum (initial_count : ℕ) (final_counts : List ℕ) : 
  initial_count = 20 →
  final_counts = [10, 18, 17, 13, 16] →
  (final_counts.filter (· ≤ 13)).sum = 23 := by
  sorry

end NUMINAMATH_CALUDE_tangerine_sum_l686_68658


namespace NUMINAMATH_CALUDE_consecutive_numbers_theorem_l686_68694

theorem consecutive_numbers_theorem (n : ℕ) (avg : ℚ) (largest : ℕ) : 
  n > 0 ∧ 
  avg = 20 ∧ 
  largest = 23 ∧ 
  (↑largest - ↑(n - 1) + ↑largest) / 2 = avg → 
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_theorem_l686_68694


namespace NUMINAMATH_CALUDE_polynomial_factorization_l686_68630

theorem polynomial_factorization (x y : ℝ) :
  x^4 + 4*y^4 = (x^2 - 2*x*y + 2*y^2) * (x^2 + 2*x*y + 2*y^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l686_68630


namespace NUMINAMATH_CALUDE_exactly_one_positive_integer_solution_l686_68699

theorem exactly_one_positive_integer_solution : 
  ∃! (n : ℕ), n > 0 ∧ 16 - 4 * n > 10 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_positive_integer_solution_l686_68699


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_l686_68698

-- Define the functions f and g
def f (a x : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

-- Part 1
theorem solution_set_when_a_eq_2 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
sorry

-- Part 2
theorem range_of_a :
  (∀ x : ℝ, f a x + g x ≥ 3) ↔ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_l686_68698


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l686_68675

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_m_value :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (m, m - 2)
  collinear a b → m = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l686_68675


namespace NUMINAMATH_CALUDE_range_of_a_l686_68649

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

-- Define the set of valid values for a
def valid_a : Set ℝ := {a | (1 < a ∧ a < 2) ∨ a ≤ -2}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ valid_a :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l686_68649


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_is_180_l686_68684

-- Define a triangle in Euclidean space
structure EuclideanTriangle where
  -- We don't need to specify the exact properties of a triangle here
  -- as we're focusing on the angle sum property

-- Define the concept of interior angles of a triangle
def interior_angles (t : EuclideanTriangle) : ℝ := sorry

-- State the theorem about the sum of interior angles
theorem sum_of_interior_angles_is_180 (t : EuclideanTriangle) :
  interior_angles t = 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_is_180_l686_68684


namespace NUMINAMATH_CALUDE_baseball_game_attendance_difference_proof_baseball_game_attendance_difference_l686_68696

theorem baseball_game_attendance_difference : ℕ → Prop :=
  fun difference =>
    ∀ (second_game_attendance : ℕ)
      (first_game_attendance : ℕ)
      (third_game_attendance : ℕ)
      (last_week_total : ℕ),
    second_game_attendance = 80 →
    first_game_attendance = second_game_attendance - 20 →
    third_game_attendance = second_game_attendance + 15 →
    last_week_total = 200 →
    difference = (first_game_attendance + second_game_attendance + third_game_attendance) - last_week_total →
    difference = 35

-- The proof of the theorem
theorem proof_baseball_game_attendance_difference : 
  baseball_game_attendance_difference 35 := by
  sorry

end NUMINAMATH_CALUDE_baseball_game_attendance_difference_proof_baseball_game_attendance_difference_l686_68696


namespace NUMINAMATH_CALUDE_ben_points_l686_68667

theorem ben_points (zach_points ben_points : ℕ) 
  (h1 : zach_points = 42)
  (h2 : zach_points = ben_points + 21) : 
  ben_points = 21 := by
sorry

end NUMINAMATH_CALUDE_ben_points_l686_68667


namespace NUMINAMATH_CALUDE_megacorp_mining_earnings_l686_68644

/-- MegaCorp's daily earnings from mining -/
def daily_mining_earnings : ℝ := 67111111.11

/-- MegaCorp's daily earnings from oil refining -/
def daily_oil_earnings : ℝ := 5000000

/-- MegaCorp's monthly expenses -/
def monthly_expenses : ℝ := 30000000

/-- MegaCorp's fine -/
def fine : ℝ := 25600000

/-- The fine percentage of annual profits -/
def fine_percentage : ℝ := 0.01

/-- Number of days in a month (approximation) -/
def days_in_month : ℝ := 30

/-- Number of months in a year -/
def months_in_year : ℝ := 12

theorem megacorp_mining_earnings :
  fine = fine_percentage * months_in_year * (days_in_month * (daily_mining_earnings + daily_oil_earnings) - monthly_expenses) :=
by sorry

end NUMINAMATH_CALUDE_megacorp_mining_earnings_l686_68644


namespace NUMINAMATH_CALUDE_circle_line_intersection_l686_68676

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Define the point P
def point_P : ℝ × ℝ := (0, 2)

-- Define a line passing through point P
def line_through_P (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the chord length
def chord_length (k : ℝ) : ℝ := sorry

-- Define the arc ratio
def arc_ratio (k : ℝ) : ℝ := sorry

theorem circle_line_intersection :
  ∀ (k : ℝ),
  (chord_length k = 2 → (k = 0 ∨ k = 3/4)) ∧
  (arc_ratio k = 3/1 → (k = 1/3 ∨ k = -3)) :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l686_68676


namespace NUMINAMATH_CALUDE_cube_coloring_theorem_l686_68645

/-- Represents a point in the cube --/
inductive CubePoint
| Center
| FaceCenter
| Vertex
| EdgeCenter

/-- Represents a color --/
inductive Color
| Blue
| Red

/-- Represents a straight line in the cube --/
structure Line where
  points : List CubePoint
  aligned : points.length = 3

/-- A coloring of the cube points --/
def Coloring := CubePoint → Color

/-- The set of all points in the cube --/
def cubePoints : List CubePoint := 
  [CubePoint.Center] ++ 
  List.replicate 6 CubePoint.FaceCenter ++
  List.replicate 8 CubePoint.Vertex ++
  List.replicate 12 CubePoint.EdgeCenter

/-- Theorem: For any coloring of the cube points, there exists a line with three points of the same color --/
theorem cube_coloring_theorem :
  ∀ (coloring : Coloring),
  ∃ (line : Line),
  ∀ (p : CubePoint),
  p ∈ line.points → coloring p = coloring (line.points.get ⟨0, by sorry⟩) :=
by sorry

end NUMINAMATH_CALUDE_cube_coloring_theorem_l686_68645


namespace NUMINAMATH_CALUDE_ellipse_slope_l686_68611

/-- Given an ellipse with eccentricity √3/2 and a point P on the ellipse such that
    the sum of tangents of angles formed by PA and PB with the x-axis is 1,
    prove that the slope of PA is (1 ± √2)/2. -/
theorem ellipse_slope (a b : ℝ) (x y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  let e := Real.sqrt 3 / 2
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  let A := (-a, 0)
  let B := (a, 0)
  let P := (x, y)
  ∀ (α β : ℝ),
    e = Real.sqrt (a^2 - b^2) / a →
    P ∈ C →
    (y / (x + a)) + (y / (x - a)) = 1 →
    (∃ (k : ℝ), k = y / (x + a) ∧ (k = (1 + Real.sqrt 2) / 2 ∨ k = (1 - Real.sqrt 2) / 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_slope_l686_68611


namespace NUMINAMATH_CALUDE_expression_evaluation_l686_68641

theorem expression_evaluation : 3^(0^(1^2)) + ((3^0)^2)^1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l686_68641


namespace NUMINAMATH_CALUDE_largest_value_is_B_l686_68653

theorem largest_value_is_B (a b c e : ℚ) : 
  a = (1/2) / (3/4) →
  b = 1 / ((2/3) / 4) →
  c = ((1/2) / 3) / 4 →
  e = (1 / (2/3)) / 4 →
  b > a ∧ b > c ∧ b > e :=
by sorry

end NUMINAMATH_CALUDE_largest_value_is_B_l686_68653


namespace NUMINAMATH_CALUDE_max_divisors_1_to_20_l686_68681

def divisorCount (n : ℕ) : ℕ := (Finset.filter (·∣n) (Finset.range (n + 1))).card

def maxDivisorCount : ℕ → ℕ
  | 0 => 0
  | n + 1 => max (maxDivisorCount n) (divisorCount (n + 1))

theorem max_divisors_1_to_20 :
  maxDivisorCount 20 = 6 ∧
  divisorCount 12 = 6 ∧
  divisorCount 18 = 6 ∧
  divisorCount 20 = 6 ∧
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → divisorCount n ≤ 6 :=
by sorry

#eval maxDivisorCount 20
#eval divisorCount 12
#eval divisorCount 18
#eval divisorCount 20

end NUMINAMATH_CALUDE_max_divisors_1_to_20_l686_68681


namespace NUMINAMATH_CALUDE_pulley_center_distance_l686_68654

def pulley_problem (r₁ r₂ d : ℝ) : Prop :=
  r₁ > 0 ∧ r₂ > 0 ∧ d > 0 ∧
  r₁ = 10 ∧ r₂ = 6 ∧ d = 26 →
  ∃ (center_distance : ℝ),
    center_distance = 2 * Real.sqrt 173

theorem pulley_center_distance :
  ∀ (r₁ r₂ d : ℝ), pulley_problem r₁ r₂ d :=
by sorry

end NUMINAMATH_CALUDE_pulley_center_distance_l686_68654


namespace NUMINAMATH_CALUDE_complex_number_location_l686_68613

theorem complex_number_location (z : ℂ) : 
  z * Complex.I = 2015 - Complex.I → 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l686_68613


namespace NUMINAMATH_CALUDE_age_ratio_after_time_l686_68617

/-- Represents a person's age -/
structure Age where
  value : ℕ

/-- Represents the ratio between two ages -/
structure AgeRatio where
  numerator : ℕ
  denominator : ℕ

def Age.addYears (a : Age) (years : ℕ) : Age :=
  ⟨a.value + years⟩

def Age.subtractYears (a : Age) (years : ℕ) : Age :=
  ⟨a.value - years⟩

def AgeRatio.fromAges (a b : Age) : AgeRatio :=
  ⟨a.value, b.value⟩

theorem age_ratio_after_time (sandy_age molly_age : Age) 
    (h1 : AgeRatio.fromAges sandy_age molly_age = ⟨7, 2⟩)
    (h2 : (sandy_age.subtractYears 6).value = 78) :
    AgeRatio.fromAges (sandy_age.addYears 16) (molly_age.addYears 16) = ⟨5, 2⟩ := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_after_time_l686_68617


namespace NUMINAMATH_CALUDE_led_messages_count_l686_68618

/-- Represents the number of LEDs in the row -/
def n : ℕ := 7

/-- Represents the number of LEDs that are lit -/
def k : ℕ := 3

/-- Represents the number of color options for each lit LED -/
def colors : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of ways to arrange k non-adjacent items in n+1 slots -/
def nonAdjacentArrangements (n k : ℕ) : ℕ := choose (n + 1 - k) k

/-- Calculates the total number of different messages -/
def totalMessages : ℕ := nonAdjacentArrangements n k * colors^k

theorem led_messages_count : totalMessages = 80 := by
  sorry

end NUMINAMATH_CALUDE_led_messages_count_l686_68618


namespace NUMINAMATH_CALUDE_value_of_b_l686_68652

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 3) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l686_68652


namespace NUMINAMATH_CALUDE_calculation_proof_l686_68623

theorem calculation_proof : 2 * (75 * 1313 - 25 * 1313) = 131300 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l686_68623


namespace NUMINAMATH_CALUDE_unique_representation_of_two_over_prime_l686_68642

theorem unique_representation_of_two_over_prime (p : ℕ) (h_prime : Nat.Prime p) (h_gt_two : p > 2) :
  ∃! (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (2 : ℚ) / p = 1 / x + 1 / y ∧ x = p * (p + 1) / 2 ∧ y = (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_representation_of_two_over_prime_l686_68642


namespace NUMINAMATH_CALUDE_snowball_partition_l686_68637

/-- A directed graph where each vertex has an out-degree of exactly 1 -/
structure SnowballGraph (V : Type) :=
  (edges : V → V)

/-- A partition of vertices into three sets -/
def ThreeTeamPartition (V : Type) := V → Fin 3

theorem snowball_partition {V : Type} (G : SnowballGraph V) :
  ∃ (partition : ThreeTeamPartition V),
    ∀ (v w : V), G.edges v = w → partition v ≠ partition w :=
sorry

end NUMINAMATH_CALUDE_snowball_partition_l686_68637


namespace NUMINAMATH_CALUDE_no_sequence_satisfying_conditions_l686_68643

theorem no_sequence_satisfying_conditions : ¬ ∃ (a : ℕ → ℤ), 
  (∀ i j : ℕ, i ≠ j → a i ≠ a j) ∧ 
  (∀ k : ℕ, k > 0 → a (k^2) > 0 ∧ a (k^2 + k) < 0) ∧
  (∀ n : ℕ, n > 0 → |a (n + 1) - a n| ≤ 2023 * Real.sqrt n) :=
by sorry

end NUMINAMATH_CALUDE_no_sequence_satisfying_conditions_l686_68643


namespace NUMINAMATH_CALUDE_inequality_proof_l686_68664

theorem inequality_proof (a b : ℝ) (h1 : b < 0) (h2 : 0 < a) : a - b > a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l686_68664


namespace NUMINAMATH_CALUDE_OTVSU_shape_l686_68601

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The figure OTVSU -/
structure OTVSU where
  O : Point2D
  T : Point2D
  V : Point2D
  S : Point2D
  U : Point2D

/-- Predicate to check if a figure is a parallelogram -/
def isParallelogram (f : OTVSU) : Prop := sorry

/-- Predicate to check if a figure is a straight line -/
def isStraightLine (f : OTVSU) : Prop := sorry

/-- Predicate to check if a figure is a trapezoid -/
def isTrapezoid (f : OTVSU) : Prop := sorry

theorem OTVSU_shape :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  let f : OTVSU := {
    O := ⟨0, 0⟩,
    T := ⟨x₁, y₁⟩,
    V := ⟨x₁ + x₂, y₁ + y₂⟩,
    S := ⟨x₁ - x₂, y₁ - y₂⟩,
    U := ⟨x₂, y₂⟩
  }
  (isParallelogram f ∨ isStraightLine f) ∧ ¬isTrapezoid f := by
  sorry

end NUMINAMATH_CALUDE_OTVSU_shape_l686_68601


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l686_68691

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, (-1 < x ∧ x < 3) → x < 3) ∧
  (∃ x : ℝ, x < 3 ∧ ¬(-1 < x ∧ x < 3)) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l686_68691


namespace NUMINAMATH_CALUDE_match_end_probability_l686_68604

/-- The probability of player A winning a single game -/
def prob_A_win : ℝ := 0.6

/-- The probability of player B winning a single game -/
def prob_B_win : ℝ := 0.4

/-- The probability that the match ends after two more games -/
def prob_match_ends : ℝ := prob_A_win * prob_A_win + prob_B_win * prob_B_win

/-- Theorem stating that the probability of the match ending after two more games is 0.52 -/
theorem match_end_probability : prob_match_ends = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_match_end_probability_l686_68604


namespace NUMINAMATH_CALUDE_polynomial_simplification_l686_68661

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + x^3 + 5) - (x^6 + 4 * x^5 + 2 * x^4 - x^3 + 7) =
  x^6 - x^5 - x^4 + 2 * x^3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l686_68661


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l686_68692

/-- A circle tangent to the coordinate axes and passing through (2, 1) -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = center.2
  passes_through : (2 - center.1)^2 + (1 - center.2)^2 = radius^2

/-- The equation of the circle -/
def circle_equation (c : TangentCircle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Theorem stating the possible equations of the circle -/
theorem tangent_circle_equation :
  ∀ c : TangentCircle,
  (∀ x y : ℝ, circle_equation c x y ↔ (x - 5)^2 + (y - 5)^2 = 25) ∨
  (∀ x y : ℝ, circle_equation c x y ↔ (x - 1)^2 + (y - 1)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l686_68692


namespace NUMINAMATH_CALUDE_distance_AC_l686_68682

-- Define the movement from A to C
def south_displacement : ℝ := 10
def east_displacement : ℝ := 5

-- Theorem statement
theorem distance_AC : Real.sqrt (south_displacement ^ 2 + east_displacement ^ 2) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_AC_l686_68682


namespace NUMINAMATH_CALUDE_square_root_difference_limit_l686_68672

theorem square_root_difference_limit : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |Real.sqrt (n + 1) - Real.sqrt n| < ε := by
sorry

end NUMINAMATH_CALUDE_square_root_difference_limit_l686_68672


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l686_68607

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- Given three collinear points (4, 10), (-3, k), and (-8, 5), prove that k = 85/12. -/
theorem collinear_points_k_value :
  collinear 4 10 (-3) k (-8) 5 → k = 85/12 :=
by
  sorry


end NUMINAMATH_CALUDE_collinear_points_k_value_l686_68607


namespace NUMINAMATH_CALUDE_ratio_from_linear_equation_l686_68690

theorem ratio_from_linear_equation (x y : ℝ) (h : 2 * y - 5 * x = 0) :
  ∃ (k : ℝ), k > 0 ∧ x = 2 * k ∧ y = 5 * k :=
sorry

end NUMINAMATH_CALUDE_ratio_from_linear_equation_l686_68690


namespace NUMINAMATH_CALUDE_parallelogram_side_lengths_l686_68621

def parallelogram_properties (angle : ℝ) (shorter_diagonal : ℝ) (perpendicular : ℝ) : Prop :=
  angle = 60 ∧ 
  shorter_diagonal = 2 * Real.sqrt 31 ∧ 
  perpendicular = Real.sqrt 75 / 2

theorem parallelogram_side_lengths 
  (angle : ℝ) (shorter_diagonal : ℝ) (perpendicular : ℝ) 
  (h : parallelogram_properties angle shorter_diagonal perpendicular) :
  ∃ (longer_side shorter_side longer_diagonal : ℝ),
    longer_side = 12 ∧ 
    shorter_side = 10 ∧ 
    longer_diagonal = 2 * Real.sqrt 91 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_side_lengths_l686_68621


namespace NUMINAMATH_CALUDE_symmetry_y_axis_l686_68665

/-- Given a point (x, y) in the plane, its reflection across the y-axis is the point (-x, y) -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- A point q is symmetric to p with respect to the y-axis if q is the reflection of p across the y-axis -/
def symmetric_y_axis (p q : ℝ × ℝ) : Prop :=
  q = reflect_y_axis p

theorem symmetry_y_axis :
  symmetric_y_axis (-2, 3) (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_y_axis_l686_68665


namespace NUMINAMATH_CALUDE_max_handshakes_specific_gathering_l686_68683

/-- Represents a gathering of people and their handshakes. -/
structure Gathering where
  people : Nat
  restricted_people : Nat
  max_handshakes_per_person : Nat
  max_handshakes_for_restricted : Nat

/-- Calculates the maximum number of handshakes in a gathering. -/
def max_handshakes (g : Gathering) : Nat :=
  sorry

/-- Theorem stating the maximum number of handshakes for the specific gathering. -/
theorem max_handshakes_specific_gathering :
  let g : Gathering := {
    people := 30,
    restricted_people := 5,
    max_handshakes_per_person := 29,
    max_handshakes_for_restricted := 10
  }
  max_handshakes g = 325 := by
  sorry

end NUMINAMATH_CALUDE_max_handshakes_specific_gathering_l686_68683


namespace NUMINAMATH_CALUDE_ratio_problem_l686_68614

theorem ratio_problem (w x y z : ℚ) 
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 3 / 4)
  (h3 : x / z = 2 / 5) :
  (x + y) / (y + z) = 26 / 53 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l686_68614


namespace NUMINAMATH_CALUDE_parabola_vertex_l686_68669

/-- Given a quadratic function f(x) = -x^2 + cx + d where f(x) ≤ 0 has solutions [1,∞) and (-∞,-7],
    the vertex of the parabola defined by f(x) is (-3, 16). -/
theorem parabola_vertex (c d : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = -x^2 + c*x + d)
    (h2 : ∀ x ≥ 1, f x ≤ 0)
    (h3 : ∀ x ≤ -7, f x ≤ 0)
    (h4 : ∀ x, -7 < x ∧ x < 1 → f x > 0) :
  (∃ y, f (-3) = y ∧ ∀ x, f x ≤ y) ∧ f (-3) = 16 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l686_68669


namespace NUMINAMATH_CALUDE_helen_lawn_mowing_gas_usage_l686_68646

/-- Represents the lawn cutting schedule and gas usage for Helen's lawn mowing --/
structure LawnCuttingSchedule where
  march_to_october_low_freq : Nat  -- Number of months with 2 cuts per month
  may_to_august_high_freq : Nat    -- Number of months with 4 cuts per month
  cuts_per_low_freq_month : Nat    -- Number of cuts in low frequency months
  cuts_per_high_freq_month : Nat   -- Number of cuts in high frequency months
  gas_usage_frequency : Nat        -- Every nth cut uses gas
  gas_usage_amount : Nat           -- Amount of gas used every nth cut

/-- Calculates the total gas usage for Helen's lawn mowing schedule --/
def calculate_gas_usage (schedule : LawnCuttingSchedule) : Nat :=
  let total_cuts := 
    schedule.march_to_october_low_freq * schedule.cuts_per_low_freq_month +
    schedule.may_to_august_high_freq * schedule.cuts_per_high_freq_month
  let gas_usage_times := total_cuts / schedule.gas_usage_frequency
  gas_usage_times * schedule.gas_usage_amount

/-- Theorem stating that Helen's lawn mowing schedule results in 12 gallons of gas usage --/
theorem helen_lawn_mowing_gas_usage :
  let schedule : LawnCuttingSchedule := {
    march_to_october_low_freq := 4
    may_to_august_high_freq := 4
    cuts_per_low_freq_month := 2
    cuts_per_high_freq_month := 4
    gas_usage_frequency := 4
    gas_usage_amount := 2
  }
  calculate_gas_usage schedule = 12 := by
  sorry


end NUMINAMATH_CALUDE_helen_lawn_mowing_gas_usage_l686_68646


namespace NUMINAMATH_CALUDE_sum_of_cubes_up_to_8_l686_68626

/-- Sum of cubes from 1³ to n³ equals the square of the sum of first n natural numbers -/
def sum_of_cubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

/-- The sum of cubes from 1³ to 8³ is 1296 -/
theorem sum_of_cubes_up_to_8 : sum_of_cubes 8 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_up_to_8_l686_68626


namespace NUMINAMATH_CALUDE_number_of_employees_l686_68678

/-- Proves the number of employees in an organization given salary information --/
theorem number_of_employees
  (avg_salary : ℝ)
  (new_avg_salary : ℝ)
  (manager_salary : ℝ)
  (h1 : avg_salary = 1500)
  (h2 : new_avg_salary = 1650)
  (h3 : manager_salary = 4650) :
  ∃ (num_employees : ℕ),
    (num_employees : ℝ) * avg_salary + manager_salary = (num_employees + 1) * new_avg_salary ∧
    num_employees = 20 := by
  sorry


end NUMINAMATH_CALUDE_number_of_employees_l686_68678


namespace NUMINAMATH_CALUDE_triangle_angles_l686_68668

theorem triangle_angles (α β γ : Real) : 
  (180 - α) / (180 - β) = 13 / 9 →
  (180 - α) - (180 - β) = 45 →
  α + β + γ = 180 →
  α = 33.75 ∧ β = 78.75 ∧ γ = 67.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_angles_l686_68668


namespace NUMINAMATH_CALUDE_game_winner_parity_l686_68660

/-- The game state representing the current rectangle -/
structure GameState where
  width : ℕ
  height : ℕ
  area : ℕ
  h_width : width > 1
  h_height : height > 1
  h_area : area = width * height

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- The game rules and win condition -/
def game_rules (initial_state : GameState) : GameResult :=
  if initial_state.area % 2 = 1 then
    GameResult.FirstPlayerWins
  else
    GameResult.SecondPlayerWins

/-- The main theorem stating the winning condition based on initial area parity -/
theorem game_winner_parity (m n : ℕ) (h_m : m > 1) (h_n : n > 1) :
  let initial_state : GameState := {
    width := m,
    height := n,
    area := m * n,
    h_width := h_m,
    h_height := h_n,
    h_area := rfl
  }
  game_rules initial_state =
    if m * n % 2 = 1 then
      GameResult.FirstPlayerWins
    else
      GameResult.SecondPlayerWins :=
sorry

end NUMINAMATH_CALUDE_game_winner_parity_l686_68660


namespace NUMINAMATH_CALUDE_material_left_proof_l686_68671

theorem material_left_proof (material1 material2 material_used : ℚ) :
  material1 = 4 / 17 →
  material2 = 3 / 10 →
  material_used = 0.23529411764705882 →
  material1 + material2 - material_used = 51 / 170 := by
  sorry

end NUMINAMATH_CALUDE_material_left_proof_l686_68671


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l686_68612

theorem convex_polygon_sides (angles_sum : ℝ) (n : ℕ) : 
  angles_sum = 3150 → 
  (180 * (n - 2) : ℝ) > angles_sum →
  (180 * (n - 2) : ℝ) - angles_sum < 180 →
  n = 20 := by
  sorry

#check convex_polygon_sides

end NUMINAMATH_CALUDE_convex_polygon_sides_l686_68612


namespace NUMINAMATH_CALUDE_geometric_series_sum_l686_68629

/-- Sum of a geometric series with n terms -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- First term of the geometric series -/
def a : ℚ := 1/4

/-- Common ratio of the geometric series -/
def r : ℚ := 1/4

/-- Number of terms to sum -/
def n : ℕ := 6

theorem geometric_series_sum :
  geometricSum a r n = 4095/12288 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l686_68629


namespace NUMINAMATH_CALUDE_second_number_value_l686_68605

theorem second_number_value (x y z : ℝ) : 
  x + y + z = 660 ∧ 
  x = 2 * y ∧ 
  z = (1 / 3) * x → 
  y = 180 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l686_68605


namespace NUMINAMATH_CALUDE_moles_of_ch4_combined_l686_68636

-- Define the chemical reaction
structure Reaction where
  ch4 : ℝ
  cl2 : ℝ
  ch3cl : ℝ
  hcl : ℝ

-- Define the stoichiometric coefficients
def stoichiometric_ratio : Reaction → Prop :=
  fun r => r.ch4 = r.cl2 ∧ r.ch4 = r.ch3cl ∧ r.ch4 = r.hcl

-- Define the theorem
theorem moles_of_ch4_combined 
  (r : Reaction) 
  (h1 : stoichiometric_ratio r) 
  (h2 : r.ch3cl = 2) 
  (h3 : r.cl2 = 2) : 
  r.ch4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_moles_of_ch4_combined_l686_68636


namespace NUMINAMATH_CALUDE_tangent_line_equation_l686_68602

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_equation :
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧
  y₀ = f x₀ ∧
  (Real.log x₀ + 1) * 0 - (-1) = (Real.log x₀ + 1) * x₀ - y₀ ∧
  ∀ (x y : ℝ), y = Real.log x₀ + 1 * (x - x₀) + y₀ ↔ x - y - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l686_68602


namespace NUMINAMATH_CALUDE_cos_a_minus_pi_l686_68608

theorem cos_a_minus_pi (a : Real) 
  (h1 : π / 2 < a ∧ a < π) 
  (h2 : 3 * Real.sin (2 * a) = 2 * Real.cos a) : 
  Real.cos (a - π) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_a_minus_pi_l686_68608


namespace NUMINAMATH_CALUDE_triangle_area_with_median_l686_68666

/-- The area of a triangle with two sides 1 and √15, and a median to the third side equal to 2, is √15/2. -/
theorem triangle_area_with_median (a b c : ℝ) (m : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 15) (h3 : m = 2) :
  (1/2 : ℝ) * a * b = (Real.sqrt 15) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_median_l686_68666


namespace NUMINAMATH_CALUDE_complex_modulus_l686_68625

theorem complex_modulus (z : ℂ) (h : (1 + 2*I)*z = 3 - 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l686_68625


namespace NUMINAMATH_CALUDE_fence_pole_count_l686_68632

/-- Calculates the number of fence poles required for a path with bridges -/
def fence_poles (total_length : ℕ) (pole_spacing : ℕ) (bridge_lengths : List ℕ) : ℕ :=
  let fenced_length := total_length - bridge_lengths.sum
  let poles_per_side := fenced_length / pole_spacing
  let total_poles := 2 * poles_per_side + 2
  total_poles

theorem fence_pole_count : 
  fence_poles 2300 8 [48, 58, 62] = 534 := by
  sorry

end NUMINAMATH_CALUDE_fence_pole_count_l686_68632


namespace NUMINAMATH_CALUDE_cone_base_radius_l686_68693

/-- The radius of the base of a cone, given its surface area and net shape. -/
theorem cone_base_radius (S : ℝ) (r : ℝ) : 
  S = 9 * Real.pi  -- Surface area condition
  → S = 3 * Real.pi * r^2  -- Surface area formula for a cone
  → r = Real.sqrt 3 :=  -- Conclusion: radius is √3
by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l686_68693


namespace NUMINAMATH_CALUDE_last_twelve_average_l686_68615

theorem last_twelve_average (total_average : ℝ) (first_twelve_average : ℝ) (thirteenth_result : ℝ) :
  total_average = 20 →
  first_twelve_average = 14 →
  thirteenth_result = 128 →
  (25 * total_average - 12 * first_twelve_average - thirteenth_result) / 12 = 17 := by
sorry

end NUMINAMATH_CALUDE_last_twelve_average_l686_68615


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l686_68662

/-- Hyperbola with given properties has eccentricity √3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let f (x y : ℝ) := x^2 / a^2 - y^2 / b^2
  let l (x : ℝ) := Real.sqrt 3 / 3 * (x + c)
  c^2 = a^2 + b^2 →
  f c ((2 * Real.sqrt 3 * c) / 3) = 1 →
  l (-c) = 0 →
  l 0 = l c / 2 →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l686_68662


namespace NUMINAMATH_CALUDE_no_five_integers_solution_l686_68674

theorem no_five_integers_solution :
  ¬ ∃ (a b c d e : ℕ),
    (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c < d) ∧ (d < e) ∧
    ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} = 
     {15, 16, 17, 18, 19, 20, 21, 23} ∪ {x | x < 15} ∪ {y | y > 23}) :=
by
  sorry

#check no_five_integers_solution

end NUMINAMATH_CALUDE_no_five_integers_solution_l686_68674


namespace NUMINAMATH_CALUDE_range_of_f_l686_68610

-- Define the function f
def f (x : ℝ) : ℝ := x + |x - 2|

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l686_68610


namespace NUMINAMATH_CALUDE_g_of_quadratic_l686_68686

/-- Given that g(x) = 3x + 1 for all real numbers x, 
    prove that g(x^2 + 2x + 2) = 3x^2 + 6x + 7 -/
theorem g_of_quadratic (x : ℝ) : 
  let g : ℝ → ℝ := fun x ↦ 3 * x + 1
  g (x^2 + 2*x + 2) = 3*x^2 + 6*x + 7 := by
  sorry

end NUMINAMATH_CALUDE_g_of_quadratic_l686_68686


namespace NUMINAMATH_CALUDE_smallest_proportional_part_l686_68663

theorem smallest_proportional_part :
  let total : ℕ := 120
  let ratios : List ℕ := [3, 5, 7]
  let parts : List ℕ := ratios.map (λ r => r * (total / ratios.sum))
  parts.minimum? = some 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_proportional_part_l686_68663


namespace NUMINAMATH_CALUDE_chord_length_hyperbola_equation_l686_68639

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line with slope 1 passing through the right focus
def line (x y : ℝ) : Prop := y = x - Real.sqrt 3

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Theorem 1: Length of chord AB
theorem chord_length : 
  ∃ (A B : ℝ × ℝ), 
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8/5 :=
sorry

-- Theorem 2: Equation of the hyperbola
theorem hyperbola_equation :
  ∃ (a b : ℝ), 
    hyperbola 3 4 (-3) (2 * Real.sqrt 3) ∧
    ∀ (x y : ℝ), hyperbola a b x y ↔ 4*x^2/9 - y^2/4 = 1 :=
sorry

end NUMINAMATH_CALUDE_chord_length_hyperbola_equation_l686_68639


namespace NUMINAMATH_CALUDE_circle_area_ratio_l686_68622

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360) * (2 * Real.pi * r₁) = (48 / 360) * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l686_68622


namespace NUMINAMATH_CALUDE_probability_three_standard_parts_l686_68657

/-- Represents a box containing parts -/
structure Box where
  total : ℕ
  standard : ℕ
  h : standard ≤ total

/-- Calculates the probability of selecting a standard part from a box -/
def probabilityStandard (box : Box) : ℚ :=
  box.standard / box.total

/-- Theorem: The probability of selecting standard parts from all three boxes is 7/10 -/
theorem probability_three_standard_parts
  (box1 : Box)
  (box2 : Box)
  (box3 : Box)
  (h1 : box1.total = 30 ∧ box1.standard = 27)
  (h2 : box2.total = 30 ∧ box2.standard = 28)
  (h3 : box3.total = 30 ∧ box3.standard = 25) :
  probabilityStandard box1 * probabilityStandard box2 * probabilityStandard box3 = 7/10 := by
  sorry


end NUMINAMATH_CALUDE_probability_three_standard_parts_l686_68657


namespace NUMINAMATH_CALUDE_conic_section_is_hyperbola_l686_68634

/-- The equation (x-3)^2 = 3(2y+4)^2 - 75 represents a hyperbola -/
theorem conic_section_is_hyperbola :
  ∃ (a b c d e f : ℝ), 
    (∀ x y : ℝ, (x - 3)^2 = 3*(2*y + 4)^2 - 75 ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0) ∧
    a > 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_conic_section_is_hyperbola_l686_68634


namespace NUMINAMATH_CALUDE_circular_film_radius_l686_68619

/-- Given a cylindrical canister filled with a liquid that forms a circular film on water,
    this theorem proves that the radius of the resulting circular film is 25√2 cm. -/
theorem circular_film_radius
  (canister_radius : ℝ)
  (canister_height : ℝ)
  (film_thickness : ℝ)
  (h_canister_radius : canister_radius = 5)
  (h_canister_height : canister_height = 10)
  (h_film_thickness : film_thickness = 0.2) :
  let canister_volume := π * canister_radius^2 * canister_height
  let film_radius := Real.sqrt (canister_volume / (π * film_thickness))
  film_radius = 25 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_circular_film_radius_l686_68619


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l686_68697

/-- If the solution set of x² - mx - 6n < 0 is {x | -3 < x < 6}, then m + n = 6 -/
theorem solution_set_implies_sum (m n : ℝ) : 
  (∀ x, x^2 - m*x - 6*n < 0 ↔ -3 < x ∧ x < 6) → 
  m + n = 6 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l686_68697


namespace NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l686_68659

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => (k + 1) / 5^(k + 1))

/-- 30! has 7 trailing zeros -/
theorem thirty_factorial_trailing_zeros :
  trailingZeros 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l686_68659


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_l686_68695

theorem line_through_parabola_vertex (a : ℝ) : 
  let line := fun x => x + a
  let parabola := fun x => x^2 + a^2
  let vertex_x := 0
  let vertex_y := parabola vertex_x
  (∃! (a1 a2 : ℝ), a1 ≠ a2 ∧ 
    line vertex_x = vertex_y ∧ 
    ∀ a', line vertex_x = vertex_y → (a' = a1 ∨ a' = a2)) := by
  sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_l686_68695


namespace NUMINAMATH_CALUDE_firewood_collection_sum_l686_68656

/-- The amount of firewood collected by Kimberley in pounds -/
def kimberley_firewood : ℕ := 10

/-- The amount of firewood collected by Houston in pounds -/
def houston_firewood : ℕ := 12

/-- The amount of firewood collected by Ela in pounds -/
def ela_firewood : ℕ := 13

/-- The total amount of firewood collected by Kimberley, Ela, and Houston -/
def total_firewood : ℕ := kimberley_firewood + ela_firewood + houston_firewood

theorem firewood_collection_sum :
  total_firewood = 35 := by
  sorry

end NUMINAMATH_CALUDE_firewood_collection_sum_l686_68656


namespace NUMINAMATH_CALUDE_log_inequality_l686_68648

theorem log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l686_68648


namespace NUMINAMATH_CALUDE_x_value_l686_68647

/-- Given that ( √x ) / ( √0.81 ) + ( √1.44 ) / ( √0.49 ) = 2.879628878919216, prove that x = 1.1 -/
theorem x_value (x : ℝ) 
  (h : Real.sqrt x / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt 0.49 = 2.879628878919216) : 
  x = 1.1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l686_68647


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_value_l686_68650

/-- Given vectors a, b, and c in ℝ², prove that if a - b is perpendicular to c,
    then the value of m in b is -3. -/
theorem perpendicular_vectors_imply_m_value (a b c : ℝ × ℝ) (m : ℝ) :
  a = (-1, 2) →
  b = (m, -1) →
  c = (3, -2) →
  (a.1 - b.1) * c.1 + (a.2 - b.2) * c.2 = 0 →
  m = -3 := by
  sorry

#check perpendicular_vectors_imply_m_value

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_value_l686_68650


namespace NUMINAMATH_CALUDE_village_population_problem_l686_68600

theorem village_population_problem (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.85 = 2907 → P = 3801 := by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l686_68600


namespace NUMINAMATH_CALUDE_power_of_seven_mod_ten_l686_68640

theorem power_of_seven_mod_ten : 7^150 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_ten_l686_68640


namespace NUMINAMATH_CALUDE_equation_solution_l686_68631

theorem equation_solution (x : ℝ) : 1 + 1 / (1 + x) = 2 / (1 + x) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l686_68631


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l686_68635

/-- 
For a quadratic equation (k+2)x^2 - 2x - 1 = 0 to have real roots, 
k must satisfy the conditions k ≥ -3 and k ≠ -2.
-/
theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k + 2) * x^2 - 2 * x - 1 = 0) ↔ (k ≥ -3 ∧ k ≠ -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l686_68635


namespace NUMINAMATH_CALUDE_polynomial_roots_l686_68620

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 13*x - 15

-- State the theorem
theorem polynomial_roots :
  (∃ a b c : ℝ, a < 0 ∧ 0 < b ∧ 0 < c ∧
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l686_68620


namespace NUMINAMATH_CALUDE_may_birth_percentage_l686_68616

def total_mathematicians : ℕ := 120
def may_births : ℕ := 15

theorem may_birth_percentage :
  (may_births : ℚ) / total_mathematicians * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_may_birth_percentage_l686_68616


namespace NUMINAMATH_CALUDE_correct_propositions_count_l686_68609

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def intersection (p1 p2 : Plane) : Line := sorry
def skew_lines (l1 l2 : Line) : Prop := sorry

-- Define the propositions
def proposition1 (m n : Line) (α : Plane) : Prop :=
  perpendicular_lines m n → perpendicular m α → parallel n α

def proposition2 (m n : Line) (α β : Plane) : Prop :=
  perpendicular m α → perpendicular n β → parallel_lines m n → parallel_planes α β

def proposition3 (m n : Line) (α β : Plane) : Prop :=
  skew_lines m n → line_in_plane m α → line_in_plane n β → parallel m β → parallel n α → parallel_planes α β

def proposition4 (m n : Line) (α β : Plane) : Prop :=
  perpendicular_planes α β → intersection α β = m → line_in_plane n β → perpendicular_lines n m → perpendicular n α

-- Theorem statement
theorem correct_propositions_count :
  ¬proposition1 m n α ∧
  proposition2 m n α β ∧
  proposition3 m n α β ∧
  proposition4 m n α β :=
sorry

end NUMINAMATH_CALUDE_correct_propositions_count_l686_68609


namespace NUMINAMATH_CALUDE_inequality_proof_l686_68638

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a*c) :
  (a*f - c*d)^2 ≥ (a*e - b*d)*(b*f - c*e) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l686_68638


namespace NUMINAMATH_CALUDE_exists_five_threes_equal_100_l686_68603

/-- An arithmetic expression using only the number 3, parentheses, and arithmetic operations. -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression. -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Count the number of threes in an expression. -/
def countThrees : Expr → Nat
  | Expr.three => 1
  | Expr.add e1 e2 => countThrees e1 + countThrees e2
  | Expr.sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.div e1 e2 => countThrees e1 + countThrees e2

/-- There exists an arithmetic expression using five threes that evaluates to 100. -/
theorem exists_five_threes_equal_100 : ∃ e : Expr, countThrees e = 5 ∧ eval e = 100 := by
  sorry

end NUMINAMATH_CALUDE_exists_five_threes_equal_100_l686_68603


namespace NUMINAMATH_CALUDE_perpendicular_slope_l686_68679

/-- Given a line with equation 5x - 4y = 20, the slope of the perpendicular line is -4/5 -/
theorem perpendicular_slope (x y : ℝ) : 
  (5 * x - 4 * y = 20) → 
  (∃ m : ℝ, m = -4/5 ∧ m * (5/4) = -1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l686_68679


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l686_68680

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > y ∧ y > 0 → abs x > abs y) ∧
  ∃ a b : ℝ, abs a > abs b ∧ ¬(a > b ∧ b > 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l686_68680
