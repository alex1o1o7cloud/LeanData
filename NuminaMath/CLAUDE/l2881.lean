import Mathlib

namespace NUMINAMATH_CALUDE_mary_income_90_percent_of_juan_l2881_288161

/-- Represents the income of an individual -/
structure Income where
  amount : ℝ
  amount_pos : amount > 0

/-- The relationship between incomes of Mary, Tim, Juan, Sophia, and Alex -/
structure IncomeRelationship where
  alex : Income
  sophia : Income
  juan : Income
  tim : Income
  mary : Income
  sophia_alex : sophia.amount = 1.25 * alex.amount
  juan_sophia : juan.amount = 0.7 * sophia.amount
  tim_juan : tim.amount = 0.6 * juan.amount
  mary_tim : mary.amount = 1.5 * tim.amount

/-- Theorem stating that Mary's income is 90% of Juan's income -/
theorem mary_income_90_percent_of_juan (r : IncomeRelationship) : 
  r.mary.amount = 0.9 * r.juan.amount := by sorry

end NUMINAMATH_CALUDE_mary_income_90_percent_of_juan_l2881_288161


namespace NUMINAMATH_CALUDE_union_P_complement_Q_l2881_288157

open Set

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def Q : Set ℝ := {x | x^2 - 4 < 0}

-- State the theorem
theorem union_P_complement_Q : P ∪ (univ \ Q) = Iic (-2) ∪ Ici 1 := by sorry

end NUMINAMATH_CALUDE_union_P_complement_Q_l2881_288157


namespace NUMINAMATH_CALUDE_coordinates_sum_of_A_l2881_288162

/-- Given points B and C, and the condition that AC/AB = BC/AB = 1/3, 
    prove that the sum of coordinates of point A is -22 -/
theorem coordinates_sum_of_A (B C : ℝ × ℝ) (h : B = (2, -3) ∧ C = (-2, 6)) :
  let A : ℝ × ℝ := (3 * C.1 - 2 * B.1, 3 * C.2 - 2 * B.2)
  (A.1 + A.2 : ℝ) = -22 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_sum_of_A_l2881_288162


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l2881_288172

theorem quadratic_roots_expression (p q : ℝ) : 
  (3 * p^2 - 7 * p + 4 = 0) →
  (3 * q^2 - 7 * q + 4 = 0) →
  p ≠ q →
  (5 * p^3 - 5 * q^3) / (p - q) = 185 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l2881_288172


namespace NUMINAMATH_CALUDE_oranges_in_box_l2881_288149

/-- The number of oranges left in a box after some are removed -/
def oranges_left (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem stating that 20 oranges are left in the box -/
theorem oranges_in_box : oranges_left 55 35 = 20 := by
  sorry

end NUMINAMATH_CALUDE_oranges_in_box_l2881_288149


namespace NUMINAMATH_CALUDE_city_population_ratio_l2881_288128

def population_ratio (pop_Z : ℝ) : Prop :=
  let pop_Y := 2.5 * pop_Z
  let pop_X := 6 * pop_Y
  let pop_A := 3 * pop_X
  let pop_B := 4 * pop_Y
  (pop_X / (pop_Z + pop_B)) = 15 / 11

theorem city_population_ratio :
  ∀ pop_Z : ℝ, pop_Z > 0 → population_ratio pop_Z :=
by
  sorry

end NUMINAMATH_CALUDE_city_population_ratio_l2881_288128


namespace NUMINAMATH_CALUDE_volleyball_team_size_l2881_288155

theorem volleyball_team_size (managers : ℕ) (employees : ℕ) (teams : ℕ) :
  managers = 23 →
  employees = 7 →
  teams = 6 →
  (managers + employees) / teams = 5 := by
sorry

end NUMINAMATH_CALUDE_volleyball_team_size_l2881_288155


namespace NUMINAMATH_CALUDE_notebook_distribution_l2881_288184

theorem notebook_distribution (total_notebooks : ℕ) (half_students_notebooks : ℕ) :
  total_notebooks = 512 →
  half_students_notebooks = 16 →
  ∃ (num_students : ℕ) (fraction : ℚ),
    num_students > 0 ∧
    fraction > 0 ∧
    fraction < 1 ∧
    (num_students / 2 : ℚ) * half_students_notebooks = total_notebooks ∧
    (num_students : ℚ) * (fraction * num_students) = total_notebooks ∧
    fraction = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_notebook_distribution_l2881_288184


namespace NUMINAMATH_CALUDE_mean_temperature_is_80_point_2_l2881_288186

def temperatures : List ℝ := [75, 77, 76, 78, 80, 81, 83, 82, 84, 86]

theorem mean_temperature_is_80_point_2 :
  (temperatures.sum / temperatures.length : ℝ) = 80.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_80_point_2_l2881_288186


namespace NUMINAMATH_CALUDE_dance_team_problem_l2881_288194

def student_heights : List ℝ := [161, 162, 162, 164, 165, 165, 165, 166, 166, 167, 168, 168, 170, 172, 172, 175]

def average_height : ℝ := 166.75

def group_A : List ℝ := [162, 165, 165, 166, 166]
def group_B : List ℝ := [161, 162, 164, 165, 175]

def preselected_heights : List ℝ := [168, 168, 172]

def median (l : List ℝ) : ℝ := sorry
def mode (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem dance_team_problem :
  (median student_heights = 166) ∧
  (mode student_heights = 165) ∧
  (variance group_A < variance group_B) ∧
  (∃ (h1 h2 : ℝ), h1 ∈ student_heights ∧ h2 ∈ student_heights ∧
    h1 = 170 ∧ h2 = 172 ∧
    variance (h1 :: h2 :: preselected_heights) < 32/9 ∧
    ∀ (x y : ℝ), x ∈ student_heights → y ∈ student_heights →
      variance (x :: y :: preselected_heights) < 32/9 →
      (x + y) / 2 ≤ (h1 + h2) / 2) :=
by sorry

#check dance_team_problem

end NUMINAMATH_CALUDE_dance_team_problem_l2881_288194


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l2881_288116

/-- Given an arithmetic sequence where a₁ = 1 and a₃ = 5, prove that a₁₀ = 19. -/
theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 1 = 1)  -- First term is 1
  (h2 : a 3 = 5)  -- Third term is 5
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- Definition of arithmetic sequence
  : a 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l2881_288116


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2881_288131

theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - 3)^2 = 8 → y = x + 4 → 
    ∀ x' y' : ℝ, (x' - a)^2 + (y' - 3)^2 < 8 → y' ≠ x' + 4) →
  a = 3 ∨ a = -5 :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2881_288131


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l2881_288118

theorem sandy_shopping_money (remaining_money : ℝ) (spent_percentage : ℝ) (h1 : remaining_money = 224) (h2 : spent_percentage = 0.3) :
  let initial_money := remaining_money / (1 - spent_percentage)
  initial_money = 320 := by
sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l2881_288118


namespace NUMINAMATH_CALUDE_cube_sum_eq_product_l2881_288122

theorem cube_sum_eq_product (m : ℕ) :
  (m = 1 ∨ m = 2 → ¬∃ (x y z : ℕ+), x^3 + y^3 + z^3 = m * x * y * z) ∧
  (m = 3 → ∀ (x y z : ℕ+), x^3 + y^3 + z^3 = 3 * x * y * z ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_eq_product_l2881_288122


namespace NUMINAMATH_CALUDE_triangle_properties_l2881_288165

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle)
  (h1 : Real.tan t.C = (Real.sin t.A + Real.sin t.B) / (Real.cos t.A + Real.cos t.B))
  (h2 : t.c = Real.sqrt 3) :
  t.C = π / 3 ∧ 3 < t.a^2 + t.b^2 ∧ t.a^2 + t.b^2 ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2881_288165


namespace NUMINAMATH_CALUDE_ceiling_minus_x_l2881_288124

theorem ceiling_minus_x (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ f : ℝ, 0 < f ∧ f < 1 ∧ ⌈x⌉ - x = 1 - f :=
by sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_l2881_288124


namespace NUMINAMATH_CALUDE_fifth_inequality_l2881_288111

theorem fifth_inequality (h1 : 1 / Real.sqrt 2 < 1)
  (h2 : 1 / Real.sqrt 2 + 1 / Real.sqrt 6 < Real.sqrt 2)
  (h3 : 1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 < Real.sqrt 3) :
  1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 + 1 / Real.sqrt 20 + 1 / Real.sqrt 30 < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_inequality_l2881_288111


namespace NUMINAMATH_CALUDE_measure_water_l2881_288146

theorem measure_water (a : ℤ) (h : -1562 ≤ a ∧ a ≤ 1562) :
  ∃ (b c d e f : ℤ), 
    (b ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (c ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (d ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (e ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (f ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (a = 625*b + 125*c + 25*d + 5*e + f) :=
by sorry

end NUMINAMATH_CALUDE_measure_water_l2881_288146


namespace NUMINAMATH_CALUDE_whole_milk_fat_percentage_l2881_288133

theorem whole_milk_fat_percentage :
  let reduced_fat_percentage : ℚ := 2
  let reduction_percentage : ℚ := 40
  let whole_milk_fat_percentage : ℚ := reduced_fat_percentage / (1 - reduction_percentage / 100)
  whole_milk_fat_percentage = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_whole_milk_fat_percentage_l2881_288133


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l2881_288168

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Two lines are parallel if and only if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem parallel_lines_slope (k : ℝ) :
  let line1 : Line := { slope := k, yIntercept := -7 }
  let line2 : Line := { slope := -3, yIntercept := 4 }
  parallel line1 line2 → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l2881_288168


namespace NUMINAMATH_CALUDE_smallest_n_with_abc_property_l2881_288129

def has_abc_property (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ∪ B = Finset.range n →
    (∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c) ∨
    (∃ (a b c : ℕ), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a * b = c)

theorem smallest_n_with_abc_property :
  (∀ k < 243, ¬ has_abc_property k) ∧ has_abc_property 243 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_abc_property_l2881_288129


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2881_288108

-- Define the polynomial
def p (z : ℂ) : ℂ := (z - 2) * (z^2 + 4*z + 10) * (z^2 + 6*z + 13)

-- Define the set of solutions
def solutions : Set ℂ := {z : ℂ | p z = 0}

-- Define the ellipse passing through the solutions
def E : Set ℂ := sorry

-- Define eccentricity
def eccentricity (E : Set ℂ) : ℝ := sorry

-- Theorem statement
theorem ellipse_eccentricity : eccentricity E = Real.sqrt (4/25) := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2881_288108


namespace NUMINAMATH_CALUDE_expression_simplification_l2881_288151

theorem expression_simplification 
  (x y z p q r : ℝ) 
  (hx : x ≠ p) 
  (hy : y ≠ q) 
  (hz : z ≠ r) 
  (hpq : p ≠ q) 
  (hqr : q ≠ r) 
  (hpr : p ≠ r) : 
  (2 * (x - p)) / (3 * (r - z)) * 
  (2 * (y - q)) / (3 * (p - x)) * 
  (2 * (z - r)) / (3 * (q - y)) = -8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2881_288151


namespace NUMINAMATH_CALUDE_base_10_to_base_7_157_l2881_288126

def base_7_digit (n : Nat) : Char :=
  if n < 7 then Char.ofNat (n + 48) else Char.ofNat (n + 55)

def to_base_7 (n : Nat) : List Char :=
  if n < 7 then [base_7_digit n]
  else base_7_digit (n % 7) :: to_base_7 (n / 7)

theorem base_10_to_base_7_157 :
  to_base_7 157 = ['3', '1', '3'] := by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_157_l2881_288126


namespace NUMINAMATH_CALUDE_cylinder_volume_with_square_section_l2881_288156

/-- Given a cylinder with a square axial section of area 4, its volume is 2π. -/
theorem cylinder_volume_with_square_section (r h : ℝ) : 
  r * h = 2 →  -- The axial section is a square
  r * r * h = 4 →  -- The area of the square is 4
  π * r * r * h = 2 * π :=  -- The volume of the cylinder is 2π
by
  sorry

#check cylinder_volume_with_square_section

end NUMINAMATH_CALUDE_cylinder_volume_with_square_section_l2881_288156


namespace NUMINAMATH_CALUDE_second_grade_sample_size_l2881_288177

/-- Given a total sample size and ratios for three grades, calculate the number of students to be drawn from a specific grade -/
def stratified_sample (total_sample : ℕ) (ratio1 ratio2 ratio3 : ℕ) (grade : ℕ) : ℕ :=
  let total_ratio := ratio1 + ratio2 + ratio3
  let grade_ratio := match grade with
    | 1 => ratio1
    | 2 => ratio2
    | 3 => ratio3
    | _ => 0
  (grade_ratio * total_sample) / total_ratio

/-- Theorem stating that for a sample size of 50 and ratios 3:3:4, the second grade should have 15 students -/
theorem second_grade_sample_size :
  stratified_sample 50 3 3 4 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_sample_size_l2881_288177


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2881_288113

theorem quadratic_minimum (x : ℝ) : 
  (∀ x, x^2 - 4*x + 3 ≥ -1) ∧ (∃ x, x^2 - 4*x + 3 = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2881_288113


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l2881_288199

theorem percentage_failed_hindi (failed_english : Real) (failed_both : Real) (passed_both : Real)
  (h1 : failed_english = 48)
  (h2 : failed_both = 27)
  (h3 : passed_both = 54) :
  failed_english + (100 - passed_both) - failed_both = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l2881_288199


namespace NUMINAMATH_CALUDE_six_students_three_colleges_l2881_288123

/-- The number of ways n students can apply to m colleges --/
def totalApplications (n m : ℕ) : ℕ := m^n

/-- The number of ways to apply to a subset of colleges --/
def subsetApplications (n k : ℕ) : ℕ := k^n

/-- The number of ways n students can apply to m colleges with each college receiving at least one application --/
def validApplications (n m : ℕ) : ℕ :=
  totalApplications n m - m * subsetApplications n (m-1) + (m.choose 2) * subsetApplications n (m-2)

theorem six_students_three_colleges :
  validApplications 6 3 = 540 := by sorry

end NUMINAMATH_CALUDE_six_students_three_colleges_l2881_288123


namespace NUMINAMATH_CALUDE_parallelogram_area_l2881_288190

/-- The area of a parallelogram with one angle of 100 degrees and two consecutive sides of lengths 10 inches and 20 inches is approximately 197.0 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) : 
  a = 10 → b = 20 → θ = 100 * π / 180 → 
  abs (a * b * Real.sin (π - θ) - 197.0) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2881_288190


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2881_288185

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The original number -/
def original_number : ℕ := 346000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation := {
  coefficient := 3.46
  exponent := 8
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2881_288185


namespace NUMINAMATH_CALUDE_rain_probability_l2881_288136

theorem rain_probability (p_friday p_saturday p_sunday : ℝ) 
  (h_friday : p_friday = 0.4)
  (h_saturday : p_saturday = 0.5)
  (h_sunday : p_sunday = 0.3)
  (h_independent : True) -- Assumption of independence
  : p_friday * p_saturday * p_sunday = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l2881_288136


namespace NUMINAMATH_CALUDE_power_equality_l2881_288121

theorem power_equality (x : ℝ) : (1/8 : ℝ) * 2^36 = 4^x → x = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2881_288121


namespace NUMINAMATH_CALUDE_valid_placements_correct_l2881_288135

/-- Represents a chess piece type -/
inductive ChessPiece
| Rook
| King
| Bishop
| Knight
| Queen

/-- Represents the size of the chessboard -/
def boardSize : Nat := 8

/-- Calculates the number of ways to place two identical pieces of the given type on an 8x8 chessboard such that they do not capture each other -/
def validPlacements (piece : ChessPiece) : Nat :=
  match piece with
  | ChessPiece.Rook => 1568
  | ChessPiece.King => 1806
  | ChessPiece.Bishop => 1972
  | ChessPiece.Knight => 1848
  | ChessPiece.Queen => 1980

/-- Theorem stating the correct number of valid placements for each piece type -/
theorem valid_placements_correct :
  (validPlacements ChessPiece.Rook = 1568) ∧
  (validPlacements ChessPiece.King = 1806) ∧
  (validPlacements ChessPiece.Bishop = 1972) ∧
  (validPlacements ChessPiece.Knight = 1848) ∧
  (validPlacements ChessPiece.Queen = 1980) :=
by sorry

end NUMINAMATH_CALUDE_valid_placements_correct_l2881_288135


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_right_triangle_l2881_288198

/-- In a right-angled triangle with legs a and b, hypotenuse c, and inscribed circle of radius r,
    the diameter of the inscribed circle is a + b - c. -/
theorem inscribed_circle_diameter_right_triangle 
  (a b c r : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0) 
  (h_inscribed : r = (a + b - c) / 2) : 
  2 * r = a + b - c := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_right_triangle_l2881_288198


namespace NUMINAMATH_CALUDE_hyperbola_equation_hyperbola_final_equation_l2881_288104

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, prove that if its eccentricity is 2
    and its asymptotes are tangent to the circle (x-a)² + y² = 3/4, then a = 1 and b = √3. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (2 : ℝ) = (Real.sqrt (a^2 + b^2)) / a  -- eccentricity is 2
  → (∃ (x y : ℝ), (y = (b/a) * x ∨ y = -(b/a) * x) ∧ (x - a)^2 + y^2 = 3/4)  -- asymptotes tangent to circle
  → a = 1 ∧ b = Real.sqrt 3 :=
by sorry

/-- The equation of the hyperbola is x² - y²/3 = 1. -/
theorem hyperbola_final_equation (x y : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (2 : ℝ) = (Real.sqrt (a^2 + b^2)) / a ∧
    (∃ (x' y' : ℝ), (y' = (b/a) * x' ∨ y' = -(b/a) * x') ∧ (x' - a)^2 + y'^2 = 3/4) ∧
    x^2 / a^2 - y^2 / b^2 = 1)
  → x^2 - y^2 / 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_hyperbola_final_equation_l2881_288104


namespace NUMINAMATH_CALUDE_min_sum_of_digits_prime_l2881_288173

def f (n : ℕ) : ℕ := n^2 - 69*n + 2250

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem min_sum_of_digits_prime :
  ∃ (p : ℕ), is_prime p ∧
    ∀ (q : ℕ), is_prime q →
      sum_of_digits (f (p^2 + 32)) ≤ sum_of_digits (f (q^2 + 32)) ∧
      p = 3 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_digits_prime_l2881_288173


namespace NUMINAMATH_CALUDE_apps_remaining_proof_l2881_288119

/-- Calculates the number of remaining apps after deletions -/
def remaining_apps (total : ℕ) (gaming : ℕ) (deleted_utility : ℕ) : ℕ :=
  total - gaming - deleted_utility

/-- Theorem: Given 12 total apps, 5 gaming apps, and deleting 3 utility apps,
    the number of remaining apps is 4 -/
theorem apps_remaining_proof :
  remaining_apps 12 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_apps_remaining_proof_l2881_288119


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l2881_288134

/-- Returns true if the given number is a palindrome in the specified base -/
def isPalindrome (n : ℕ) (base : ℕ) : Bool := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∃ (n : ℕ), n > 7 ∧
    isPalindrome n 3 = true ∧
    isPalindrome n 5 = true ∧
    (∀ (m : ℕ), m > 7 ∧ m < n →
      isPalindrome m 3 = false ∨ isPalindrome m 5 = false) ∧
    n = 26 := by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l2881_288134


namespace NUMINAMATH_CALUDE_sum_of_cyclic_equations_l2881_288171

theorem sum_of_cyclic_equations (x y z : ℝ) 
  (eq1 : x + y = 1) 
  (eq2 : y + z = 1) 
  (eq3 : z + x = 1) : 
  x + y + z = 3/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cyclic_equations_l2881_288171


namespace NUMINAMATH_CALUDE_total_shoes_l2881_288195

theorem total_shoes (brian_shoes : ℕ) (edward_shoes : ℕ) (jacob_shoes : ℕ) 
  (h1 : brian_shoes = 22)
  (h2 : edward_shoes = 3 * brian_shoes)
  (h3 : jacob_shoes = edward_shoes / 2) :
  brian_shoes + edward_shoes + jacob_shoes = 121 := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_l2881_288195


namespace NUMINAMATH_CALUDE_cos_eleven_pi_thirds_l2881_288141

theorem cos_eleven_pi_thirds : Real.cos (11 * Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_eleven_pi_thirds_l2881_288141


namespace NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l2881_288105

theorem joel_age_when_dad_twice_as_old (joel_current_age dad_current_age : ℕ) : 
  joel_current_age = 5 → 
  dad_current_age = 32 → 
  ∃ (years : ℕ), 
    dad_current_age + years = 2 * (joel_current_age + years) ∧ 
    joel_current_age + years = 27 := by
sorry

end NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l2881_288105


namespace NUMINAMATH_CALUDE_true_proposition_l2881_288101

/-- Proposition p: For any x ∈ ℝ, 2^x > x^2 -/
def p : Prop := ∀ x : ℝ, 2^x > x^2

/-- Proposition q: "ab > 1" is a sufficient but not necessary condition for "a > 1, b > 1" -/
def q : Prop := ∀ a b : ℝ, (a * b > 1 → (a > 1 ∧ b > 1)) ∧ ¬(∀ a b : ℝ, (a > 1 ∧ b > 1) → a * b > 1)

/-- The true proposition is ¬p ∧ ¬q -/
theorem true_proposition : ¬p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_true_proposition_l2881_288101


namespace NUMINAMATH_CALUDE_arrangement_count_l2881_288140

/-- Represents the number of people wearing each color -/
structure ColorCount where
  red : Nat
  yellow : Nat
  blue : Nat

/-- Represents the total number of people -/
def totalPeople (cc : ColorCount) : Nat :=
  cc.red + cc.yellow + cc.blue

/-- Calculates the number of valid arrangements -/
noncomputable def validArrangements (cc : ColorCount) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem arrangement_count (cc : ColorCount) : 
  cc.red = 2 → cc.yellow = 2 → cc.blue = 1 → 
  totalPeople cc = 5 → validArrangements cc = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l2881_288140


namespace NUMINAMATH_CALUDE_particle_position_at_5pm_l2881_288148

-- Define the particle's position as a function of time
def particle_position (t : ℝ) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem particle_position_at_5pm :
  -- Given conditions
  (particle_position 7 = (1, 2)) →
  (particle_position 9 = (3, -2)) →
  -- Constant speed along a straight line (slope remains constant)
  (∀ t₁ t₂ t₃ t₄ : ℝ, t₁ ≠ t₂ ∧ t₃ ≠ t₄ →
    (particle_position t₂).1 - (particle_position t₁).1 ≠ 0 →
    ((particle_position t₂).2 - (particle_position t₁).2) / ((particle_position t₂).1 - (particle_position t₁).1) =
    ((particle_position t₄).2 - (particle_position t₃).2) / ((particle_position t₄).1 - (particle_position t₃).1)) →
  -- Conclusion
  particle_position 17 = (11, -18) :=
by sorry

end NUMINAMATH_CALUDE_particle_position_at_5pm_l2881_288148


namespace NUMINAMATH_CALUDE_canoe_production_sum_l2881_288117

def geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * r^(n - 1)

def sum_geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  (a * (r^n - 1)) / (r - 1)

theorem canoe_production_sum :
  let a := 10
  let r := 3
  let n := 4
  sum_geometric_sequence a r n = 400 := by
sorry

end NUMINAMATH_CALUDE_canoe_production_sum_l2881_288117


namespace NUMINAMATH_CALUDE_unique_function_property_l2881_288114

theorem unique_function_property (f : ℕ → ℕ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ, f (m * n) = f m * f n)
  (h3 : ∀ n : ℕ, f (n + 1) > f n) :
  ∀ n : ℕ, f n = n :=
by sorry

end NUMINAMATH_CALUDE_unique_function_property_l2881_288114


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_angles_sum_l2881_288159

/-- A rectangular solid with a diagonal forming angles with edges. -/
structure RectangularSolid where
  /-- Length of the diagonal -/
  diagonal : ℝ
  /-- Angle between diagonal and first edge -/
  α : ℝ
  /-- Angle between diagonal and second edge -/
  β : ℝ
  /-- Angle between diagonal and third edge -/
  γ : ℝ
  /-- The angles are formed by the diagonal and edges of the rectangular solid -/
  angles_from_edges : True

/-- 
In a rectangular solid, if one of the diagonals forms angles α, β, and γ 
with the three edges emanating from one of the vertices, 
then cos²α + cos²β + cos²γ = 1.
-/
theorem rectangular_solid_diagonal_angles_sum 
  (rs : RectangularSolid) : Real.cos rs.α ^ 2 + Real.cos rs.β ^ 2 + Real.cos rs.γ ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_angles_sum_l2881_288159


namespace NUMINAMATH_CALUDE_area_relation_implies_parallel_diagonals_l2881_288196

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Check if two line segments are parallel -/
def parallel (A B C D : Point) : Prop := sorry

/-- Points A, B, C, D lie on the sides of quadrilateral PQRS -/
def pointsOnSides (PQRS : Quadrilateral) (A B C D : Point) : Prop := sorry

theorem area_relation_implies_parallel_diagonals 
  (PQRS : Quadrilateral) (A B C D : Point) :
  pointsOnSides PQRS A B C D →
  area PQRS = 2 * area ⟨A, B, C, D⟩ →
  parallel A C Q R ∨ parallel B D P Q := by
  sorry

end NUMINAMATH_CALUDE_area_relation_implies_parallel_diagonals_l2881_288196


namespace NUMINAMATH_CALUDE_sequence_sum_l2881_288137

/-- Given a geometric sequence a and an arithmetic sequence b,
    if 2a₃ - a₂a₄ = 0 and b₃ = a₃, then the sum of the first 5 terms of b is 10 -/
theorem sequence_sum (a b : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1)  -- a is geometric
  (h_arith : ∀ n : ℕ, b (n + 1) - b n = b 2 - b 1)  -- b is arithmetic
  (h_eq : 2 * a 3 - a 2 * a 4 = 0)
  (h_b3 : b 3 = a 3) :
  (b 1 + b 2 + b 3 + b 4 + b 5) = 10 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l2881_288137


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l2881_288139

/-- The function f(x) = (x+1)^2 -/
def f (x : ℝ) : ℝ := (x + 1)^2

/-- The theorem stating that the derivative of f(x) at x = 1 is 4 -/
theorem derivative_f_at_1 : 
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l2881_288139


namespace NUMINAMATH_CALUDE_cat_food_consumed_by_wednesday_l2881_288125

/-- Represents the days of the week -/
inductive Day : Type
| Monday : Day
| Tuesday : Day
| Wednesday : Day
| Thursday : Day
| Friday : Day
| Saturday : Day
| Sunday : Day

/-- Calculates the number of days until all cat food is consumed -/
def daysUntilFoodConsumed (morningPortion : ℚ) (eveningPortion : ℚ) (fullCans : ℕ) (leftoverCan : ℚ) (leftoverExpiry : Day) : Day :=
  sorry

/-- Theorem stating that all cat food will be consumed by Wednesday -/
theorem cat_food_consumed_by_wednesday :
  let morningPortion : ℚ := 1/4
  let eveningPortion : ℚ := 1/6
  let fullCans : ℕ := 10
  let leftoverCan : ℚ := 1/2
  let leftoverExpiry : Day := Day.Tuesday
  daysUntilFoodConsumed morningPortion eveningPortion fullCans leftoverCan leftoverExpiry = Day.Wednesday :=
by sorry

end NUMINAMATH_CALUDE_cat_food_consumed_by_wednesday_l2881_288125


namespace NUMINAMATH_CALUDE_milburg_adult_population_l2881_288188

theorem milburg_adult_population (total_population children : ℝ) 
  (h1 : total_population = 5256.0)
  (h2 : children = 2987.0) :
  total_population - children = 2269.0 := by
sorry

end NUMINAMATH_CALUDE_milburg_adult_population_l2881_288188


namespace NUMINAMATH_CALUDE_prime_square_plus_13_divisibility_l2881_288100

theorem prime_square_plus_13_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ k : ℕ, p^2 + 13 = 2*k + 2 := by
sorry

end NUMINAMATH_CALUDE_prime_square_plus_13_divisibility_l2881_288100


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_three_pi_halves_l2881_288154

theorem sum_of_solutions_is_three_pi_halves :
  ∃ (x₁ x₂ : Real),
    0 ≤ x₁ ∧ x₁ ≤ 2 * Real.pi ∧
    0 ≤ x₂ ∧ x₂ ≤ 2 * Real.pi ∧
    (1 / Real.sin x₁ + 1 / Real.cos x₁ = 4) ∧
    (1 / Real.sin x₂ + 1 / Real.cos x₂ = 4) ∧
    x₁ + x₂ = 3 * Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_three_pi_halves_l2881_288154


namespace NUMINAMATH_CALUDE_fishmonger_sales_l2881_288179

/-- Given a first week's sales and a multiplier for the second week's sales,
    calculate the total sales over two weeks. -/
def totalSales (firstWeekSales secondWeekMultiplier : ℕ) : ℕ :=
  firstWeekSales + firstWeekSales * secondWeekMultiplier

/-- Theorem stating that given the specific conditions of the problem,
    the total sales over two weeks is 200 kg. -/
theorem fishmonger_sales : totalSales 50 3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_fishmonger_sales_l2881_288179


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l2881_288150

/-- The volume of a cylinder obtained by rotating a rectangle about its longer side -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (length_positive : 0 < length) (width_positive : 0 < width) (length_longer : width ≤ length) :
  let radius := width / 2
  let height := length
  let volume := π * radius^2 * height
  (length = 10 ∧ width = 8) → volume = 160 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l2881_288150


namespace NUMINAMATH_CALUDE_swimming_speed_is_15_l2881_288181

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  /-- The man's swimming speed in still water (km/h) -/
  v : ℝ
  /-- The speed of the stream (km/h) -/
  s : ℝ
  /-- The time it takes to swim downstream (hours) -/
  t : ℝ
  /-- Assertion that it takes twice as long to swim upstream -/
  upstream_time : (v - s) * (2 * t) = (v + s) * t
  /-- The speed of the stream is 5 km/h -/
  stream_speed : s = 5

/-- Theorem stating that the man's swimming speed in still water is 15 km/h -/
theorem swimming_speed_is_15 (scenario : SwimmingScenario) : scenario.v = 15 := by
  sorry

end NUMINAMATH_CALUDE_swimming_speed_is_15_l2881_288181


namespace NUMINAMATH_CALUDE_sector_central_angle_invariant_l2881_288106

/-- Theorem: If both the radius and arc length of a circular sector are doubled, then the central angle of the sector remains unchanged. -/
theorem sector_central_angle_invariant 
  (r₁ r₂ l₁ l₂ θ₁ θ₂ : Real) 
  (h1 : r₂ = 2 * r₁) 
  (h2 : l₂ = 2 * l₁) 
  (h3 : θ₁ = l₁ / r₁) 
  (h4 : θ₂ = l₂ / r₂) : 
  θ₁ = θ₂ := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_invariant_l2881_288106


namespace NUMINAMATH_CALUDE_book_difference_l2881_288191

def initial_books : ℕ := 28
def jungkook_bought : ℕ := 18
def seokjin_bought : ℕ := 11

theorem book_difference : 
  (initial_books + jungkook_bought) - (initial_books + seokjin_bought) = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_difference_l2881_288191


namespace NUMINAMATH_CALUDE_sum_of_seven_odd_integers_remainder_l2881_288180

def consecutive_odd_integers (start : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + 2 * i)

theorem sum_of_seven_odd_integers_remainder (start : ℕ) (h : start = 12095) :
  (consecutive_odd_integers start 7).sum % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_odd_integers_remainder_l2881_288180


namespace NUMINAMATH_CALUDE_sum_and_decimal_shift_l2881_288170

theorem sum_and_decimal_shift (A B : ℝ) (h1 : A + B = 13.2) (h2 : 10 * A = B) : A = 1.2 ∧ B = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_decimal_shift_l2881_288170


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2881_288152

theorem complex_modulus_problem : 
  Complex.abs ((3 + Complex.I) / (1 - Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2881_288152


namespace NUMINAMATH_CALUDE_solve_for_q_l2881_288192

theorem solve_for_q (n m q : ℚ) 
  (h1 : 7/9 = n/108)
  (h2 : 7/9 = (m+n)/126)
  (h3 : 7/9 = (q-m)/162) : 
  q = 140 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l2881_288192


namespace NUMINAMATH_CALUDE_sandy_grew_eight_carrots_l2881_288182

/-- The number of carrots Sandy grew -/
def sandys_carrots : ℕ := sorry

/-- The number of carrots Mary grew -/
def marys_carrots : ℕ := 6

/-- The total number of carrots grown by Sandy and Mary -/
def total_carrots : ℕ := 14

/-- Theorem stating that Sandy grew 8 carrots -/
theorem sandy_grew_eight_carrots : sandys_carrots = 8 := by
  sorry

end NUMINAMATH_CALUDE_sandy_grew_eight_carrots_l2881_288182


namespace NUMINAMATH_CALUDE_fred_has_eighteen_balloons_l2881_288130

/-- The number of blue balloons Sally has -/
def sally_balloons : ℕ := 6

/-- The factor by which Fred has more balloons than Sally -/
def fred_factor : ℕ := 3

/-- The number of blue balloons Fred has -/
def fred_balloons : ℕ := sally_balloons * fred_factor

/-- Theorem stating that Fred has 18 blue balloons -/
theorem fred_has_eighteen_balloons : fred_balloons = 18 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_eighteen_balloons_l2881_288130


namespace NUMINAMATH_CALUDE_no_solution_for_divisibility_l2881_288183

theorem no_solution_for_divisibility (n : ℕ) : n ≥ 1 → ¬(9 ∣ (7^n + n^3)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_divisibility_l2881_288183


namespace NUMINAMATH_CALUDE_cab_journey_time_l2881_288145

/-- Given a cab walking at 5/6 of its usual speed and arriving 12 minutes late,
    prove that its usual time to cover the journey is 1 hour. -/
theorem cab_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
    (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (usual_speed * usual_time = (5/6 * usual_speed) * (usual_time + 1/5)) → 
  usual_time = 1 := by
  sorry

#check cab_journey_time

end NUMINAMATH_CALUDE_cab_journey_time_l2881_288145


namespace NUMINAMATH_CALUDE_lens_savings_l2881_288110

/-- The price of the more expensive lens before discount -/
def original_price : ℝ := 300

/-- The discount rate as a decimal -/
def discount_rate : ℝ := 0.20

/-- The price of the cheaper lens -/
def cheaper_price : ℝ := 220

/-- The discounted price of the more expensive lens -/
def discounted_price : ℝ := original_price * (1 - discount_rate)

/-- The amount saved by buying the cheaper lens -/
def savings : ℝ := discounted_price - cheaper_price

theorem lens_savings : savings = 20 := by
  sorry

end NUMINAMATH_CALUDE_lens_savings_l2881_288110


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_3_range_of_a_for_nonempty_solution_l2881_288164

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x - 2|

-- Theorem 1: Solution set of f(x) < 3
theorem solution_set_f_less_than_3 :
  {x : ℝ | f x < 3} = {x : ℝ | -1/2 < x ∧ x < 5/2} :=
sorry

-- Theorem 2: Range of a for non-empty solution set
theorem range_of_a_for_nonempty_solution (a : ℝ) :
  (∃ x : ℝ, f x < a) → a > 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_3_range_of_a_for_nonempty_solution_l2881_288164


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l2881_288103

/-- A polynomial of the form x^3 - 2ax^2 + bx - 2a -/
def cubic_polynomial (a b : ℝ) (x : ℝ) : ℝ := x^3 - 2*a*x^2 + b*x - 2*a

/-- The condition that a polynomial has all real roots -/
def has_all_real_roots (p : ℝ → ℝ) : Prop :=
  ∃ r s t : ℝ, ∀ x : ℝ, p x = (x - r) * (x - s) * (x - t)

/-- The theorem stating the relationship between a and b for the given polynomial -/
theorem cubic_polynomial_roots (a b : ℝ) :
  (a > 0 ∧ a = 3 * Real.sqrt 3 / 2 ∧ b = 81 / 4) ↔
  (has_all_real_roots (cubic_polynomial a b) ∧
   ∀ a' > 0, has_all_real_roots (cubic_polynomial a' b) → a ≤ a') :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l2881_288103


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l2881_288176

theorem smallest_resolvable_debt (pig_value goat_value : ℕ) 
  (h_pig : pig_value = 400) (h_goat : goat_value = 250) :
  ∃ (D : ℕ), D > 0 ∧ 
  (∃ (p g : ℤ), D = pig_value * p + goat_value * g) ∧
  (∀ (D' : ℕ), D' > 0 → 
    (∃ (p' g' : ℤ), D' = pig_value * p' + goat_value * g') → 
    D ≤ D') :=
by sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l2881_288176


namespace NUMINAMATH_CALUDE_work_completion_time_b_l2881_288115

/-- The number of days it takes for worker b to complete a work alone,
    given that workers a and b together can finish the work in 16 days,
    and worker a alone can do the same work in 32 days. -/
theorem work_completion_time_b (work_rate_a_and_b : ℚ) (work_rate_a : ℚ) :
  work_rate_a_and_b = 1 / 16 →
  work_rate_a = 1 / 32 →
  (1 : ℚ) / (work_rate_a_and_b - work_rate_a) = 32 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_b_l2881_288115


namespace NUMINAMATH_CALUDE_cost_difference_white_brown_socks_l2881_288107

-- Define the cost of two white socks in cents
def cost_two_white_socks : ℕ := 45

-- Define the cost of 15 brown socks in cents
def cost_fifteen_brown_socks : ℕ := 300

-- Define the number of brown socks
def num_brown_socks : ℕ := 15

-- Theorem to prove
theorem cost_difference_white_brown_socks : 
  cost_two_white_socks - (cost_fifteen_brown_socks / num_brown_socks) = 25 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_white_brown_socks_l2881_288107


namespace NUMINAMATH_CALUDE_square_perimeter_l2881_288160

theorem square_perimeter (total_area common_area circle_area : ℝ) 
  (h1 : total_area = 329)
  (h2 : common_area = 101)
  (h3 : circle_area = 234) :
  let square_area := total_area + common_area - circle_area
  let side_length := Real.sqrt square_area
  4 * side_length = 56 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_l2881_288160


namespace NUMINAMATH_CALUDE_common_roots_product_l2881_288143

/-- Given two cubic equations with two common roots, prove that the product of these common roots is 10 * ∛2 -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (u v w t : ℝ), 
    (u^3 + C*u^2 + 20 = 0) ∧ 
    (v^3 + C*v^2 + 20 = 0) ∧ 
    (w^3 + C*w^2 + 20 = 0) ∧
    (u^3 + D*u + 100 = 0) ∧ 
    (v^3 + D*v + 100 = 0) ∧ 
    (t^3 + D*t + 100 = 0) ∧
    (u ≠ v) ∧ 
    (u * v = 10 * (2 : ℝ)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_common_roots_product_l2881_288143


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2881_288147

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + 3 * a 3 + a 15 = 10) :
  a 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2881_288147


namespace NUMINAMATH_CALUDE_stating_volume_division_ratio_l2881_288127

/-- Represents a truncated triangular pyramid -/
structure TruncatedTriangularPyramid where
  -- The ratio of corresponding sides of the upper and lower bases
  base_ratio : ℝ
  -- Assume base_ratio > 0
  base_ratio_pos : base_ratio > 0

/-- 
  Theorem stating that for a truncated triangular pyramid with base ratio 1:2,
  a plane drawn through a side of the upper base parallel to the opposite lateral edge
  divides the volume in the ratio 3:4
-/
theorem volume_division_ratio 
  (pyramid : TruncatedTriangularPyramid) 
  (h_ratio : pyramid.base_ratio = 1/2) :
  ∃ (v1 v2 : ℝ), v1 > 0 ∧ v2 > 0 ∧ v1 / v2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_stating_volume_division_ratio_l2881_288127


namespace NUMINAMATH_CALUDE_sphere_tangency_loci_l2881_288167

/-- Given a sphere of radius R touching a plane, and spheres of radius r
    touching both the given sphere and the plane, this theorem proves the radii
    of the circles formed by the centers and points of tangency of the r-radius spheres. -/
theorem sphere_tangency_loci (R r : ℝ) (h : R > 0) (h' : r > 0) :
  ∃ (center_locus tangent_plane_locus tangent_sphere_locus : ℝ),
    center_locus = 2 * Real.sqrt (R * r) ∧
    tangent_plane_locus = 2 * Real.sqrt (R * r) ∧
    tangent_sphere_locus = (2 * R * Real.sqrt (R * r)) / (R + r) :=
sorry

end NUMINAMATH_CALUDE_sphere_tangency_loci_l2881_288167


namespace NUMINAMATH_CALUDE_circle_radius_l2881_288197

theorem circle_radius (C : ℝ) (r : ℝ) (h : C = 72 * Real.pi) : C = 2 * Real.pi * r → r = 36 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2881_288197


namespace NUMINAMATH_CALUDE_real_part_of_z_l2881_288187

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2881_288187


namespace NUMINAMATH_CALUDE_probability_at_least_two_same_l2881_288142

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The probability of at least two dice showing the same number when rolling 8 fair 8-sided dice -/
theorem probability_at_least_two_same : 
  (1 - (Nat.factorial num_dice : ℚ) / (num_sides ^ num_dice : ℚ)) = 16736996 / 16777216 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_same_l2881_288142


namespace NUMINAMATH_CALUDE_candy_distribution_l2881_288169

theorem candy_distribution (total_candy : ℕ) (candy_per_bag : ℕ) (num_bags : ℕ) : 
  total_candy = 42 → 
  candy_per_bag = 21 → 
  total_candy = num_bags * candy_per_bag → 
  num_bags = 2 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2881_288169


namespace NUMINAMATH_CALUDE_m_intersect_n_equals_m_l2881_288193

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | ∃ y, y = Real.log (4 - x^2)}

-- Theorem statement
theorem m_intersect_n_equals_m : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_m_intersect_n_equals_m_l2881_288193


namespace NUMINAMATH_CALUDE_fabric_needed_l2881_288102

-- Define the constants
def yards_per_dress : ℝ := 5.5
def num_dresses : ℕ := 4
def fabric_on_hand : ℝ := 7
def feet_per_yard : ℝ := 3

-- State the theorem
theorem fabric_needed : 
  ∃ (additional_fabric : ℝ), 
    additional_fabric = num_dresses * (yards_per_dress * feet_per_yard) - fabric_on_hand ∧
    additional_fabric = 59 := by
  sorry

end NUMINAMATH_CALUDE_fabric_needed_l2881_288102


namespace NUMINAMATH_CALUDE_total_fish_l2881_288174

theorem total_fish (lilly_fish rosy_fish tom_fish : ℕ) 
  (h1 : lilly_fish = 10)
  (h2 : rosy_fish = 14)
  (h3 : tom_fish = 8) :
  lilly_fish + rosy_fish + tom_fish = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_l2881_288174


namespace NUMINAMATH_CALUDE_marks_lost_is_one_l2881_288138

/-- Represents an examination with given parameters -/
structure Examination where
  total_questions : ℕ
  marks_per_correct : ℕ
  total_score : ℕ
  correct_answers : ℕ

/-- Calculates the marks lost per wrong answer in an examination -/
def marks_lost_per_wrong (exam : Examination) : ℚ :=
  let wrong_answers := exam.total_questions - exam.correct_answers
  let total_marks_for_correct := exam.marks_per_correct * exam.correct_answers
  let total_marks_lost := total_marks_for_correct - exam.total_score
  total_marks_lost / wrong_answers

/-- Theorem stating that for the given examination parameters, 
    the marks lost per wrong answer is 1 -/
theorem marks_lost_is_one (exam : Examination) 
  (h1 : exam.total_questions = 60)
  (h2 : exam.marks_per_correct = 4)
  (h3 : exam.total_score = 150)
  (h4 : exam.correct_answers = 42) :
  marks_lost_per_wrong exam = 1 := by
  sorry

#eval marks_lost_per_wrong { 
  total_questions := 60, 
  marks_per_correct := 4, 
  total_score := 150, 
  correct_answers := 42 
}

end NUMINAMATH_CALUDE_marks_lost_is_one_l2881_288138


namespace NUMINAMATH_CALUDE_sams_original_portion_l2881_288189

theorem sams_original_portion (s j r : ℝ) :
  s + j + r = 1200 →
  s - 200 + 3 * j + 3 * r = 1800 →
  s = 800 :=
by sorry

end NUMINAMATH_CALUDE_sams_original_portion_l2881_288189


namespace NUMINAMATH_CALUDE_least_days_same_date_l2881_288132

/-- A calendar date represented by a day and a month -/
structure CalendarDate where
  day : Nat
  month : Nat

/-- Function to move a given number of days forward or backward from a date -/
def moveDays (date : CalendarDate) (days : Int) : CalendarDate :=
  sorry

/-- Predicate to check if two dates have the same day of the month -/
def sameDayOfMonth (date1 date2 : CalendarDate) : Prop :=
  date1.day = date2.day

theorem least_days_same_date :
  ∃ k : Nat, k > 0 ∧
    (∀ date : CalendarDate, sameDayOfMonth (moveDays date k) (moveDays date (-k))) ∧
    (∀ j : Nat, 0 < j → j < k →
      ∃ date : CalendarDate, ¬sameDayOfMonth (moveDays date j) (moveDays date (-j))) ∧
    k = 14 :=
  sorry

end NUMINAMATH_CALUDE_least_days_same_date_l2881_288132


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2881_288109

theorem inequality_solution_set (x : ℝ) : 9 > -3 * x ↔ x > -3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2881_288109


namespace NUMINAMATH_CALUDE_circus_tickets_cost_l2881_288120

/-- Given the cost per ticket and the number of tickets bought, 
    calculate the total amount spent on tickets. -/
def total_spent (cost_per_ticket : ℕ) (num_tickets : ℕ) : ℕ :=
  cost_per_ticket * num_tickets

/-- Theorem: If each ticket costs 44 dollars and 7 tickets are bought,
    the total amount spent is 308 dollars. -/
theorem circus_tickets_cost :
  let cost_per_ticket : ℕ := 44
  let num_tickets : ℕ := 7
  total_spent cost_per_ticket num_tickets = 308 := by
  sorry

end NUMINAMATH_CALUDE_circus_tickets_cost_l2881_288120


namespace NUMINAMATH_CALUDE_ball_picking_probabilities_l2881_288166

/-- Represents the probability of selecting ball 3 using strategy 1 -/
def P₁ : ℚ := 1/3

/-- Represents the probability of selecting ball 3 using strategy 2 -/
def P₂ : ℚ := 1/2

/-- Represents the probability of selecting ball 3 using strategy 3 -/
def P₃ : ℚ := 2/3

/-- Theorem stating the relationships between P₁, P₂, and P₃ -/
theorem ball_picking_probabilities : P₁ < P₂ ∧ P₁ < P₃ ∧ 2 * P₁ = P₃ := by
  sorry

end NUMINAMATH_CALUDE_ball_picking_probabilities_l2881_288166


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2881_288163

/-- Given a cube with side length a, prove that if we form a rectangular solid
    by increasing one edge by 2, decreasing another by 1, and leaving the third unchanged,
    and if the volume of this new solid is 14 more than the original cube,
    then the volume of the original cube is 64. -/
theorem cube_volume_problem (a : ℕ) : 
  (a + 2) * (a - 1) * a = a^3 + 14 → a^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2881_288163


namespace NUMINAMATH_CALUDE_pandas_weekly_bamboo_consumption_l2881_288175

/-- The amount of bamboo eaten by pandas in a week -/
def bamboo_eaten_in_week (adult_daily : ℕ) (baby_daily : ℕ) : ℕ :=
  (adult_daily + baby_daily) * 7

/-- Theorem: The total amount of bamboo eaten by an adult panda and a baby panda in a week -/
theorem pandas_weekly_bamboo_consumption :
  bamboo_eaten_in_week 138 50 = 1316 := by
  sorry

end NUMINAMATH_CALUDE_pandas_weekly_bamboo_consumption_l2881_288175


namespace NUMINAMATH_CALUDE_triangle_tan_A_l2881_288178

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a/b = (b + √3c)/a and sin C = 2√3 sin B, then tan A = √3/3 -/
theorem triangle_tan_A (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π → B > 0 → B < π → C > 0 → C < π →
  A + B + C = π →
  (a / b = (b + Real.sqrt 3 * c) / a) →
  (Real.sin C = 2 * Real.sqrt 3 * Real.sin B) →
  Real.tan A = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tan_A_l2881_288178


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l2881_288153

theorem imaginary_part_of_product : Complex.im ((2 - Complex.I) * (4 + Complex.I)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l2881_288153


namespace NUMINAMATH_CALUDE_unique_distribution_l2881_288158

structure Desserts where
  coconut : Nat
  meringue : Nat
  caramel : Nat

def total_desserts (d : Desserts) : Nat :=
  d.coconut + d.meringue + d.caramel

def is_valid_distribution (d : Desserts) : Prop :=
  total_desserts d = 10 ∧
  d.coconut < d.meringue ∧
  d.meringue < d.caramel ∧
  d.caramel ≥ 6 ∧
  (d.coconut + d.meringue ≥ 3)

theorem unique_distribution :
  ∃! d : Desserts, is_valid_distribution d ∧ d.coconut = 1 ∧ d.meringue = 2 ∧ d.caramel = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_distribution_l2881_288158


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l2881_288112

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  lastInningRuns : ℕ
  averageIncrease : ℚ

/-- Calculates the average score of a batsman -/
def calculateAverage (b : Batsman) : ℚ :=
  (b.totalRuns : ℚ) / b.innings

/-- Theorem stating the batsman's average after the 17th inning -/
theorem batsman_average_after_17th_inning (b : Batsman)
  (h1 : b.innings = 17)
  (h2 : b.lastInningRuns = 90)
  (h3 : b.averageIncrease = 3)
  (h4 : calculateAverage b = calculateAverage { b with
    innings := b.innings - 1,
    totalRuns := b.totalRuns - b.lastInningRuns
  } + b.averageIncrease) :
  calculateAverage b = 42 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l2881_288112


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2881_288144

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^4 + 14 * X^3 - 55 * X^2 - 73 * X + 65 = 
  (X^2 + 8 * X - 6) * q + (-477 * X + 323) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2881_288144
