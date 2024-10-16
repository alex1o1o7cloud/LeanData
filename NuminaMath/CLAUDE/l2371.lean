import Mathlib

namespace NUMINAMATH_CALUDE_machines_working_first_scenario_l2371_237186

/-- The number of machines working in the first scenario -/
def num_machines : ℕ := 8

/-- The time taken in the first scenario (in hours) -/
def time_first_scenario : ℕ := 6

/-- The number of machines in the second scenario -/
def num_machines_second : ℕ := 6

/-- The time taken in the second scenario (in hours) -/
def time_second : ℕ := 8

/-- The total work done in one job lot -/
def total_work : ℕ := 1

theorem machines_working_first_scenario :
  num_machines * time_first_scenario = num_machines_second * time_second :=
by sorry

end NUMINAMATH_CALUDE_machines_working_first_scenario_l2371_237186


namespace NUMINAMATH_CALUDE_problem_solution_l2371_237136

theorem problem_solution : ∀ M N X : ℕ,
  M = 2098 / 2 →
  N = M * 2 →
  X = M + N →
  X = 3147 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2371_237136


namespace NUMINAMATH_CALUDE_fixed_point_on_AB_l2371_237142

-- Define the circle C
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line L
def Line (x y : ℝ) : Prop := x + y = 9

-- Define a point P on line L
def PointOnLine (P : ℝ × ℝ) : Prop := Line P.1 P.2

-- Define tangent line from P to circle C
def TangentLine (P A : ℝ × ℝ) : Prop :=
  Circle A.1 A.2 ∧ (∃ t : ℝ, A.1 = P.1 + t * (A.2 - P.2) ∧ A.2 = P.2 - t * (A.1 - P.1))

-- Theorem statement
theorem fixed_point_on_AB (P A B : ℝ × ℝ) :
  PointOnLine P →
  TangentLine P A →
  TangentLine P B →
  A ≠ B →
  ∃ t : ℝ, (4/9 : ℝ) = A.1 + t * (B.1 - A.1) ∧ (8/9 : ℝ) = A.2 + t * (B.2 - A.2) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_on_AB_l2371_237142


namespace NUMINAMATH_CALUDE_nonnegative_sum_one_inequality_l2371_237151

theorem nonnegative_sum_one_inequality (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum_one : x + y + z = 1) : 
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ 
  x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_sum_one_inequality_l2371_237151


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l2371_237185

/-- Represents a triangular figure constructed with toothpicks -/
structure TriangularFigure where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : TriangularFigure) : ℕ :=
  figure.upward_triangles

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_removal (figure : TriangularFigure) 
  (h1 : figure.total_toothpicks = 45)
  (h2 : figure.upward_triangles = 15)
  (h3 : figure.downward_triangles = 10) :
  min_toothpicks_to_remove figure = 15 := by
  sorry

#check min_toothpicks_removal

end NUMINAMATH_CALUDE_min_toothpicks_removal_l2371_237185


namespace NUMINAMATH_CALUDE_complex_power_modulus_l2371_237131

theorem complex_power_modulus : Complex.abs ((2 + Complex.I) ^ 6) = 125 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l2371_237131


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l2371_237147

/-- Given a quadratic equation 4x^2 - 8x - 320 = 0, prove that when transformed
    into the form (x+p)^2 = q by completing the square, the value of q is 81. -/
theorem complete_square_quadratic :
  ∃ (p : ℝ), ∀ (x : ℝ),
    (4 * x^2 - 8 * x - 320 = 0) ↔ ((x + p)^2 = 81) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l2371_237147


namespace NUMINAMATH_CALUDE_paint_intensity_problem_l2371_237109

/-- Given an original paint intensity of 50%, a new paint intensity of 30%,
    and 2/3 of the original paint replaced, prove that the intensity of
    the added paint solution is 20%. -/
theorem paint_intensity_problem (original_intensity new_intensity fraction_replaced : ℚ)
    (h1 : original_intensity = 50/100)
    (h2 : new_intensity = 30/100)
    (h3 : fraction_replaced = 2/3) :
    let added_intensity := (new_intensity - original_intensity * (1 - fraction_replaced)) / fraction_replaced
    added_intensity = 20/100 := by
  sorry

end NUMINAMATH_CALUDE_paint_intensity_problem_l2371_237109


namespace NUMINAMATH_CALUDE_no_solutions_exist_l2371_237179

/-- P(n) denotes the greatest prime factor of n -/
def greatest_prime_factor (n : ℕ) : ℕ :=
  sorry

/-- The theorem states that there are no positive integers n > 1 satisfying the given conditions -/
theorem no_solutions_exist : ¬ ∃ (n : ℕ), n > 1 ∧ 
  (greatest_prime_factor n = Nat.sqrt n) ∧ 
  (greatest_prime_factor (n + 60) = Nat.sqrt (n + 60)) :=
sorry

end NUMINAMATH_CALUDE_no_solutions_exist_l2371_237179


namespace NUMINAMATH_CALUDE_functional_equation_result_l2371_237190

theorem functional_equation_result (g : ℝ → ℝ) 
  (h₁ : ∀ c d : ℝ, c^2 * g d = d^2 * g c) 
  (h₂ : g 4 ≠ 0) : 
  (g 7 - g 3) / g 4 = 5/2 := by sorry

end NUMINAMATH_CALUDE_functional_equation_result_l2371_237190


namespace NUMINAMATH_CALUDE_set_union_problem_l2371_237187

theorem set_union_problem (M N : Set ℕ) (x : ℕ) :
  M = {0, x} →
  N = {1, 2} →
  M ∩ N = {1} →
  M ∪ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l2371_237187


namespace NUMINAMATH_CALUDE_two_circles_in_triangle_l2371_237154

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle touches two sides of a triangle --/
def touchesTwoSides (c : Circle) (t : Triangle) : Prop := sorry

/-- Predicate to check if two circles touch each other --/
def circlesAreEqual (c1 c2 : Circle) : Prop := c1.radius = c2.radius

/-- Predicate to check if two circles touch each other --/
def circlesAreInscribed (c1 c2 : Circle) (t : Triangle) : Prop :=
  touchesTwoSides c1 t ∧ touchesTwoSides c2 t ∧ circlesAreEqual c1 c2

/-- Theorem stating that two equal circles can be inscribed in a triangle --/
theorem two_circles_in_triangle (t : Triangle) :
  ∃ c1 c2 : Circle, circlesAreInscribed c1 c2 t := by sorry

end NUMINAMATH_CALUDE_two_circles_in_triangle_l2371_237154


namespace NUMINAMATH_CALUDE_distance_knoxville_los_angeles_l2371_237127

theorem distance_knoxville_los_angeles : 
  let los_angeles : ℂ := 0
  let knoxville : ℂ := 900 + 1200 * I
  Complex.abs (knoxville - los_angeles) = 1500 := by
  sorry

end NUMINAMATH_CALUDE_distance_knoxville_los_angeles_l2371_237127


namespace NUMINAMATH_CALUDE_seating_arrangements_l2371_237158

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where two specific people sit next to each other -/
def adjacentArrangements (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * 2

/-- The number of ways to arrange n people in a row where two specific people cannot sit next to each other -/
def nonAdjacentArrangements (n : ℕ) : ℕ := totalArrangements n - adjacentArrangements n

theorem seating_arrangements :
  nonAdjacentArrangements 7 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2371_237158


namespace NUMINAMATH_CALUDE_digit_sum_problem_l2371_237195

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number only uses specified digits -/
def uses_specified_digits (n : ℕ) : Prop := sorry

theorem digit_sum_problem (M : ℕ) 
  (h_even : Even M)
  (h_digits : uses_specified_digits M)
  (h_double : sum_of_digits (2 * M) = 35)
  (h_half : sum_of_digits (M / 2) = 29) :
  sum_of_digits M = 31 := by sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l2371_237195


namespace NUMINAMATH_CALUDE_reciprocal_sum_property_l2371_237128

theorem reciprocal_sum_property (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : 1 / a + 1 / b + 1 / c = 1 / (a + b + c)) :
  ∀ n : ℤ, (1 / a^n + 1 / b^n + 1 / c^n = 1 / (a^n + b^n + c^n)) ↔ ∃ k : ℕ, n = 2 * k - 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_property_l2371_237128


namespace NUMINAMATH_CALUDE_max_lateral_surface_area_l2371_237170

/-- A right prism ABCD-A₁B₁C₁D₁ inscribed in a sphere O -/
structure InscribedPrism where
  /-- The base edge length of the prism -/
  a : ℝ
  /-- The height of the prism -/
  h : ℝ
  /-- The radius of the sphere -/
  r : ℝ
  /-- The surface area of the sphere is 12π -/
  sphere_area : 4 * π * r^2 = 12 * π
  /-- The prism is inscribed in the sphere -/
  inscribed : 2 * a^2 + h^2 = 4 * r^2

/-- The lateral surface area of the prism -/
def lateralSurfaceArea (p : InscribedPrism) : ℝ := 4 * p.a * p.h

/-- The theorem stating the maximum lateral surface area of the inscribed prism -/
theorem max_lateral_surface_area (p : InscribedPrism) : 
  lateralSurfaceArea p ≤ 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_lateral_surface_area_l2371_237170


namespace NUMINAMATH_CALUDE_card_game_guarantee_l2371_237139

/-- Represents a cell on the 4x9 board --/
structure Cell :=
  (row : Fin 4)
  (col : Fin 9)

/-- Represents a pair of cells --/
structure CellPair :=
  (cell1 : Cell)
  (cell2 : Cell)

/-- Represents the state of the board --/
def Board := Fin 4 → Fin 9 → Bool

/-- A valid pairing of cells --/
def ValidPairing (board : Board) (pairs : List CellPair) : Prop :=
  ∀ p ∈ pairs,
    (board p.cell1.row p.cell1.col ≠ board p.cell2.row p.cell2.col) ∧
    ((p.cell1.row = p.cell2.row) ∨ (p.cell1.col = p.cell2.col))

/-- The main theorem --/
theorem card_game_guarantee (board : Board) :
  (∃ black_count : ℕ, black_count = 18 ∧ 
    (∀ r : Fin 4, ∀ c : Fin 9, (board r c = true) → black_count = black_count - 1)) →
  ∃ pairs : List CellPair, ValidPairing board pairs ∧ pairs.length ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_card_game_guarantee_l2371_237139


namespace NUMINAMATH_CALUDE_min_value_of_function_l2371_237164

theorem min_value_of_function (x : ℝ) (h : x < 0) :
  -x - 2/x ≥ 2 * Real.sqrt 2 ∧
  (-(-Real.sqrt 2) - 2/(-Real.sqrt 2) = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2371_237164


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l2371_237115

theorem binomial_square_coefficient (a : ℝ) : 
  (∃ r s : ℝ, (r * x + s)^2 = a * x^2 + 18 * x + 9) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l2371_237115


namespace NUMINAMATH_CALUDE_unique_number_l2371_237199

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if a natural number has exactly two prime factors -/
def hasTwoPrimeFactors (n : ℕ) : Prop :=
  ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ p ≠ q ∧ n = p * q

/-- A function that checks if a number doesn't contain the digit 7 -/
def noSeven (n : ℕ) : Prop :=
  ∀ d : ℕ, d < n → (n / (10^d)) % 10 ≠ 7

theorem unique_number : 
  ∃! n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧  -- four-digit number
    noSeven n ∧             -- doesn't contain 7
    hasTwoPrimeFactors n ∧  -- product of exactly two primes
    ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ n = p * q ∧ q = p + 4 ∧  -- prime factors differ by 4
    n = 2021                -- the number is 2021
  := by sorry

end NUMINAMATH_CALUDE_unique_number_l2371_237199


namespace NUMINAMATH_CALUDE_equation_solution_l2371_237114

theorem equation_solution (m n : ℝ) : 21 * (m + n) + 21 = 21 * (-m + n) + 21 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2371_237114


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l2371_237103

theorem cubic_inequality_solution (x : ℝ) : x^3 - 9*x^2 > -27*x ↔ (0 < x ∧ x < 3) ∨ (x > 6) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l2371_237103


namespace NUMINAMATH_CALUDE_seating_probability_l2371_237105

/-- Represents a seating arrangement of 6 students in a 2x3 grid -/
def SeatingArrangement := Fin 6 → Fin 6

/-- The total number of possible seating arrangements -/
def totalArrangements : ℕ := 720

/-- Checks if three students are seated next to each other and adjacent in the same row or column -/
def isAdjacentArrangement (arr : SeatingArrangement) (a b c : Fin 6) : Prop :=
  sorry

/-- The number of arrangements where Abby, Bridget, and Chris are seated next to each other and adjacent in the same row or column -/
def favorableArrangements : ℕ := 114

/-- The probability of Abby, Bridget, and Chris being seated in a specific arrangement -/
def probability : ℚ := 19 / 120

theorem seating_probability :
  (favorableArrangements : ℚ) / totalArrangements = probability :=
sorry

end NUMINAMATH_CALUDE_seating_probability_l2371_237105


namespace NUMINAMATH_CALUDE_equation_solution_l2371_237107

theorem equation_solution :
  ∃! x : ℚ, x ≠ -5 ∧ (x^2 + 3*x + 4) / (x + 5) = x + 7 :=
by
  use (-31 / 9)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2371_237107


namespace NUMINAMATH_CALUDE_tens_digit_of_3_to_405_l2371_237173

theorem tens_digit_of_3_to_405 : ∃ n : ℕ, 3^405 ≡ 40 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_to_405_l2371_237173


namespace NUMINAMATH_CALUDE_contrapositive_squared_sum_l2371_237159

theorem contrapositive_squared_sum (x y : ℝ) : x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_squared_sum_l2371_237159


namespace NUMINAMATH_CALUDE_boat_speed_proof_l2371_237126

def river_speed : ℝ := 7
def downstream_distance : ℝ := 10
def lake_distance : ℝ := 20
def total_time : ℝ := 1

theorem boat_speed_proof (v : ℝ) (h : v > 0) :
  (downstream_distance / (v + river_speed) + lake_distance / v = total_time) →
  v = 28 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_proof_l2371_237126


namespace NUMINAMATH_CALUDE_oak_willow_difference_l2371_237144

theorem oak_willow_difference (total_trees : ℕ) (willows : ℕ) : 
  total_trees = 83 → willows = 36 → total_trees - willows - willows = 11 := by
  sorry

end NUMINAMATH_CALUDE_oak_willow_difference_l2371_237144


namespace NUMINAMATH_CALUDE_pythagorean_chord_l2371_237129

theorem pythagorean_chord (m : ℕ) (h : m ≥ 3) : 
  let width := 2 * m
  let height := m^2 - 1
  let diagonal := height + 2
  width^2 + height^2 = diagonal^2 ∧ diagonal = m^2 + 1 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_chord_l2371_237129


namespace NUMINAMATH_CALUDE_ellipse_problem_l2371_237148

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define points A, B, and P
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def P : ℝ × ℝ := (0, 1)

-- Define line AB
def lineAB (x y : ℝ) : Prop := sorry

-- Define line y = -x + 2
def intersectLine (x y : ℝ) : Prop := y = -x + 2

-- Define points C and D
def C : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry

-- Define slopes
def slopePA : ℝ := sorry
def slopeAB : ℝ := sorry
def slopePB : ℝ := sorry

-- Theorem statement
theorem ellipse_problem :
  ellipse A.1 A.2 ∧ 
  ellipse B.1 B.2 ∧ 
  A ≠ P ∧ 
  B ≠ P ∧ 
  lineAB 0 0 ∧
  intersectLine C.1 C.2 ∧
  intersectLine D.1 D.2 →
  (∃ k : ℝ, slopePA + slopePB = 2 * slopeAB) ∧
  (∃ minArea : ℝ, minArea = Real.sqrt 2 / 3 ∧ 
    ∀ area : ℝ, area ≥ minArea) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_problem_l2371_237148


namespace NUMINAMATH_CALUDE_curve_and_circle_properties_l2371_237132

-- Define the points and vectors
def E : ℝ × ℝ := (-2, 0)
def F : ℝ × ℝ := (2, 0)
def A : ℝ × ℝ := (2, 1)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the curve C
def C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the condition for point M on curve C
def M_condition (x y : ℝ) : Prop :=
  dot_product (x + 2, y) (x - 2, y) = -3

-- Define the point P and the tangent condition
def P (a b : ℝ) : ℝ × ℝ := (a, b)
def tangent_condition (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), C x y ∧ (a - x)^2 + (b - y)^2 = (a - 2)^2 + (b - 1)^2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 6/5)^2 + (y - 3/5)^2 = (3 * Real.sqrt 5 / 5 - 1)^2

-- State the theorem
theorem curve_and_circle_properties :
  ∀ (x y a b : ℝ),
    C x y ∧
    M_condition x y ∧
    tangent_condition a b →
    (∀ (u v : ℝ), C u v ↔ (u - 1)^2 + v^2 = 1) ∧
    (∀ (r : ℝ), r > 0 → 
      (∀ (u v : ℝ), (u - a)^2 + (v - b)^2 = r^2 → ¬(C u v)) →
      r ≥ 3 * Real.sqrt 5 / 5 - 1) ∧
    circle_equation a b :=
by sorry

end NUMINAMATH_CALUDE_curve_and_circle_properties_l2371_237132


namespace NUMINAMATH_CALUDE_problem_statement_l2371_237191

theorem problem_statement (a b c k : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ k > 0) 
  (h_eq : 2 * a * b * c + k * (a^2 + b^2 + c^2) = k^3) : 
  Real.sqrt ((k - a) * (k - b) / ((k + a) * (k + b))) + 
  Real.sqrt ((k - b) * (k - c) / ((k + b) * (k + c))) + 
  Real.sqrt ((k - c) * (k - a) / ((k + c) * (k + a))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2371_237191


namespace NUMINAMATH_CALUDE_treasure_chest_gems_l2371_237110

theorem treasure_chest_gems (diamonds : ℕ) (rubies : ℕ) 
    (h1 : diamonds = 45) 
    (h2 : rubies = 5110) : 
  diamonds + rubies = 5155 := by
  sorry

end NUMINAMATH_CALUDE_treasure_chest_gems_l2371_237110


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_l2371_237112

def A (m : ℝ) : Set ℝ := {m + 1, -3}
def B (m : ℝ) : Set ℝ := {2*m + 1, m - 3}

theorem intersection_implies_m_value :
  ∀ m : ℝ, (A m ∩ B m = {-3}) → m = -2 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_l2371_237112


namespace NUMINAMATH_CALUDE_project_work_time_l2371_237111

/-- Calculates the time spent working on a project given the total days and nap information -/
def timeSpentWorking (totalDays : ℕ) (numberOfNaps : ℕ) (hoursPerNap : ℕ) : ℕ :=
  totalDays * 24 - numberOfNaps * hoursPerNap

/-- Theorem: Given 4 days and 6 seven-hour naps, the time spent working is 54 hours -/
theorem project_work_time :
  timeSpentWorking 4 6 7 = 54 := by
  sorry

end NUMINAMATH_CALUDE_project_work_time_l2371_237111


namespace NUMINAMATH_CALUDE_average_first_six_l2371_237174

theorem average_first_six (total_count : Nat) (total_avg : ℝ) (last_six_avg : ℝ) (sixth_num : ℝ) :
  total_count = 11 →
  total_avg = 10.7 →
  last_six_avg = 11.4 →
  sixth_num = 13.700000000000017 →
  (6 * ((total_count : ℝ) * total_avg - 6 * last_six_avg + sixth_num)) / 6 = 10.5 := by
  sorry

#check average_first_six

end NUMINAMATH_CALUDE_average_first_six_l2371_237174


namespace NUMINAMATH_CALUDE_problem_statement_l2371_237113

noncomputable section

def f (a b x : ℝ) : ℝ := Real.exp x - a * x^2 - b * x - 1

def g (a b : ℝ) : ℝ → ℝ := λ x ↦ Real.exp x - 2 * a * x - b

theorem problem_statement (a b : ℝ) :
  (∀ x, |x - a| ≥ f a b x) →
  (∀ x, (Real.exp 1 - 1) * x - 1 = (f a b x - f a b 1) / (x - 1) + f a b 1) →
  (a ≤ 1/2) ∧
  (a = 0 ∧ b = 1) ∧
  (∀ x ∈ Set.Icc 0 1,
    g a b x ≥ min (1 - b)
      (min (2*a - 2*a * Real.log (2*a) - b)
        (1 - 2*a - b))) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2371_237113


namespace NUMINAMATH_CALUDE_equal_cost_for_54_students_option2_cheaper_for_50_students_students_more_than_40_l2371_237196

/-- Represents the discount options for movie tickets. -/
inductive DiscountOption
  | Option1
  | Option2

/-- Calculates the total cost for a given number of students and discount option. -/
def calculateCost (students : ℕ) (option : DiscountOption) : ℚ :=
  match option with
  | DiscountOption.Option1 => 30 * students * (1 - 1/5)
  | DiscountOption.Option2 => 30 * (students - 6) * (1 - 1/10)

/-- Theorem stating that both discount options result in the same cost for 54 students. -/
theorem equal_cost_for_54_students :
  calculateCost 54 DiscountOption.Option1 = calculateCost 54 DiscountOption.Option2 :=
by sorry

/-- Theorem stating that Option 2 is cheaper for 50 students. -/
theorem option2_cheaper_for_50_students :
  calculateCost 50 DiscountOption.Option2 < calculateCost 50 DiscountOption.Option1 :=
by sorry

/-- Theorem stating that the number of students is more than 40. -/
theorem students_more_than_40 : 54 > 40 :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_for_54_students_option2_cheaper_for_50_students_students_more_than_40_l2371_237196


namespace NUMINAMATH_CALUDE_domino_arrangements_4_5_l2371_237183

/-- The number of distinct arrangements for placing dominoes on a grid. -/
def dominoArrangements (m n k : ℕ) : ℕ :=
  Nat.choose (m + n - 2) (m - 1)

/-- Theorem: The number of distinct arrangements for placing 4 dominoes on a 4 by 5 grid,
    moving only right or down from upper left to lower right corner, is 35. -/
theorem domino_arrangements_4_5 :
  dominoArrangements 4 5 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_domino_arrangements_4_5_l2371_237183


namespace NUMINAMATH_CALUDE_add_248_64_l2371_237138

theorem add_248_64 : 248 + 64 = 312 := by
  sorry

end NUMINAMATH_CALUDE_add_248_64_l2371_237138


namespace NUMINAMATH_CALUDE_digits_for_369_pages_l2371_237102

/-- The total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  (min n 9) + 
  (max (min n 99 - 9) 0) * 2 + 
  (max (n - 99) 0) * 3

/-- Theorem: The total number of digits used in numbering the pages of a book with 369 pages is 999 -/
theorem digits_for_369_pages : totalDigits 369 = 999 := by
  sorry

end NUMINAMATH_CALUDE_digits_for_369_pages_l2371_237102


namespace NUMINAMATH_CALUDE_total_money_found_l2371_237153

-- Define the amount each person receives
def individual_share : ℝ := 32.50

-- Define the number of people sharing the money
def number_of_people : ℕ := 2

-- Theorem to prove
theorem total_money_found (even_split : ℝ → ℕ → ℝ) :
  even_split individual_share number_of_people = 65.00 :=
by sorry

end NUMINAMATH_CALUDE_total_money_found_l2371_237153


namespace NUMINAMATH_CALUDE_david_twice_rosy_age_l2371_237193

/-- Represents the current age of Rosy -/
def rosy_age : ℕ := 8

/-- Represents the current age difference between David and Rosy -/
def age_difference : ℕ := 12

/-- Calculates the number of years until David is twice Rosy's age -/
def years_until_double : ℕ :=
  let david_age := rosy_age + age_difference
  (david_age - 2 * rosy_age)

theorem david_twice_rosy_age : years_until_double = 4 := by
  sorry

end NUMINAMATH_CALUDE_david_twice_rosy_age_l2371_237193


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2371_237146

theorem expression_simplification_and_evaluation :
  let f (x : ℚ) := (x^2 - 4*x) / (x^2 - 16) / ((x^2 + 4*x) / (x^2 + 8*x + 16)) - 2*x / (x - 4)
  f (-2 : ℚ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2371_237146


namespace NUMINAMATH_CALUDE_flower_arrangement_daisies_percentage_l2371_237141

theorem flower_arrangement_daisies_percentage
  (total_flowers : ℕ)
  (h1 : total_flowers > 0)
  (yellow_flowers : ℕ)
  (h2 : yellow_flowers = (7 * total_flowers) / 10)
  (white_flowers : ℕ)
  (h3 : white_flowers = total_flowers - yellow_flowers)
  (yellow_tulips : ℕ)
  (h4 : yellow_tulips = yellow_flowers / 2)
  (white_daisies : ℕ)
  (h5 : white_daisies = (2 * white_flowers) / 3)
  (yellow_daisies : ℕ)
  (h6 : yellow_daisies = yellow_flowers - yellow_tulips)
  (total_daisies : ℕ)
  (h7 : total_daisies = yellow_daisies + white_daisies) :
  (total_daisies : ℚ) / total_flowers = 11 / 20 :=
sorry

end NUMINAMATH_CALUDE_flower_arrangement_daisies_percentage_l2371_237141


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2371_237189

-- Define the types for planes and lines
variable (Plane : Type) (Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel
  (α β γ : Plane) (m n : Line)
  (h₁ : α ≠ β) (h₂ : α ≠ γ) (h₃ : β ≠ γ) (h₄ : m ≠ n)
  (h₅ : perpendicular m α) (h₆ : perpendicular n α) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2371_237189


namespace NUMINAMATH_CALUDE_delta_curve_from_rotations_l2371_237137

/-- A curve in 2D space -/
structure Curve where
  -- Add necessary fields for a curve

/-- Rotation of a curve around a point by an angle -/
def rotate (c : Curve) (center : ℝ × ℝ) (angle : ℝ) : Curve :=
  sorry

/-- Sum of curves -/
def sum_curves (curves : List Curve) : Curve :=
  sorry

/-- Check if a curve is a circle with given radius -/
def is_circle (c : Curve) (radius : ℝ) : Prop :=
  sorry

/-- Check if a curve is convex -/
def is_convex (c : Curve) : Prop :=
  sorry

/-- Check if a curve is a Δ-curve -/
def is_delta_curve (c : Curve) : Prop :=
  sorry

/-- Main theorem -/
theorem delta_curve_from_rotations (K : Curve) (O : ℝ × ℝ) (h : ℝ) :
  is_convex K →
  let K' := rotate K O (2 * π / 3)
  let K'' := rotate K O (4 * π / 3)
  let M := sum_curves [K, K', K'']
  is_circle M h →
  is_delta_curve K :=
sorry

end NUMINAMATH_CALUDE_delta_curve_from_rotations_l2371_237137


namespace NUMINAMATH_CALUDE_one_bee_has_six_legs_l2371_237108

/-- The number of legs a bee has -/
def bee_legs : ℕ := sorry

/-- Two bees have 12 legs -/
axiom two_bees_legs : 2 * bee_legs = 12

/-- Prove that one bee has 6 legs -/
theorem one_bee_has_six_legs : bee_legs = 6 := by sorry

end NUMINAMATH_CALUDE_one_bee_has_six_legs_l2371_237108


namespace NUMINAMATH_CALUDE_sum_of_non_solutions_l2371_237125

/-- Given an equation with infinitely many solutions, prove the sum of non-solutions -/
theorem sum_of_non_solutions (A B C : ℝ) : 
  (∀ x : ℝ, (x + B) * (A * x + 28) = 2 * (x + C) * (x + 7)) →
  (∃ S : Finset ℝ, (∀ x ∉ S, (x + B) * (A * x + 28) = 2 * (x + C) * (x + 7)) ∧ 
    (∀ x ∈ S, (x + B) * (A * x + 28) ≠ 2 * (x + C) * (x + 7)) ∧
    (Finset.sum S id = -21)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_non_solutions_l2371_237125


namespace NUMINAMATH_CALUDE_problem_statement_l2371_237198

theorem problem_statement (n : ℝ) (h : n + 1/n = 5) : n^2 + 1/n^2 + 7 = 30 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2371_237198


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_zero_or_one_l2371_237117

/-- Given a real number a, define the set A as the solutions to ax^2 + 2x + 1 = 0 -/
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

/-- Theorem: If A(a) has exactly one element, then a = 0 or a = 1 -/
theorem unique_solution_implies_a_zero_or_one (a : ℝ) : (∃! x, x ∈ A a) → a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_zero_or_one_l2371_237117


namespace NUMINAMATH_CALUDE_min_socks_for_pairs_l2371_237156

/-- Represents a sock with a color -/
inductive Sock
| Blue
| Red

/-- Represents a drawer containing socks -/
structure Drawer where
  socks : List Sock
  blue_count : Nat
  red_count : Nat
  balanced : blue_count = red_count

/-- Checks if a list of socks contains a pair of the same color -/
def hasSameColorPair (socks : List Sock) : Bool :=
  sorry

/-- Checks if a list of socks contains a pair of different colors -/
def hasDifferentColorPair (socks : List Sock) : Bool :=
  sorry

/-- Theorem stating the minimum number of socks required -/
theorem min_socks_for_pairs (d : Drawer) :
  (∀ n : Nat, n < 4 → ¬(∀ subset : List Sock, subset.length = n →
    (hasSameColorPair subset ∧ hasDifferentColorPair subset))) ∧
  (∃ subset : List Sock, subset.length = 4 ∧
    (hasSameColorPair subset ∧ hasDifferentColorPair subset)) :=
  sorry

end NUMINAMATH_CALUDE_min_socks_for_pairs_l2371_237156


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2371_237134

theorem inequality_solution_set (x : ℝ) :
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ∧ x^2 - 6*x + 8 ≥ 0) ↔ 
  (x ∈ Set.Icc (-5) 1 ∪ Set.Icc 5 11) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2371_237134


namespace NUMINAMATH_CALUDE_count_valid_integers_eq_44_l2371_237162

def digit_set : List Nat := [2, 3, 5, 5, 6, 6, 6]

def is_valid_integer (n : Nat) : Bool :=
  let digits := n.digits 10
  digits.length == 3 ∧ 
  digits.all (λ d => d ∈ digit_set) ∧
  digits.count 2 ≤ 1 ∧
  digits.count 3 ≤ 1 ∧
  digits.count 5 ≤ 2 ∧
  digits.count 6 ≤ 3

def count_valid_integers : Nat :=
  (List.range 900).map (λ n => n + 100)
    |>.filter is_valid_integer
    |>.length

theorem count_valid_integers_eq_44 : count_valid_integers = 44 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_integers_eq_44_l2371_237162


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2371_237123

theorem polynomial_expansion (x : ℝ) :
  (7 * x^2 + 3) * (5 * x^3 + 4 * x + 1) = 35 * x^5 + 43 * x^3 + 7 * x^2 + 12 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2371_237123


namespace NUMINAMATH_CALUDE_hoseoks_number_l2371_237120

theorem hoseoks_number (x : ℤ) : x - 10 = 15 → x + 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_hoseoks_number_l2371_237120


namespace NUMINAMATH_CALUDE_archer_problem_l2371_237177

theorem archer_problem (n m : ℕ) : 
  (10 < n) → 
  (n < 20) → 
  (5 * m = 3 * (n - m)) → 
  (n = 16 ∧ m = 6) := by
sorry

end NUMINAMATH_CALUDE_archer_problem_l2371_237177


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a3_l2371_237169

/-- An arithmetic sequence with common difference 2 where a₂ is the geometric mean of a₁ and a₅ -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧ a 2 ^ 2 = a 1 * a 5

theorem arithmetic_sequence_a3 (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a3_l2371_237169


namespace NUMINAMATH_CALUDE_decagon_game_outcome_dodecagon_game_outcome_l2371_237178

/-- Represents the possible outcomes of the game -/
inductive GameOutcome
| FirstPlayerWins
| SecondPlayerWins

/-- Represents a regular polygon with alternating colored vertices -/
structure ColoredPolygon where
  sides : ℕ
  vertices_alternating_colors : sides > 0

/-- The game played on a colored polygon -/
def polygon_segment_game (p : ColoredPolygon) : GameOutcome :=
  sorry

/-- Theorem stating the outcome for a decagon -/
theorem decagon_game_outcome :
  polygon_segment_game ⟨10, by norm_num⟩ = GameOutcome.SecondPlayerWins :=
sorry

/-- Theorem stating the outcome for a dodecagon -/
theorem dodecagon_game_outcome :
  polygon_segment_game ⟨12, by norm_num⟩ = GameOutcome.FirstPlayerWins :=
sorry

end NUMINAMATH_CALUDE_decagon_game_outcome_dodecagon_game_outcome_l2371_237178


namespace NUMINAMATH_CALUDE_teeth_removal_theorem_l2371_237182

theorem teeth_removal_theorem :
  let total_teeth : ℕ := 32
  let first_person_removed : ℕ := total_teeth / 4
  let second_person_removed : ℕ := total_teeth * 3 / 8
  let third_person_removed : ℕ := total_teeth / 2
  let fourth_person_removed : ℕ := 4
  first_person_removed + second_person_removed + third_person_removed + fourth_person_removed = 40 := by
  sorry

end NUMINAMATH_CALUDE_teeth_removal_theorem_l2371_237182


namespace NUMINAMATH_CALUDE_ellipse_sum_l2371_237160

theorem ellipse_sum (h k a b : ℝ) : 
  (h = 3) → 
  (k = -5) → 
  (a = 7) → 
  (b = 4) → 
  h + k + a + b = 9 := by
sorry

end NUMINAMATH_CALUDE_ellipse_sum_l2371_237160


namespace NUMINAMATH_CALUDE_vector_angle_in_circle_l2371_237167

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define the theorem
theorem vector_angle_in_circle (O A B C : ℝ × ℝ) (r : ℝ) :
  A ∈ Circle O r →
  B ∈ Circle O r →
  C ∈ Circle O r →
  (A.1 - O.1, A.2 - O.2) = (1/2) * ((B.1 - A.1, B.2 - A.2) + (C.1 - A.1, C.2 - A.2)) →
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_vector_angle_in_circle_l2371_237167


namespace NUMINAMATH_CALUDE_girls_fraction_proof_l2371_237188

theorem girls_fraction_proof (T G B : ℕ) (x : ℚ) : 
  (x * G = (1 / 6) * T) →  -- Some fraction of girls is 1/6 of total
  (B = 2 * G) →            -- Ratio of boys to girls is 2
  (T = B + G) →            -- Total is sum of boys and girls
  (x = 1 / 2) :=           -- Fraction of girls is 1/2
by sorry

end NUMINAMATH_CALUDE_girls_fraction_proof_l2371_237188


namespace NUMINAMATH_CALUDE_desired_circle_properties_l2371_237143

/-- The first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

/-- The second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

/-- The line on which the center of the desired circle lies -/
def centerLine (x y : ℝ) : Prop := x + y = 0

/-- The equation of the desired circle -/
def desiredCircle (x y : ℝ) : Prop := (x + 3)^2 + (y - 3)^2 = 10

/-- Theorem stating that the desired circle passes through the intersection points of circle1 and circle2,
    and its center lies on the centerLine -/
theorem desired_circle_properties :
  ∀ x y : ℝ, 
    (circle1 x y ∧ circle2 x y) → 
    desiredCircle x y ∧ 
    ∃ cx cy : ℝ, centerLine cx cy ∧ desiredCircle (x - cx) (y - cy) := by
  sorry


end NUMINAMATH_CALUDE_desired_circle_properties_l2371_237143


namespace NUMINAMATH_CALUDE_incircle_tangent_inequality_l2371_237118

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the incircle points
variable (A₁ B₁ : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h_triangle : Triangle A B C)
variable (h_incircle : IsIncircle A₁ B₁ A B C)
variable (h_AC_gt_BC : dist A C > dist B C)

-- State the theorem
theorem incircle_tangent_inequality :
  dist A A₁ > dist B B₁ := by sorry

end NUMINAMATH_CALUDE_incircle_tangent_inequality_l2371_237118


namespace NUMINAMATH_CALUDE_gumball_problem_solution_l2371_237149

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- 
Given a gumball machine with the specified number of gumballs for each color,
this function returns the minimum number of gumballs one must buy to guarantee
getting four of the same color.
-/
def minGumballsToBuy (machine : GumballMachine) : Nat :=
  sorry

/-- The theorem stating the correct answer for the given problem -/
theorem gumball_problem_solution :
  let machine : GumballMachine := { red := 10, white := 6, blue := 8, green := 9 }
  minGumballsToBuy machine = 13 := by
  sorry

end NUMINAMATH_CALUDE_gumball_problem_solution_l2371_237149


namespace NUMINAMATH_CALUDE_complex_product_theorem_l2371_237124

theorem complex_product_theorem (z₁ z₂ : ℂ) (a b : ℝ) : 
  z₁ = (1 - Complex.I) * (3 + Complex.I) →
  a = z₁.im →
  z₂ = (1 + Complex.I) / (2 - Complex.I) →
  b = z₂.re →
  a * b = -2/5 := by
    sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l2371_237124


namespace NUMINAMATH_CALUDE_function_composition_inverse_l2371_237145

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 3
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem function_composition_inverse (a b : ℝ) :
  (∀ x, h a b x = (x - 6) / 2) →
  a - b = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_function_composition_inverse_l2371_237145


namespace NUMINAMATH_CALUDE_milk_carton_volume_l2371_237140

theorem milk_carton_volume (surface_area : ℝ) (h : surface_area = 600) :
  let side_length := Real.sqrt (surface_area / 6)
  side_length ^ 3 = 1000 := by
sorry

end NUMINAMATH_CALUDE_milk_carton_volume_l2371_237140


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l2371_237101

/-- An arithmetic sequence with common difference d ≠ 0 and first term a₁ = 2d -/
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ :=
  2 * d + (n - 1) * d

theorem arithmetic_sequence_geometric_mean (d : ℝ) (k : ℕ) (h : d ≠ 0) :
  (arithmetic_sequence d k) ^ 2 = (arithmetic_sequence d 1) * (arithmetic_sequence d (2 * k + 7)) →
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l2371_237101


namespace NUMINAMATH_CALUDE_sandy_loses_two_marks_l2371_237197

/-- Represents Sandy's math test results -/
structure SandyTest where
  correct_mark : ℕ  -- marks for each correct sum
  total_sums : ℕ    -- total number of sums attempted
  total_marks : ℕ   -- total marks obtained
  correct_sums : ℕ  -- number of correct sums

/-- Calculates the marks lost for each incorrect sum -/
def marks_lost_per_incorrect (test : SandyTest) : ℚ :=
  let correct_marks := test.correct_mark * test.correct_sums
  let incorrect_sums := test.total_sums - test.correct_sums
  let total_marks_lost := correct_marks - test.total_marks
  (total_marks_lost : ℚ) / incorrect_sums

/-- Theorem stating that Sandy loses 2 marks for each incorrect sum -/
theorem sandy_loses_two_marks (test : SandyTest) 
  (h1 : test.correct_mark = 3)
  (h2 : test.total_sums = 30)
  (h3 : test.total_marks = 50)
  (h4 : test.correct_sums = 22) :
  marks_lost_per_incorrect test = 2 := by
  sorry

#eval marks_lost_per_incorrect { correct_mark := 3, total_sums := 30, total_marks := 50, correct_sums := 22 }

end NUMINAMATH_CALUDE_sandy_loses_two_marks_l2371_237197


namespace NUMINAMATH_CALUDE_rectangle_and_triangle_l2371_237171

/-- Given a rectangle ABCD and an isosceles right triangle DCE, prove that DE = 4√3 -/
theorem rectangle_and_triangle (AB AD DC DE : ℝ) : 
  AB = 6 →
  AD = 8 →
  DC = DE →
  AB * AD = 2 * (1/2 * DC * DE) →
  DE = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_and_triangle_l2371_237171


namespace NUMINAMATH_CALUDE_alpha_computation_l2371_237152

theorem alpha_computation (α β : ℂ) :
  (α + β).re > 0 →
  (Complex.I * (α - 3 * β)).re > 0 →
  β = 4 + 3 * Complex.I →
  α = 3 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_alpha_computation_l2371_237152


namespace NUMINAMATH_CALUDE_range_of_a_l2371_237194

-- Define the proposition
def proposition (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 + 2*x - a ≥ 0

-- State the theorem
theorem range_of_a (h : ∀ a : ℝ, proposition a ↔ a ∈ Set.Iic 15) :
  {a : ℝ | proposition a} = Set.Iic 15 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2371_237194


namespace NUMINAMATH_CALUDE_shaded_area_problem_l2371_237104

/-- The area of the shaded region in a figure where a 4-inch by 4-inch square 
    adjoins a 12-inch by 12-inch square. -/
theorem shaded_area_problem : 
  let small_square_side : ℝ := 4
  let large_square_side : ℝ := 12
  let small_square_area := small_square_side ^ 2
  let triangle_base := small_square_side
  let triangle_height := small_square_side * large_square_side / (large_square_side + small_square_side)
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let shaded_area := small_square_area - triangle_area
  shaded_area = 10
  := by sorry

end NUMINAMATH_CALUDE_shaded_area_problem_l2371_237104


namespace NUMINAMATH_CALUDE_distance_for_boy_problem_l2371_237184

/-- Calculates the distance covered given time in minutes and speed in meters per second -/
def distance_covered (time_minutes : ℕ) (speed_meters_per_second : ℕ) : ℕ :=
  time_minutes * 60 * speed_meters_per_second

/-- Proves that given 30 minutes and a speed of 1 meter per second, the distance covered is 1800 meters -/
theorem distance_for_boy_problem : distance_covered 30 1 = 1800 := by
  sorry

#eval distance_covered 30 1

end NUMINAMATH_CALUDE_distance_for_boy_problem_l2371_237184


namespace NUMINAMATH_CALUDE_eventually_single_digit_or_zero_l2371_237165

/-- Function to calculate the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let digits := Nat.digits 10 n
  digits.foldl (·*·) 1

/-- Predicate to check if a number is single-digit or zero -/
def isSingleDigitOrZero (n : ℕ) : Prop :=
  n < 10

/-- Theorem stating that repeatedly applying digitProduct will eventually
    result in a single-digit number or zero -/
theorem eventually_single_digit_or_zero (n : ℕ) :
  ∃ k : ℕ, isSingleDigitOrZero ((digitProduct^[k]) n) :=
sorry


end NUMINAMATH_CALUDE_eventually_single_digit_or_zero_l2371_237165


namespace NUMINAMATH_CALUDE_star_difference_l2371_237192

def star (x y : ℤ) : ℤ := x * y - 2 * x + y ^ 2

theorem star_difference : (star 7 4) - (star 4 7) = -39 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_l2371_237192


namespace NUMINAMATH_CALUDE_tenth_term_is_19_l2371_237119

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  /-- The sum of the first n terms -/
  S : ℕ → ℝ
  /-- The nth term of the sequence -/
  a : ℕ → ℝ
  /-- The sum of the first 9 terms is 81 -/
  sum_9 : S 9 = 81
  /-- The second term is 3 -/
  second_term : a 2 = 3
  /-- The sequence follows the arithmetic sequence property -/
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- The 10th term of the specified arithmetic sequence is 19 -/
theorem tenth_term_is_19 (seq : ArithmeticSequence) : seq.a 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_19_l2371_237119


namespace NUMINAMATH_CALUDE_steven_peach_count_l2371_237133

-- Define the number of peaches Jake and Steven have
def jake_peaches : ℕ := 7
def steven_peaches : ℕ := jake_peaches + 12

-- Theorem to prove
theorem steven_peach_count : steven_peaches = 19 := by
  sorry

end NUMINAMATH_CALUDE_steven_peach_count_l2371_237133


namespace NUMINAMATH_CALUDE_max_ac_value_max_ac_value_is_510050_l2371_237180

theorem max_ac_value (a b c d : ℤ) 
  (h1 : a > b ∧ b > c ∧ c > d ∧ d ≥ -2021)
  (h2 : (a + b) * (d + a) = (b + c) * (c + d))
  (h3 : b + c ≠ 0)
  (h4 : d + a ≠ 0) :
  ∀ (a' b' c' d' : ℤ), 
    (a' > b' ∧ b' > c' ∧ c' > d' ∧ d' ≥ -2021) →
    ((a' + b') * (d' + a') = (b' + c') * (c' + d')) →
    (b' + c' ≠ 0) →
    (d' + a' ≠ 0) →
    a * c ≥ a' * c' :=
by sorry

theorem max_ac_value_is_510050 (a b c d : ℤ) 
  (h1 : a > b ∧ b > c ∧ c > d ∧ d ≥ -2021)
  (h2 : (a + b) * (d + a) = (b + c) * (c + d))
  (h3 : b + c ≠ 0)
  (h4 : d + a ≠ 0) :
  a * c ≤ 510050 :=
by sorry

end NUMINAMATH_CALUDE_max_ac_value_max_ac_value_is_510050_l2371_237180


namespace NUMINAMATH_CALUDE_system_solution_l2371_237150

theorem system_solution (x y z : ℝ) : 
  (2 * x^2 + 3 * y + 5 = 2 * Real.sqrt (2 * z + 5)) ∧
  (2 * y^2 + 3 * z + 5 = 2 * Real.sqrt (2 * x + 5)) ∧
  (2 * z^2 + 3 * x + 5 = 2 * Real.sqrt (2 * y + 5)) →
  x = -1/2 ∧ y = -1/2 ∧ z = -1/2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2371_237150


namespace NUMINAMATH_CALUDE_equation_one_solutions_l2371_237155

theorem equation_one_solutions (x : ℝ) :
  (x - 2)^2 = 4 → x = 4 ∨ x = 0 := by
  sorry

#check equation_one_solutions

end NUMINAMATH_CALUDE_equation_one_solutions_l2371_237155


namespace NUMINAMATH_CALUDE_part_one_part_two_l2371_237172

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 3*x - 10 > 0

-- Part I
theorem part_one (x : ℝ) :
  (∃ a : ℝ, a = 1 ∧ a > 0 ∧ p x a ∧ q x) → 2 < x ∧ x < 3 :=
sorry

-- Part II
theorem part_two :
  (∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x)) →
  (∃ a : ℝ, 1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2371_237172


namespace NUMINAMATH_CALUDE_equation_solution_l2371_237168

theorem equation_solution : ∃ x : ℝ, 45 * x = 0.4 * 900 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2371_237168


namespace NUMINAMATH_CALUDE_janice_purchase_l2371_237121

theorem janice_purchase (a b c : ℕ) : 
  a + b + c = 50 →
  30 * a + 200 * b + 300 * c = 5000 →
  a = 10 :=
by sorry

end NUMINAMATH_CALUDE_janice_purchase_l2371_237121


namespace NUMINAMATH_CALUDE_pears_equivalent_to_24_bananas_is_12_l2371_237122

/-- The number of pears equivalent in cost to 24 bananas -/
def pears_equivalent_to_24_bananas (banana_apple_ratio : ℚ) (apple_pear_ratio : ℚ) : ℚ :=
  24 * banana_apple_ratio * apple_pear_ratio

theorem pears_equivalent_to_24_bananas_is_12 :
  pears_equivalent_to_24_bananas (3/4) (6/9) = 12 := by
  sorry

end NUMINAMATH_CALUDE_pears_equivalent_to_24_bananas_is_12_l2371_237122


namespace NUMINAMATH_CALUDE_triangle_division_exists_l2371_237181

/-- Represents a part of the triangle -/
structure TrianglePart where
  numbers : List Nat
  sum : Nat

/-- Represents the entire triangle -/
structure Triangle where
  total_sum : Nat
  parts : List TrianglePart

/-- Checks if a triangle is valid according to the problem conditions -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.total_sum = 63 ∧
  t.parts.length = 3 ∧
  (∀ p ∈ t.parts, p.sum = p.numbers.sum) ∧
  (∀ p ∈ t.parts, p.sum = t.total_sum / 3) ∧
  (t.parts.map (·.numbers)).join.sum = t.total_sum

theorem triangle_division_exists :
  ∃ t : Triangle, is_valid_triangle t :=
sorry

end NUMINAMATH_CALUDE_triangle_division_exists_l2371_237181


namespace NUMINAMATH_CALUDE_acute_angle_range_characterization_l2371_237161

/-- The angle between two vectors is acute if and only if their dot product is positive and they are not collinear -/
def is_acute_angle (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2 > 0) ∧ (a.1 * b.2 ≠ a.2 * b.1)

/-- The set of real numbers m for which the angle between vectors a and b is acute -/
def acute_angle_range : Set ℝ :=
  {m | is_acute_angle (m - 2, m + 3) (2*m + 1, m - 2)}

theorem acute_angle_range_characterization :
  acute_angle_range = {m | m > 2 ∨ (m < (-11 - 5*Real.sqrt 5) / 2) ∨ 
    (((-11 + 5*Real.sqrt 5) / 2 < m) ∧ (m < -4/3))} := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_range_characterization_l2371_237161


namespace NUMINAMATH_CALUDE_complex_power_series_sum_l2371_237116

def complex_power_sequence (n : ℕ) : ℂ := (2 + Complex.I) ^ n

def real_part_sequence (n : ℕ) : ℝ := (complex_power_sequence n).re
def imag_part_sequence (n : ℕ) : ℝ := (complex_power_sequence n).im

theorem complex_power_series_sum :
  (∑' n, (real_part_sequence n * imag_part_sequence n) / 7 ^ n) = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_series_sum_l2371_237116


namespace NUMINAMATH_CALUDE_shoe_color_probability_l2371_237163

theorem shoe_color_probability (n : ℕ) (h : n = 6) :
  let total_shoes := 2 * n
  let same_color_selections := n
  let total_selections := total_shoes.choose 2
  (same_color_selections : ℚ) / total_selections = 1 / 11 :=
by sorry

end NUMINAMATH_CALUDE_shoe_color_probability_l2371_237163


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_minus_i_l2371_237166

/-- The imaginary part of the complex number i / (1 - i) is 1/2 -/
theorem imaginary_part_of_i_over_one_minus_i : Complex.im (Complex.I / (1 - Complex.I)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_minus_i_l2371_237166


namespace NUMINAMATH_CALUDE_natural_number_pairs_l2371_237130

theorem natural_number_pairs : 
  ∀ a b : ℕ, 
    90 < a + b ∧ a + b < 100 ∧ 
    (9/10 : ℚ) < (a : ℚ) / (b : ℚ) ∧ (a : ℚ) / (b : ℚ) < (91/100 : ℚ) → 
    ((a = 46 ∧ b = 51) ∨ (a = 47 ∧ b = 52)) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pairs_l2371_237130


namespace NUMINAMATH_CALUDE_bryden_quarters_value_l2371_237157

/-- The face value of a regular quarter in dollars -/
def regular_quarter_value : ℚ := 1/4

/-- The number of regular quarters Bryden has -/
def regular_quarters : ℕ := 4

/-- The number of special quarters Bryden has -/
def special_quarters : ℕ := 1

/-- The value multiplier for a special quarter compared to a regular quarter -/
def special_quarter_multiplier : ℚ := 2

/-- The percentage of face value the collector offers -/
def collector_offer_percentage : ℚ := 1500

theorem bryden_quarters_value :
  let total_face_value := regular_quarter_value * regular_quarters +
                          regular_quarter_value * special_quarter_multiplier * special_quarters
  let collector_offer_multiplier := collector_offer_percentage / 100
  collector_offer_multiplier * total_face_value = 45/2 :=
sorry

end NUMINAMATH_CALUDE_bryden_quarters_value_l2371_237157


namespace NUMINAMATH_CALUDE_sum_complex_exp_argument_l2371_237176

/-- The sum of five complex exponentials has an argument of 59π/120 -/
theorem sum_complex_exp_argument :
  let z₁ := Complex.exp (11 * Real.pi * Complex.I / 120)
  let z₂ := Complex.exp (31 * Real.pi * Complex.I / 120)
  let z₃ := Complex.exp (-13 * Real.pi * Complex.I / 120)
  let z₄ := Complex.exp (-53 * Real.pi * Complex.I / 120)
  let z₅ := Complex.exp (-73 * Real.pi * Complex.I / 120)
  let sum := z₁ + z₂ + z₃ + z₄ + z₅
  ∃ (r : ℝ), sum = r * Complex.exp (59 * Real.pi * Complex.I / 120) ∧ r > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_complex_exp_argument_l2371_237176


namespace NUMINAMATH_CALUDE_fish_filets_count_l2371_237100

/-- The number of fish filets Ben and his family will have -/
def fish_filets : ℕ :=
  let ben_fish := 4
  let judy_fish := 1
  let billy_fish := 3
  let jim_fish := 2
  let susie_fish := 5
  let total_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let thrown_back := 3
  let kept_fish := total_caught - thrown_back
  let filets_per_fish := 2
  kept_fish * filets_per_fish

theorem fish_filets_count : fish_filets = 24 := by
  sorry

end NUMINAMATH_CALUDE_fish_filets_count_l2371_237100


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l2371_237175

theorem smallest_lcm_with_gcd_5 (m n : ℕ) : 
  1000 ≤ m ∧ m < 10000 ∧ 
  1000 ≤ n ∧ n < 10000 ∧ 
  Nat.gcd m n = 5 →
  201000 ≤ Nat.lcm m n ∧ 
  ∃ (a b : ℕ), 1000 ≤ a ∧ a < 10000 ∧ 
               1000 ≤ b ∧ b < 10000 ∧ 
               Nat.gcd a b = 5 ∧ 
               Nat.lcm a b = 201000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l2371_237175


namespace NUMINAMATH_CALUDE_simplify_expression_l2371_237106

theorem simplify_expression (x : ℝ) : 
  (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*(x - 1) + 1 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2371_237106


namespace NUMINAMATH_CALUDE_triangle_sum_theorem_l2371_237135

def is_valid_triangle (t : Finset Nat) : Prop :=
  t.card = 3 ∧ ∀ x ∈ t, 1 ≤ x ∧ x ≤ 9

def sum_of_triangle (t : Finset Nat) : Nat :=
  t.sum id

def valid_sum (s : Nat) : Prop :=
  12 ≤ s ∧ s ≤ 27 ∧ s ≠ 14 ∧ s ≠ 25

theorem triangle_sum_theorem :
  {s : Nat | ∃ t1 t2 : Finset Nat,
    is_valid_triangle t1 ∧
    is_valid_triangle t2 ∧
    t1 ∩ t2 = ∅ ∧
    sum_of_triangle t1 = s ∧
    sum_of_triangle t2 = s ∧
    valid_sum s} =
  {12, 13, 15, 16, 17, 18, 19} :=
sorry

end NUMINAMATH_CALUDE_triangle_sum_theorem_l2371_237135
