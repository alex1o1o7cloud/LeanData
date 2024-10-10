import Mathlib

namespace student_number_problem_l1222_122242

theorem student_number_problem (x : ℤ) : x = 60 ↔ 4 * x - 138 = 102 := by
  sorry

end student_number_problem_l1222_122242


namespace building_shadow_length_l1222_122267

/-- Given a flagpole and a building under similar conditions, 
    calculate the length of the shadow cast by the building. -/
theorem building_shadow_length 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : flagpole_shadow = 45)
  (h3 : building_height = 22) :
  (building_height * flagpole_shadow) / flagpole_height = 55 := by
sorry

end building_shadow_length_l1222_122267


namespace sin_cos_range_l1222_122207

theorem sin_cos_range (x : ℝ) : 29/27 ≤ Real.sin x ^ 6 + Real.cos x ^ 4 ∧ Real.sin x ^ 6 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end sin_cos_range_l1222_122207


namespace reciprocal_of_negative_2023_l1222_122230

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_negative_2023 :
  reciprocal (-2023) = -1/2023 := by
  sorry

end reciprocal_of_negative_2023_l1222_122230


namespace algebraic_expression_solution_l1222_122295

theorem algebraic_expression_solution (m : ℚ) : 
  (5 * (2 - 1) + 3 * m * 2 = -7) → 
  (∃ x : ℚ, 5 * (x - 1) + 3 * m * x = -1 ∧ x = -4) :=
by sorry

end algebraic_expression_solution_l1222_122295


namespace broken_calculator_multiplication_l1222_122233

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem broken_calculator_multiplication :
  ∀ a b : ℕ, is_two_digit a → is_two_digit b →
  (a * b = 1001 ∨ a * b = 1100) ↔ 
  ((a = 11 ∧ b = 91) ∨ (a = 91 ∧ b = 11) ∨
   (a = 13 ∧ b = 77) ∨ (a = 77 ∧ b = 13) ∨
   (a = 25 ∧ b = 44) ∨ (a = 44 ∧ b = 25)) :=
by sorry

end broken_calculator_multiplication_l1222_122233


namespace geometric_sequence_fifth_term_l1222_122200

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem states that in a geometric sequence of positive numbers where the third term is 16 and the seventh term is 2, the fifth term is 2. -/
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : IsGeometricSequence a)
  (h_third_term : a 3 = 16)
  (h_seventh_term : a 7 = 2) :
  a 5 = 2 := by
  sorry


end geometric_sequence_fifth_term_l1222_122200


namespace equation_solutions_count_l1222_122231

theorem equation_solutions_count : 
  ∃! (pairs : List (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ pairs ↔ (1 : ℚ) / y - (1 : ℚ) / (y + 2) = (1 : ℚ) / (3 * 2^x)) ∧
    pairs.length = 6 :=
by sorry

end equation_solutions_count_l1222_122231


namespace two_numbers_difference_l1222_122213

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 40)
  (triple_minus_four : 3 * y - 4 * x = 14) :
  |y - x| = 9.714 := by
sorry

end two_numbers_difference_l1222_122213


namespace prime_square_sum_l1222_122254

theorem prime_square_sum (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ ∃ (n : ℕ), p^q + p^r = n^2 ↔ 
  ((p = 2 ∧ q = 2 ∧ r = 5) ∨ 
   (p = 2 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 3) ∨ 
   (p = 3 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 2 ∧ q = r ∧ q ≥ 3 ∧ Prime q)) :=
by sorry

end prime_square_sum_l1222_122254


namespace line_equations_l1222_122291

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.a + l₁.b * l₂.b = 0

/-- Check if a line has equal intercepts on both axes -/
def Line.equalIntercepts (l : Line) : Prop :=
  l.a = l.b ∧ l.a ≠ 0

theorem line_equations (l₁ : Line) :
  (l₁.contains 2 3) →
  (∃ l₂ : Line, l₂.a = 1 ∧ l₂.b = 2 ∧ l₂.c = 4 ∧ l₁.perpendicular l₂) →
  (l₁.a = 2 ∧ l₁.b = -1 ∧ l₁.c = -1) ∨
  (l₁.equalIntercepts → (l₁.a = 1 ∧ l₁.b = 1 ∧ l₁.c = -5) ∨ (l₁.a = 3 ∧ l₁.b = -2 ∧ l₁.c = 0)) :=
by sorry

end line_equations_l1222_122291


namespace midpoint_square_sum_l1222_122290

def A : ℝ × ℝ := (2, 6)
def C : ℝ × ℝ := (4, 1)

theorem midpoint_square_sum (x y : ℝ) : 
  (∀ (p : ℝ × ℝ), p = ((A.1 + x) / 2, (A.2 + y) / 2) → p = C) →
  x^2 + y^2 = 52 := by
  sorry

end midpoint_square_sum_l1222_122290


namespace taxi_fare_calculation_l1222_122219

/-- Represents the fare structure of a taxi service -/
structure TaxiFare where
  baseFare : ℝ
  mileageRate : ℝ
  minuteRate : ℝ

/-- Calculates the total fare for a taxi trip -/
def calculateFare (fare : TaxiFare) (miles : ℝ) (minutes : ℝ) : ℝ :=
  fare.baseFare + fare.mileageRate * miles + fare.minuteRate * minutes

/-- Theorem: Given the fare structure and initial trip data, 
    a 60-mile trip lasting 90 minutes will cost $200 -/
theorem taxi_fare_calculation 
  (fare : TaxiFare)
  (h1 : fare.baseFare = 20)
  (h2 : fare.minuteRate = 0.5)
  (h3 : calculateFare fare 40 60 = 140) :
  calculateFare fare 60 90 = 200 := by
  sorry


end taxi_fare_calculation_l1222_122219


namespace perpendicular_slope_l1222_122216

/-- The slope of a line perpendicular to the line containing points (2, -3) and (-4, -8) is -6/5 -/
theorem perpendicular_slope : 
  let p₁ : ℚ × ℚ := (2, -3)
  let p₂ : ℚ × ℚ := (-4, -8)
  let m : ℚ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  (-1 / m) = -6/5 := by sorry

end perpendicular_slope_l1222_122216


namespace intersection_point_of_lines_l1222_122270

theorem intersection_point_of_lines (x y : ℝ) :
  y = x ∧ y = -x + 2 → (x = 1 ∧ y = 1) :=
by sorry

end intersection_point_of_lines_l1222_122270


namespace max_sum_of_squares_l1222_122220

theorem max_sum_of_squares (x y : ℤ) : 3 * x^2 + 5 * y^2 = 345 → (x + y ≤ 13) ∧ ∃ (a b : ℤ), 3 * a^2 + 5 * b^2 = 345 ∧ a + b = 13 := by
  sorry

end max_sum_of_squares_l1222_122220


namespace intersection_of_A_and_B_l1222_122284

-- Define sets A and B
def A : Set ℝ := {x | x - 1 ≤ 0}
def B : Set ℝ := {x | x^2 - 4*x ≤ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l1222_122284


namespace original_amount_calculation_l1222_122280

theorem original_amount_calculation (total : ℚ) : 
  (3/4 : ℚ) * total - (1/5 : ℚ) * total = 132 → total = 240 := by
  sorry

end original_amount_calculation_l1222_122280


namespace right_triangle_area_right_triangle_area_proof_l1222_122221

theorem right_triangle_area : ℕ → ℕ → ℕ → Prop :=
  fun a b c =>
    (a * a + b * b = c * c) →  -- Pythagorean theorem
    (2 * b * b - 23 * b + 11 = 0) →  -- One leg satisfies the equation
    (a > 0 ∧ b > 0 ∧ c > 0) →  -- All sides are positive
    ((a * b) / 2 = 330)  -- Area of the triangle

-- The proof of this theorem
theorem right_triangle_area_proof :
  ∃ (a b c : ℕ), right_triangle_area a b c :=
sorry

end right_triangle_area_right_triangle_area_proof_l1222_122221


namespace min_value_sum_l1222_122211

/-- Given two circles C₁ and C₂, where C₁ always bisects the circumference of C₂,
    prove that the minimum value of 1/m + 2/n is 3 -/
theorem min_value_sum (m n : ℝ) : m > 0 → n > 0 → 
  (∀ x y : ℝ, (x - m)^2 + (y - 2*n)^2 = m^2 + 4*n^2 + 10 → 
              (x + 1)^2 + (y + 1)^2 = 2 → 
              ∃ k : ℝ, (m + 1)*x + (2*n + 1)*y + 5 = k * ((x + 1)^2 + (y + 1)^2 - 2)) →
  (1 / m + 2 / n) ≥ 3 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 1 / m₀ + 2 / n₀ = 3 :=
by sorry

end min_value_sum_l1222_122211


namespace ellipse_sum_coordinates_and_axes_l1222_122297

/-- Represents an ellipse on a coordinate plane -/
structure Ellipse where
  center : ℝ × ℝ
  semiMajorAxis : ℝ
  semiMinorAxis : ℝ

/-- Theorem: For the given ellipse, h + k + a + b = 9 -/
theorem ellipse_sum_coordinates_and_axes (e : Ellipse) 
  (h_center : e.center = (1, -3))
  (h_major : e.semiMajorAxis = 7)
  (h_minor : e.semiMinorAxis = 4) :
  e.center.1 + e.center.2 + e.semiMajorAxis + e.semiMinorAxis = 9 := by
  sorry

end ellipse_sum_coordinates_and_axes_l1222_122297


namespace nadine_pebbles_l1222_122281

def white_pebbles : ℕ := 20

def red_pebbles : ℕ := white_pebbles / 2

def total_pebbles : ℕ := white_pebbles + red_pebbles

theorem nadine_pebbles : total_pebbles = 30 := by
  sorry

end nadine_pebbles_l1222_122281


namespace z_squared_and_modulus_l1222_122271

-- Define the complex number z
def z : ℂ := 5 + 3 * Complex.I

-- Theorem statement
theorem z_squared_and_modulus :
  z ^ 2 = 16 + 30 * Complex.I ∧ Complex.abs (z ^ 2) = 34 := by
  sorry

end z_squared_and_modulus_l1222_122271


namespace escalator_steps_l1222_122223

theorem escalator_steps (n : ℕ) : 
  n % 2 = 1 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 6 = 5 ∧ 
  n % 7 = 6 ∧ 
  n % 20 = 19 ∧ 
  n < 1000 → 
  n = 839 :=
by sorry

end escalator_steps_l1222_122223


namespace fourth_term_of_geometric_sequence_l1222_122286

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence satisfying certain conditions, its 4th term equals 8. -/
theorem fourth_term_of_geometric_sequence (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_sum : a 6 + a 2 = 34) 
    (h_diff : a 6 - a 2 = 30) : 
  a 4 = 8 := by
  sorry


end fourth_term_of_geometric_sequence_l1222_122286


namespace max_revenue_l1222_122228

def price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def revenue (t : ℕ) : ℝ :=
  price t * sales_volume t

theorem max_revenue :
  ∃ (t : ℕ), t = 25 ∧ revenue t = 1125 ∧
  ∀ (s : ℕ), 0 < s ∧ s ≤ 30 → revenue s ≤ revenue t := by
  sorry

end max_revenue_l1222_122228


namespace min_cost_theorem_l1222_122262

/-- Represents the cost of tickets in rubles -/
def N : ℝ := sorry

/-- The number of southern cities -/
def num_southern_cities : ℕ := 4

/-- The number of northern cities -/
def num_northern_cities : ℕ := 5

/-- The cost of a one-way ticket between any two connected cities -/
def one_way_cost : ℝ := N

/-- The cost of a round-trip ticket between any two connected cities -/
def round_trip_cost : ℝ := 1.6 * N

/-- A route represents a sequence of city visits -/
def Route := List ℕ

/-- Predicate to check if a route is valid according to the problem constraints -/
def is_valid_route (r : Route) : Prop := sorry

/-- The cost of a given route -/
def route_cost (r : Route) : ℝ := sorry

/-- Theorem stating the minimum cost to visit all southern cities and return to the start -/
theorem min_cost_theorem :
  ∀ (r : Route), is_valid_route r →
    route_cost r ≥ 6.4 * N ∧
    ∃ (optimal_route : Route), 
      is_valid_route optimal_route ∧ 
      route_cost optimal_route = 6.4 * N :=
by sorry

end min_cost_theorem_l1222_122262


namespace exists_unprovable_by_induction_l1222_122201

-- Define a proposition that represents a mathematical statement
def MathStatement : Type := Prop

-- Define a function that represents the ability to prove a statement by induction
def ProvableByInduction (s : MathStatement) : Prop := sorry

-- Theorem: There exists a true mathematical statement that cannot be proven by induction
theorem exists_unprovable_by_induction : 
  ∃ (s : MathStatement), s ∧ ¬(ProvableByInduction s) := by sorry

end exists_unprovable_by_induction_l1222_122201


namespace four_intersection_points_range_l1222_122239

/-- Parabola C: x^2 = 4y -/
def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

/-- Circle M: x^2 + (y-4)^2 = r^2 -/
def circle_M (x y r : ℝ) : Prop := x^2 + (y-4)^2 = r^2

/-- The number of intersection points between C and M -/
noncomputable def intersection_count (r : ℝ) : ℕ := sorry

theorem four_intersection_points_range (r : ℝ) :
  r > 0 ∧ intersection_count r = 4 → 2 * Real.sqrt 3 < r ∧ r < 4 := by sorry

end four_intersection_points_range_l1222_122239


namespace a_51_value_l1222_122237

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) - a n = 2

theorem a_51_value (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 51 = 101 := by
  sorry

end a_51_value_l1222_122237


namespace last_card_in_box_three_l1222_122247

/-- The number of boxes -/
def num_boxes : ℕ := 7

/-- The total number of cards -/
def total_cards : ℕ := 2015

/-- The length of a complete cycle -/
def cycle_length : ℕ := 12

/-- Function to determine the box number for a given card number -/
def box_number (card : ℕ) : ℕ :=
  let cycle_position := card % cycle_length
  if cycle_position ≤ num_boxes
  then cycle_position
  else 2 * num_boxes - cycle_position

/-- Theorem stating that the 2015th card will be placed in box 3 -/
theorem last_card_in_box_three :
  box_number total_cards = 3 := by
  sorry


end last_card_in_box_three_l1222_122247


namespace measure_of_angle_ABC_l1222_122204

-- Define the angles
def angle_ABC : ℝ := sorry
def angle_ABD : ℝ := 30
def angle_CBD : ℝ := 90

-- State the theorem
theorem measure_of_angle_ABC :
  angle_ABC = 60 ∧ 
  angle_CBD = 90 ∧ 
  angle_ABD = 30 ∧ 
  angle_ABC + angle_ABD + angle_CBD = 180 :=
by sorry

end measure_of_angle_ABC_l1222_122204


namespace x_coord_difference_at_y_10_l1222_122283

/-- Represents a line in 2D space -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Calculates the x-coordinate for a given y-coordinate on a line -/
def xCoordAtY (l : Line) (y : ℚ) : ℚ :=
  (y - l.intercept) / l.slope

/-- Creates a line from two points -/
def lineFromPoints (x1 y1 x2 y2 : ℚ) : Line where
  slope := (y2 - y1) / (x2 - x1)
  intercept := y1 - (y2 - y1) / (x2 - x1) * x1

theorem x_coord_difference_at_y_10 : 
  let p := lineFromPoints 0 3 4 0
  let q := lineFromPoints 0 1 8 0
  let xp := xCoordAtY p 10
  let xq := xCoordAtY q 10
  |xp - xq| = 188 / 3 := by
    sorry

end x_coord_difference_at_y_10_l1222_122283


namespace polygon_exterior_angle_72_l1222_122289

theorem polygon_exterior_angle_72 (n : ℕ) (exterior_angle : ℝ) :
  exterior_angle = 72 →
  (360 : ℝ) / exterior_angle = n →
  n = 5 ∧ (n - 2) * 180 = 540 := by
  sorry

end polygon_exterior_angle_72_l1222_122289


namespace maximal_cross_section_area_l1222_122246

/-- A triangular prism with vertical edges parallel to the z-axis -/
structure TriangularPrism where
  base : Set (ℝ × ℝ)
  height : ℝ → ℝ

/-- The cross-section of the prism is an equilateral triangle with side length 8 -/
def equilateralBase (p : TriangularPrism) : Prop :=
  ∃ (A B C : ℝ × ℝ),
    A ∈ p.base ∧ B ∈ p.base ∧ C ∈ p.base ∧
    dist A B = 8 ∧ dist B C = 8 ∧ dist C A = 8

/-- The plane that intersects the prism -/
def intersectingPlane (x y z : ℝ) : Prop :=
  3 * x - 5 * y + 2 * z = 30

/-- The cross-section formed by the intersection of the prism and the plane -/
def crossSection (p : TriangularPrism) : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | (x, y) ∈ p.base ∧ z = p.height x ∧ intersectingPlane x y z}

/-- The area of the cross-section -/
noncomputable def crossSectionArea (p : TriangularPrism) : ℝ :=
  sorry

/-- The main theorem stating that the maximal area of the cross-section is 92 -/
theorem maximal_cross_section_area (p : TriangularPrism) 
  (h : equilateralBase p) : 
  crossSectionArea p ≤ 92 ∧ ∃ (p' : TriangularPrism), equilateralBase p' ∧ crossSectionArea p' = 92 :=
sorry

end maximal_cross_section_area_l1222_122246


namespace cosine_of_specific_line_l1222_122294

/-- A line in 2D space represented by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The inclination angle of a line -/
def inclinationAngle (l : ParametricLine) : ℝ := sorry

/-- Cosine of the inclination angle of a line -/
def cosInclinationAngle (l : ParametricLine) : ℝ := sorry

theorem cosine_of_specific_line :
  let l : ParametricLine := {
    x := λ t => 1 + 3 * t,
    y := λ t => 2 - 4 * t
  }
  cosInclinationAngle l = -3/5 := by sorry

end cosine_of_specific_line_l1222_122294


namespace mark_gigs_total_duration_l1222_122205

/-- Represents the duration of Mark's gigs over two weeks -/
def MarkGigsDuration : ℕ :=
  let days_in_two_weeks : ℕ := 2 * 7
  let gigs_count : ℕ := days_in_two_weeks / 2
  let short_song_duration : ℕ := 5
  let long_song_duration : ℕ := 2 * short_song_duration
  let gig_duration : ℕ := 2 * short_song_duration + long_song_duration
  gigs_count * gig_duration

theorem mark_gigs_total_duration :
  MarkGigsDuration = 140 := by
  sorry

end mark_gigs_total_duration_l1222_122205


namespace biology_enrollment_percentage_l1222_122265

theorem biology_enrollment_percentage (total_students : ℕ) (not_enrolled : ℕ) : 
  total_students = 880 →
  not_enrolled = 528 →
  (((total_students - not_enrolled) : ℚ) / total_students) * 100 = 40 := by
sorry

end biology_enrollment_percentage_l1222_122265


namespace officer_average_salary_l1222_122263

/-- Proves that the average salary of officers is 420 Rs/month given the specified conditions -/
theorem officer_average_salary
  (total_employees : ℕ)
  (officers : ℕ)
  (non_officers : ℕ)
  (average_salary : ℚ)
  (non_officer_salary : ℚ)
  (h1 : total_employees = officers + non_officers)
  (h2 : total_employees = 465)
  (h3 : officers = 15)
  (h4 : non_officers = 450)
  (h5 : average_salary = 120)
  (h6 : non_officer_salary = 110) :
  (total_employees * average_salary - non_officers * non_officer_salary) / officers = 420 :=
by sorry

end officer_average_salary_l1222_122263


namespace brother_ages_l1222_122266

theorem brother_ages (B E : ℕ) (h1 : B = E + 4) (h2 : B + E = 28) : E = 12 := by
  sorry

end brother_ages_l1222_122266


namespace solution_set_equality_l1222_122296

theorem solution_set_equality (m : ℝ) : 
  (Set.Iio m = {x : ℝ | 2 * x + 1 < 5}) → m ≤ 2 := by
  sorry

end solution_set_equality_l1222_122296


namespace hyperbola_through_C_l1222_122274

/-- Given a point A on the parabola y = x^2 and a point B such that OB is perpendicular to OA,
    prove that the point C formed by the rectangle AOBC lies on the hyperbola y = -2/x -/
theorem hyperbola_through_C (A B C : ℝ × ℝ) : 
  A.1 = -1/2 ∧ A.2 = 1/4 ∧                          -- A is (-1/2, 1/4)
  A.2 = A.1^2 ∧                                     -- A is on the parabola y = x^2
  B.1 = 2 ∧ B.2 = 4 ∧                               -- B is (2, 4)
  (B.2 - 0) / (B.1 - 0) = -(A.2 - 0) / (A.1 - 0) ∧  -- OB ⟂ OA
  C.1 = A.1 ∧ C.2 = B.2                             -- C forms rectangle AOBC
  →
  C.2 = -2 / C.1                                    -- C is on the hyperbola y = -2/x
:= by sorry

end hyperbola_through_C_l1222_122274


namespace rectangle_area_l1222_122275

/-- Given a rectangle where the length is four times the width and the perimeter is 200 cm,
    prove that its area is 1600 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 200 →
  l * w = 1600 := by sorry

end rectangle_area_l1222_122275


namespace repeating_decimal_sum_l1222_122215

def repeating_decimal_to_fraction (d : ℚ) : ℚ := d

theorem repeating_decimal_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : Nat.gcd a b = 1) (h4 : repeating_decimal_to_fraction (35/99 : ℚ) = a / b) : 
  a + b = 134 := by
  sorry

end repeating_decimal_sum_l1222_122215


namespace complex_norm_problem_l1222_122210

theorem complex_norm_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 17)
  (h2 : Complex.abs (z + 3 * w) = 4)
  (h3 : Complex.abs (z + w) = 6) :
  Complex.abs z = 5 := by sorry

end complex_norm_problem_l1222_122210


namespace book_cost_price_l1222_122209

theorem book_cost_price (selling_price profit_percentage : ℝ)
  (h1 : profit_percentage = 0.10)
  (h2 : selling_price = (1 + profit_percentage) * 2800)
  (h3 : selling_price + 140 = (1 + 0.15) * 2800) :
  2800 = (selling_price - (1 + profit_percentage) * 2800) / profit_percentage :=
by sorry

end book_cost_price_l1222_122209


namespace football_team_right_handed_players_l1222_122235

theorem football_team_right_handed_players (total_players : ℕ) (throwers : ℕ) :
  total_players = 70 →
  throwers = 37 →
  (total_players - throwers) % 3 = 0 →
  59 = throwers + (total_players - throwers) * 2 / 3 :=
by sorry

end football_team_right_handed_players_l1222_122235


namespace cubic_equation_roots_l1222_122257

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ (r₁ r₂ r₃ : ℕ+), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    (∀ (x : ℝ), x^3 - 6*x^2 + p*x - q = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃))) →
  p + q = 17 := by
sorry

end cubic_equation_roots_l1222_122257


namespace roots_sum_zero_l1222_122227

/-- Given two quadratic trinomials with specific properties, prove their product's roots sum to 0 -/
theorem roots_sum_zero (a b : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₃ ≠ x₄ ∧ 
    (∀ x : ℝ, x^2 + a*x + b = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
    (∀ x : ℝ, x^2 + b*x + a = 0 ↔ (x = x₃ ∨ x = x₄))) →
  (∃ y₁ y₂ y₃ : ℝ, y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₂ ≠ y₃ ∧
    (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + b*x + a) = 0 ↔ (x = y₁ ∨ x = y₂ ∨ x = y₃))) →
  (∃ z₁ z₂ z₃ : ℝ, 
    (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + b*x + a) = 0 ↔ (x = z₁ ∨ x = z₂ ∨ x = z₃)) ∧
    z₁ + z₂ + z₃ = 0) :=
by sorry

end roots_sum_zero_l1222_122227


namespace root_product_expression_l1222_122251

theorem root_product_expression (p q : ℝ) 
  (α β γ δ : ℂ) 
  (hαβ : α^2 + p*α = 1 ∧ β^2 + p*β = 1) 
  (hγδ : γ^2 + q*γ = -1 ∧ δ^2 + q*δ = -1) : 
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = p^2 - q^2 := by
  sorry

end root_product_expression_l1222_122251


namespace largest_guaranteed_divisor_l1222_122279

def die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def is_valid_roll (roll : Finset ℕ) : Prop :=
  roll ⊆ die_numbers ∧ roll.card = 7

def roll_product (roll : Finset ℕ) : ℕ :=
  roll.prod id

theorem largest_guaranteed_divisor :
  ∀ roll : Finset ℕ, is_valid_roll roll →
    ∃ m : ℕ, m = 192 ∧ 
      (∀ n : ℕ, n > 192 → ¬(∀ r : Finset ℕ, is_valid_roll r → n ∣ roll_product r)) ∧
      (192 ∣ roll_product roll) :=
by sorry

end largest_guaranteed_divisor_l1222_122279


namespace sale_price_calculation_l1222_122253

def original_price : ℝ := 100
def discount_percentage : ℝ := 25

theorem sale_price_calculation :
  let discount_amount := (discount_percentage / 100) * original_price
  let sale_price := original_price - discount_amount
  sale_price = 75 := by sorry

end sale_price_calculation_l1222_122253


namespace gym_students_count_l1222_122202

theorem gym_students_count :
  ∀ (students_on_floor : ℕ) (total_students : ℕ),
    -- 4 students are on the bleachers
    total_students = students_on_floor + 4 →
    -- The ratio of students on the floor to total students is 11:13
    (students_on_floor : ℚ) / total_students = 11 / 13 →
    -- The total number of students is 26
    total_students = 26 := by
  sorry

end gym_students_count_l1222_122202


namespace find_a_l1222_122259

theorem find_a (x y a : ℤ) 
  (eq1 : 3 * x + y = 40)
  (eq2 : a * x - y = 20)
  (eq3 : 3 * y^2 = 48) :
  a = 2 := by
  sorry

end find_a_l1222_122259


namespace undefined_fraction_roots_product_l1222_122218

theorem undefined_fraction_roots_product : ∃ (r₁ r₂ : ℝ), 
  (r₁^2 - 4*r₁ - 12 = 0) ∧ 
  (r₂^2 - 4*r₂ - 12 = 0) ∧ 
  (r₁ ≠ r₂) ∧
  (r₁ * r₂ = -12) := by
  sorry

end undefined_fraction_roots_product_l1222_122218


namespace pizza_class_size_l1222_122214

/-- Proves that the number of students in a class is 68, given the pizza ordering scenario. -/
theorem pizza_class_size :
  let pizza_slices : ℕ := 18  -- Number of slices in a large pizza
  let total_pizzas : ℕ := 6   -- Total number of pizzas ordered
  let cheese_leftover : ℕ := 8  -- Number of cheese slices leftover
  let onion_leftover : ℕ := 4   -- Number of onion slices leftover
  let cheese_per_student : ℕ := 2  -- Number of cheese slices per student
  let onion_per_student : ℕ := 1   -- Number of onion slices per student

  let total_slices : ℕ := pizza_slices * total_pizzas
  let used_cheese : ℕ := total_slices - cheese_leftover
  let used_onion : ℕ := total_slices - onion_leftover

  (∃ (num_students : ℕ),
    num_students * cheese_per_student = used_cheese ∧
    num_students * onion_per_student = used_onion) →
  (∃! (num_students : ℕ), num_students = 68) :=
by sorry

end pizza_class_size_l1222_122214


namespace more_girls_than_boys_in_class_l1222_122225

theorem more_girls_than_boys_in_class (num_students : ℕ) (num_teachers : ℕ) 
  (h_students : num_students = 42)
  (h_teachers : num_teachers = 6)
  (h_ratio : ∃ (x : ℕ), num_students = 7 * x ∧ 3 * x = num_boys ∧ 4 * x = num_girls) :
  num_girls - num_boys = 6 :=
by sorry

end more_girls_than_boys_in_class_l1222_122225


namespace union_of_M_and_N_l1222_122241

def M : Set ℝ := {x | x^2 + 2*x - 3 = 0}
def N : Set ℝ := {-1, 2, 3}

theorem union_of_M_and_N : M ∪ N = {-1, 1, 2, -3, 3} := by sorry

end union_of_M_and_N_l1222_122241


namespace gcd_of_squares_l1222_122288

theorem gcd_of_squares : Nat.gcd (111^2 + 222^2 + 333^2) (110^2 + 221^2 + 334^2) = 3 := by
  sorry

end gcd_of_squares_l1222_122288


namespace song_guessing_game_theorem_l1222_122217

/-- The Song Guessing Game -/
structure SongGuessingGame where
  /-- Probability of correctly guessing a song from group A -/
  probA : ℝ
  /-- Probability of correctly guessing a song from group B -/
  probB : ℝ
  /-- Number of songs played from group A -/
  numA : ℕ
  /-- Number of songs played from group B -/
  numB : ℕ
  /-- Points earned for correctly guessing a song from group A -/
  pointsA : ℕ
  /-- Points earned for correctly guessing a song from group B -/
  pointsB : ℕ

/-- The probability of guessing at least 2 song titles correctly -/
def probAtLeastTwo (game : SongGuessingGame) : ℝ := sorry

/-- The expectation of the total score -/
def expectedScore (game : SongGuessingGame) : ℝ := sorry

/-- Main theorem about the Song Guessing Game -/
theorem song_guessing_game_theorem (game : SongGuessingGame) 
  (h1 : game.probA = 2/3)
  (h2 : game.probB = 1/2)
  (h3 : game.numA = 2)
  (h4 : game.numB = 2)
  (h5 : game.pointsA = 1)
  (h6 : game.pointsB = 2) :
  probAtLeastTwo game = 29/36 ∧ expectedScore game = 10/3 := by sorry

end song_guessing_game_theorem_l1222_122217


namespace square_sum_equality_l1222_122252

theorem square_sum_equality (x y P Q : ℝ) :
  x^2 + y^2 = (x + y)^2 + P ∧ x^2 + y^2 = (x - y)^2 + Q →
  P = -2*x*y ∧ Q = 2*x*y := by
sorry

end square_sum_equality_l1222_122252


namespace original_earnings_before_raise_l1222_122229

theorem original_earnings_before_raise (new_earnings : ℝ) (percent_increase : ℝ) 
  (h1 : new_earnings = 80)
  (h2 : percent_increase = 60) :
  let original_earnings := new_earnings / (1 + percent_increase / 100)
  original_earnings = 50 := by
sorry

end original_earnings_before_raise_l1222_122229


namespace functional_equation_implies_ge_l1222_122276

/-- A function f: ℝ⁺ → ℝ⁺ satisfying f(f(x)) + x = f(2x) for all x > 0 -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f x > 0 ∧ f (f x) + x = f (2 * x)

/-- Theorem: If f satisfies the functional equation, then f(x) ≥ x for all x > 0 -/
theorem functional_equation_implies_ge (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x > 0, f x ≥ x := by
  sorry

end functional_equation_implies_ge_l1222_122276


namespace arithmetic_sequence_problem_l1222_122285

/-- An arithmetic sequence -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The theorem to prove -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h : arithmeticSequence a) 
    (h_sum : a 5 + a 10 = 12) : 
  3 * a 7 + a 9 = 24 := by
  sorry

end arithmetic_sequence_problem_l1222_122285


namespace max_yellow_balls_l1222_122260

/-- Represents the total number of balls -/
def n : ℕ := 91

/-- Represents the number of yellow balls in the first 70 picked -/
def initial_yellow : ℕ := 63

/-- Represents the total number of balls initially picked -/
def initial_total : ℕ := 70

/-- Represents the number of yellow balls in each subsequent batch of 7 -/
def batch_yellow : ℕ := 5

/-- Represents the total number of balls in each subsequent batch -/
def batch_total : ℕ := 7

/-- The minimum percentage of yellow balls required -/
def min_percentage : ℚ := 85 / 100

theorem max_yellow_balls :
  n = initial_total + batch_total * ((n - initial_total) / batch_total) ∧
  (initial_yellow + batch_yellow * ((n - initial_total) / batch_total)) / n ≥ min_percentage ∧
  ∀ m : ℕ, m > n →
    (initial_yellow + batch_yellow * ((m - initial_total) / batch_total)) / m < min_percentage :=
by sorry

end max_yellow_balls_l1222_122260


namespace greatest_three_digit_multiple_of_17_l1222_122203

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, n = 986 ∧ 
  n % 17 = 0 ∧ 
  n ≥ 100 ∧ n < 1000 ∧
  ∀ m : ℕ, m % 17 = 0 → m ≥ 100 → m < 1000 → m ≤ n :=
by sorry

end greatest_three_digit_multiple_of_17_l1222_122203


namespace percentage_of_a_l1222_122212

theorem percentage_of_a (a b c : ℝ) (P : ℝ) : 
  (P / 100) * a = 8 →
  (8 / 100) * b = 4 →
  c = b / a →
  P = 16 := by
sorry

end percentage_of_a_l1222_122212


namespace sugar_solution_volume_l1222_122245

/-- Given a sugar solution, prove that the initial volume was 3 liters -/
theorem sugar_solution_volume (V : ℝ) : 
  V > 0 → -- Initial volume is positive
  (0.4 * V) / (V + 1) = 0.30000000000000004 → -- New concentration after adding 1 liter of water
  V = 3 := by
sorry

end sugar_solution_volume_l1222_122245


namespace scientific_notation_of_161000_l1222_122249

/-- The scientific notation representation of 161,000 -/
theorem scientific_notation_of_161000 : 161000 = 1.61 * (10 ^ 5) := by
  sorry

end scientific_notation_of_161000_l1222_122249


namespace cube_edge_length_l1222_122292

theorem cube_edge_length (material_volume : ℕ) (num_cubes : ℕ) (edge_length : ℕ) : 
  material_volume = 12 * 18 * 6 →
  num_cubes = 48 →
  material_volume = num_cubes * edge_length * edge_length * edge_length →
  edge_length = 3 := by
sorry

end cube_edge_length_l1222_122292


namespace triangle_side_length_l1222_122298

theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  a = 5 →
  b = 7 →
  B = π / 3 →  -- 60° in radians
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  c = 8 := by
sorry

end triangle_side_length_l1222_122298


namespace paint_cube_cost_l1222_122222

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem paint_cube_cost
  (paint_cost_per_kg : ℝ)
  (paint_coverage_per_kg : ℝ)
  (cube_side_length : ℝ)
  (h1 : paint_cost_per_kg = 60)
  (h2 : paint_coverage_per_kg = 20)
  (h3 : cube_side_length = 10) :
  cube_side_length ^ 2 * 6 / paint_coverage_per_kg * paint_cost_per_kg = 1800 :=
by sorry

end paint_cube_cost_l1222_122222


namespace circle_area_tripled_l1222_122258

theorem circle_area_tripled (r m : ℝ) : 
  (r > 0) → (m > 0) → (π * (r + m)^2 = 3 * π * r^2) → (r = m * (Real.sqrt 3 - 1) / 2) := by
  sorry

end circle_area_tripled_l1222_122258


namespace tractors_count_l1222_122299

/-- Represents the number of tractors initially ploughing the field -/
def T : ℕ := sorry

/-- The area of the field in hectares -/
def field_area : ℕ := sorry

/-- Each tractor ploughs this many hectares per day -/
def hectares_per_tractor_per_day : ℕ := 120

/-- The number of days it takes all tractors to plough the field -/
def days_all_tractors : ℕ := 4

/-- The number of tractors remaining after two are removed -/
def remaining_tractors : ℕ := 4

/-- The number of days it takes the remaining tractors to plough the field -/
def days_remaining_tractors : ℕ := 5

theorem tractors_count :
  (T * hectares_per_tractor_per_day * days_all_tractors = field_area) ∧
  (remaining_tractors * hectares_per_tractor_per_day * days_remaining_tractors = field_area) ∧
  (T = remaining_tractors + 2) →
  T = 10 := by sorry

end tractors_count_l1222_122299


namespace horse_cow_pricing_system_l1222_122208

theorem horse_cow_pricing_system (x y : ℝ) :
  (4 * x + 6 * y = 48 ∧ 3 * x + 5 * y = 38) ↔
  (∃ (horse_price cow_price : ℝ),
    horse_price = x ∧
    cow_price = y ∧
    4 * horse_price + 6 * cow_price = 48 ∧
    3 * horse_price + 5 * cow_price = 38) :=
by sorry

end horse_cow_pricing_system_l1222_122208


namespace unique_valid_pair_l1222_122206

def has_one_solution (a b c : ℝ) : Prop :=
  (b^2 - 4*a*c = 0) ∧ (a ≠ 0)

def valid_pair (b c : ℕ+) : Prop :=
  has_one_solution 1 (2*b) (2*c) ∧ has_one_solution 1 (3*c) (3*b)

theorem unique_valid_pair : ∃! p : ℕ+ × ℕ+, valid_pair p.1 p.2 :=
sorry

end unique_valid_pair_l1222_122206


namespace box_value_proof_l1222_122240

theorem box_value_proof : ∃ x : ℝ, (1 + 1.1 + 1.11 + x = 4.44) ∧ (x = 1.23) := by
  sorry

end box_value_proof_l1222_122240


namespace no_integer_solution_l1222_122293

theorem no_integer_solution : ¬ ∃ (x : ℤ), x + (2*x + 33) + (3*x - 24) = 100 := by
  sorry

end no_integer_solution_l1222_122293


namespace zibo_barbecue_analysis_l1222_122255

/-- Contingency table data --/
structure ContingencyData where
  male_very_like : ℕ
  male_average : ℕ
  female_very_like : ℕ
  female_average : ℕ

/-- Chi-square test result --/
inductive ChiSquareResult
  | Significant
  | NotSignificant

/-- Distribution of ξ --/
def DistributionXi := List (ℕ × ℚ)

/-- Theorem statement --/
theorem zibo_barbecue_analysis 
  (data : ContingencyData)
  (total_sample : ℕ)
  (chi_square_formula : ContingencyData → ℝ)
  (chi_square_critical : ℝ)
  (calculate_distribution : ContingencyData → DistributionXi)
  (calculate_expectation : DistributionXi → ℚ)
  (h_total : data.male_very_like + data.male_average + data.female_very_like + data.female_average = total_sample)
  (h_female_total : data.female_very_like + data.female_average = 100)
  (h_average_total : data.male_average + data.female_average = 70)
  (h_female_very_like : data.female_very_like = 2 * data.male_average)
  : 
  let chi_square_value := chi_square_formula data
  let result := if chi_square_value < chi_square_critical then ChiSquareResult.NotSignificant else ChiSquareResult.Significant
  let distribution := calculate_distribution data
  let expectation := calculate_expectation distribution
  result = ChiSquareResult.NotSignificant ∧ expectation = 17 / 6 := by
  sorry

end zibo_barbecue_analysis_l1222_122255


namespace elizabeth_ate_four_bananas_l1222_122256

/-- The number of bananas Elizabeth ate -/
def bananas_eaten (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Proof that Elizabeth ate 4 bananas -/
theorem elizabeth_ate_four_bananas :
  let initial := 12
  let remaining := 8
  bananas_eaten initial remaining = 4 := by
  sorry

end elizabeth_ate_four_bananas_l1222_122256


namespace tissue_paper_count_l1222_122272

theorem tissue_paper_count (remaining : ℕ) (used : ℕ) (initial : ℕ) : 
  remaining = 93 → used = 4 → initial = remaining + used :=
by sorry

end tissue_paper_count_l1222_122272


namespace sum_of_cubes_of_roots_l1222_122277

theorem sum_of_cubes_of_roots (a b : ℝ) (α β : ℝ) : 
  (α^2 + a*α + b = 0) → (β^2 + a*β + b = 0) → α^3 + β^3 = -(a^3 - 3*a*b) := by
  sorry

end sum_of_cubes_of_roots_l1222_122277


namespace ryan_solution_unique_l1222_122282

/-- Represents the solution to Ryan's grocery purchase --/
structure GrocerySolution where
  corn : ℝ
  beans : ℝ
  rice : ℝ

/-- Checks if a given solution satisfies all the problem conditions --/
def is_valid_solution (s : GrocerySolution) : Prop :=
  s.corn + s.beans + s.rice = 30 ∧
  1.20 * s.corn + 0.60 * s.beans + 0.80 * s.rice = 24 ∧
  s.beans = s.rice

/-- The unique solution to the problem --/
def ryan_solution : GrocerySolution :=
  { corn := 6, beans := 12, rice := 12 }

/-- Theorem stating that ryan_solution is the only valid solution --/
theorem ryan_solution_unique :
  is_valid_solution ryan_solution ∧
  ∀ s : GrocerySolution, is_valid_solution s → s = ryan_solution :=
sorry

end ryan_solution_unique_l1222_122282


namespace intersection_P_complement_Q_l1222_122234

open Set Real

def P : Set ℝ := {x | x^2 + 2*x - 3 = 0}
def Q : Set ℝ := {x | log x < 1}

theorem intersection_P_complement_Q : P ∩ (univ \ Q) = {-3} := by sorry

end intersection_P_complement_Q_l1222_122234


namespace tutor_schedule_lcm_l1222_122261

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 10 11))) = 1320 := by
  sorry

end tutor_schedule_lcm_l1222_122261


namespace regular_polygon_perimeter_l1222_122268

/-- Given a regular polygon with central angle 45° and side length 5, its perimeter is 40. -/
theorem regular_polygon_perimeter (central_angle : ℝ) (side_length : ℝ) :
  central_angle = 45 →
  side_length = 5 →
  (360 / central_angle) * side_length = 40 := by
  sorry

end regular_polygon_perimeter_l1222_122268


namespace combination_ratio_problem_l1222_122243

theorem combination_ratio_problem (m n : ℕ) : 
  (Nat.choose (n + 1) (m + 1) : ℚ) / (Nat.choose (n + 1) m : ℚ) = 5 / 5 ∧
  (Nat.choose (n + 1) m : ℚ) / (Nat.choose (n + 1) m : ℚ) = 5 / 5 ∧
  (Nat.choose (n + 1) m : ℚ) / (Nat.choose (n + 1) (m - 1) : ℚ) = 5 / 3 →
  m = 3 ∧ n = 6 := by
sorry

end combination_ratio_problem_l1222_122243


namespace probability_two_successes_four_trials_l1222_122273

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The probability of exactly k successes in n trials with probability p of success per trial -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The theorem stating the probability of 2 successes in 4 trials with 0.3 probability of success -/
theorem probability_two_successes_four_trials :
  binomialProbability 4 2 0.3 = 0.2646 := by sorry

end probability_two_successes_four_trials_l1222_122273


namespace units_digit_of_7_power_75_plus_6_l1222_122232

-- Define the function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the function to get the units digit of 7^n
def unitsDigitOf7Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case should never occur

-- Theorem statement
theorem units_digit_of_7_power_75_plus_6 :
  unitsDigit (unitsDigitOf7Power 75 + 6) = 9 := by
  sorry

end units_digit_of_7_power_75_plus_6_l1222_122232


namespace abc_perfect_cube_l1222_122269

theorem abc_perfect_cube (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : ∃ (n : ℤ), (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a = n) :
  ∃ (k : ℤ), a * b * c = k^3 := by
sorry

end abc_perfect_cube_l1222_122269


namespace perimeter_inscribable_equivalence_l1222_122250

/-- Triangle represented by its side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a line segment intersecting two sides of a triangle -/
structure IntersectingLine (T : Triangle) where
  A' : ℝ  -- Distance from A to A' on AC
  B' : ℝ  -- Distance from B to B' on BC

/-- Condition for the perimeter of the inner triangle -/
def perimeterCondition (T : Triangle) (L : IntersectingLine T) : Prop :=
  L.A' + L.B' + (T.c - L.A' - L.B') = T.a + T.b - T.c

/-- Condition for the quadrilateral to be inscribable -/
def inscribableCondition (T : Triangle) (L : IntersectingLine T) : Prop :=
  T.c + (T.a + T.b - T.c - (L.A' + L.B')) = (T.a - L.A') + (T.b - L.B')

theorem perimeter_inscribable_equivalence (T : Triangle) (L : IntersectingLine T) :
  perimeterCondition T L ↔ inscribableCondition T L := by sorry

end perimeter_inscribable_equivalence_l1222_122250


namespace water_fraction_in_mixture_l1222_122226

theorem water_fraction_in_mixture (alcohol_to_water_ratio : ℚ) :
  alcohol_to_water_ratio = 2/3 →
  (water_volume / (water_volume + alcohol_volume) = 3/5) :=
by
  sorry

end water_fraction_in_mixture_l1222_122226


namespace binomial_sum_equals_120_l1222_122236

theorem binomial_sum_equals_120 : 
  Nat.choose 8 2 + Nat.choose 8 3 + Nat.choose 9 2 = 120 := by
  sorry

end binomial_sum_equals_120_l1222_122236


namespace symmetric_point_coordinates_l1222_122264

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis. -/
def symmetricXAxis (p p' : Point2D) : Prop :=
  p'.x = p.x ∧ p'.y = -p.y

/-- Theorem: If P(-3, 2) is symmetric to P' with respect to the x-axis,
    then P' has coordinates (-3, -2). -/
theorem symmetric_point_coordinates :
  let P : Point2D := ⟨-3, 2⟩
  let P' : Point2D := ⟨-3, -2⟩
  symmetricXAxis P P' → P' = ⟨-3, -2⟩ :=
by
  sorry

end symmetric_point_coordinates_l1222_122264


namespace quadratic_completing_square_l1222_122224

theorem quadratic_completing_square (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → b + c = 5 := by
sorry

end quadratic_completing_square_l1222_122224


namespace cube_inequality_iff_l1222_122238

theorem cube_inequality_iff (a b : ℝ) : a > b ↔ a^3 > b^3 := by
  sorry

end cube_inequality_iff_l1222_122238


namespace cos_shift_odd_condition_l1222_122248

open Real

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem cos_shift_odd_condition (φ : ℝ) :
  (φ = π / 2 → is_odd_function (λ x => cos (x + φ))) ∧
  (∃ φ', φ' ≠ π / 2 ∧ is_odd_function (λ x => cos (x + φ'))) :=
sorry

end cos_shift_odd_condition_l1222_122248


namespace triangle_theorem_l1222_122278

/-- Triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) (h1 : t.a * (Real.cos (t.C / 2))^2 + t.c * (Real.cos (t.A / 2))^2 = (3/2) * t.b)
  (h2 : t.B = π/3) (h3 : (1/2) * t.a * t.c * Real.sin t.B = 8 * Real.sqrt 3) :
  (2 * t.b = t.a + t.c) ∧ (t.b = 4 * Real.sqrt 2) := by
  sorry

end triangle_theorem_l1222_122278


namespace sqrt_equation_solution_l1222_122287

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end sqrt_equation_solution_l1222_122287


namespace quadratic_has_two_distinct_roots_l1222_122244

theorem quadratic_has_two_distinct_roots (a : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x^2 - (2*a - 1)*x + a^2 - a
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) := by
  sorry

end quadratic_has_two_distinct_roots_l1222_122244
