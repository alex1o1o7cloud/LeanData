import Mathlib

namespace NUMINAMATH_CALUDE_largest_pot_cost_l1304_130482

/-- The cost of the largest pot in a set of 6 pots with specific pricing rules -/
theorem largest_pot_cost (total_cost : ℚ) (num_pots : ℕ) (price_diff : ℚ) :
  total_cost = 33/4 ∧ num_pots = 6 ∧ price_diff = 1/10 →
  ∃ (smallest_cost : ℚ),
    smallest_cost > 0 ∧
    (smallest_cost + (num_pots - 1) * price_diff) = 13/8 := by
  sorry

end NUMINAMATH_CALUDE_largest_pot_cost_l1304_130482


namespace NUMINAMATH_CALUDE_segment_length_l1304_130415

-- Define the line segment AB and points P and Q
structure Segment where
  length : ℝ

structure Point where
  position : ℝ

-- Define the ratios for P and Q
def ratio_P : ℚ := 3 / 7
def ratio_Q : ℚ := 4 / 9

-- State the theorem
theorem segment_length 
  (AB : Segment) 
  (P Q : Point) 
  (h1 : P.position ≤ Q.position) -- P and Q are on the same side of the midpoint
  (h2 : P.position = ratio_P * AB.length) -- P divides AB in ratio 3:4
  (h3 : Q.position = ratio_Q * AB.length) -- Q divides AB in ratio 4:5
  (h4 : Q.position - P.position = 3) -- PQ = 3
  : AB.length = 189 := by
  sorry


end NUMINAMATH_CALUDE_segment_length_l1304_130415


namespace NUMINAMATH_CALUDE_distinct_two_mark_grids_l1304_130481

/-- Represents a 4x4 grid --/
def Grid := Fin 4 → Fin 4 → Bool

/-- Represents a rotation of the grid --/
inductive Rotation
| r0 | r90 | r180 | r270

/-- Applies a rotation to a grid --/
def applyRotation (r : Rotation) (g : Grid) : Grid :=
  sorry

/-- Checks if two grids are equivalent under rotation --/
def areEquivalent (g1 g2 : Grid) : Bool :=
  sorry

/-- Counts the number of marked cells in a grid --/
def countMarked (g : Grid) : Nat :=
  sorry

/-- Generates all possible grids with exactly two marked cells --/
def allGridsWithTwoMarked : List Grid :=
  sorry

/-- Counts the number of distinct grids under rotation --/
def countDistinctGrids (grids : List Grid) : Nat :=
  sorry

/-- The main theorem to be proved --/
theorem distinct_two_mark_grids :
  countDistinctGrids allGridsWithTwoMarked = 32 :=
sorry

end NUMINAMATH_CALUDE_distinct_two_mark_grids_l1304_130481


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l1304_130469

-- Define the line equation
def line_equation (x y m : ℝ) : Prop := x - y + m = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y = 0

-- Define the center of the circle
def circle_center (x y : ℝ) : Prop := circle_equation x y ∧ ∀ x' y', circle_equation x' y' → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2

-- Theorem statement
theorem line_passes_through_circle_center :
  ∃ x y : ℝ, circle_center x y ∧ line_equation x y (-3) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_circle_center_l1304_130469


namespace NUMINAMATH_CALUDE_a_perpendicular_to_a_minus_b_l1304_130426

def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-1, 3)

theorem a_perpendicular_to_a_minus_b : a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_perpendicular_to_a_minus_b_l1304_130426


namespace NUMINAMATH_CALUDE_system_solution_proof_l1304_130450

theorem system_solution_proof (x y z : ℝ) : 
  (2 * x + y = 3) ∧ 
  (3 * x - z = 7) ∧ 
  (x - y + 3 * z = 0) → 
  (x = 2 ∧ y = -1 ∧ z = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_proof_l1304_130450


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l1304_130485

/-- 
Given a rectangular box with dimensions a, b, and c, 
if the sum of the lengths of its twelve edges is 172 
and the distance from one corner to the farthest corner is 21, 
then its total surface area is 1408.
-/
theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 172) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 21) : 
  2 * (a * b + b * c + c * a) = 1408 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l1304_130485


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1304_130464

open Complex

theorem complex_equation_solution (a : ℝ) : 
  (1 - I)^3 / (1 + I) = a + 3*I → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1304_130464


namespace NUMINAMATH_CALUDE_cube_sphere_volume_l1304_130422

theorem cube_sphere_volume (n : ℕ) (hn : n > 2) : 
  (n^3 : ℝ) - (4/3 * Real.pi * (n/2)^3) = 2 * (4/3 * Real.pi * (n/2)^3) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_sphere_volume_l1304_130422


namespace NUMINAMATH_CALUDE_flame_time_calculation_l1304_130456

/-- Represents the duration of one minute in seconds -/
def minute_duration : ℕ := 60

/-- Represents the interval between weapon fires in seconds -/
def fire_interval : ℕ := 15

/-- Represents the duration of each flame shot in seconds -/
def flame_duration : ℕ := 5

/-- Calculates the total time spent shooting flames in one minute -/
def flame_time_per_minute : ℕ := (minute_duration / fire_interval) * flame_duration

theorem flame_time_calculation :
  flame_time_per_minute = 20 := by sorry

end NUMINAMATH_CALUDE_flame_time_calculation_l1304_130456


namespace NUMINAMATH_CALUDE_tangent_line_exists_tangent_line_equation_l1304_130401

/-- The function f(x) = x³ - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 3

/-- Theorem: There exists a tangent line to y = f(x) passing through (2, -6) -/
theorem tangent_line_exists : 
  ∃ (x₀ : ℝ), 
    (f x₀ + f_deriv x₀ * (2 - x₀) = -6) ∧ 
    ((f_deriv x₀ = -3) ∨ (f_deriv x₀ = 24)) :=
sorry

/-- Theorem: The tangent line equation is y = -3x or y = 24x - 54 -/
theorem tangent_line_equation (x₀ : ℝ) 
  (h : (f x₀ + f_deriv x₀ * (2 - x₀) = -6) ∧ 
       ((f_deriv x₀ = -3) ∨ (f_deriv x₀ = 24))) : 
  (∀ x y, y = -3*x) ∨ (∀ x y, y = 24*x - 54) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_exists_tangent_line_equation_l1304_130401


namespace NUMINAMATH_CALUDE_problem_statement_l1304_130471

theorem problem_statement (n : ℝ) (h : n + 1/n = 6) : n^2 + 1/n^2 + 9 = 43 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1304_130471


namespace NUMINAMATH_CALUDE_simplify_fraction_l1304_130447

theorem simplify_fraction :
  (140 : ℚ) / 2100 = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1304_130447


namespace NUMINAMATH_CALUDE_johns_donation_is_260_average_increase_70_percent_new_average_is_85_five_initial_contributions_l1304_130453

/-- The size of John's donation to the charity fund -/
def johns_donation (initial_average : ℝ) (num_initial_contributions : ℕ) : ℝ :=
  let new_average := 85
  let num_total_contributions := num_initial_contributions + 1
  let total_initial_amount := initial_average * num_initial_contributions
  new_average * num_total_contributions - total_initial_amount

/-- Proof that John's donation is $260 given the conditions -/
theorem johns_donation_is_260 :
  let initial_average := 50
  let num_initial_contributions := 5
  johns_donation initial_average num_initial_contributions = 260 :=
by sorry

/-- The average contribution size increases by 70% after John's donation -/
theorem average_increase_70_percent (initial_average : ℝ) (num_initial_contributions : ℕ) :
  let new_average := 85
  new_average = initial_average * 1.7 :=
by sorry

/-- The new average contribution size is $85 per person -/
theorem new_average_is_85 (initial_average : ℝ) (num_initial_contributions : ℕ) :
  let new_average := 85
  let num_total_contributions := num_initial_contributions + 1
  let total_amount := initial_average * num_initial_contributions + johns_donation initial_average num_initial_contributions
  total_amount / num_total_contributions = new_average :=
by sorry

/-- There were 5 other contributions made before John's -/
theorem five_initial_contributions :
  let num_initial_contributions := 5
  num_initial_contributions = 5 :=
by sorry

end NUMINAMATH_CALUDE_johns_donation_is_260_average_increase_70_percent_new_average_is_85_five_initial_contributions_l1304_130453


namespace NUMINAMATH_CALUDE_binomial_coefficient_18_8_l1304_130497

theorem binomial_coefficient_18_8 (h1 : Nat.choose 16 6 = 8008)
                                  (h2 : Nat.choose 16 7 = 11440)
                                  (h3 : Nat.choose 16 8 = 12870) :
  Nat.choose 18 8 = 43758 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_18_8_l1304_130497


namespace NUMINAMATH_CALUDE_necessary_condition_l1304_130449

theorem necessary_condition (p q : Prop) 
  (h : p → q) : ¬q → ¬p := by sorry

end NUMINAMATH_CALUDE_necessary_condition_l1304_130449


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1005th_term_l1304_130488

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (p r : ℚ) : ℕ → ℚ
  | 0 => p
  | 1 => 10
  | 2 => 4 * p - r
  | 3 => 4 * p + r
  | n + 4 => ArithmeticSequence p r 3 + (n + 1) * (ArithmeticSequence p r 3 - ArithmeticSequence p r 2)

/-- The 1005th term of the arithmetic sequence is 5480 -/
theorem arithmetic_sequence_1005th_term (p r : ℚ) :
  ArithmeticSequence p r 1004 = 5480 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_1005th_term_l1304_130488


namespace NUMINAMATH_CALUDE_find_number_l1304_130466

theorem find_number : ∃ N : ℚ, (5/6 : ℚ) * N = (5/16 : ℚ) * N + 50 → N = 96 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1304_130466


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1304_130465

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 1, p x) ↔ (∀ x > 1, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x > 1, x^2 - 1 > 0) ↔ (∀ x > 1, x^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1304_130465


namespace NUMINAMATH_CALUDE_deepak_age_l1304_130403

/-- Given the ratio of Arun's age to Deepak's age and Arun's future age, prove Deepak's current age -/
theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 2 / 5 →
  arun_age + 10 = 30 →
  deepak_age = 50 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1304_130403


namespace NUMINAMATH_CALUDE_sum_two_angles_greater_90_implies_acute_l1304_130425

-- Define a triangle type
structure Triangle where
  α : Real
  β : Real
  γ : Real
  angle_sum : α + β + γ = 180
  positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ

-- Define the property of sum of any two angles being greater than 90°
def sum_of_two_angles_greater_than_90 (t : Triangle) : Prop :=
  t.α + t.β > 90 ∧ t.α + t.γ > 90 ∧ t.β + t.γ > 90

-- Define an acute triangle
def is_acute_triangle (t : Triangle) : Prop :=
  t.α < 90 ∧ t.β < 90 ∧ t.γ < 90

-- Theorem statement
theorem sum_two_angles_greater_90_implies_acute (t : Triangle) :
  sum_of_two_angles_greater_than_90 t → is_acute_triangle t :=
by
  sorry


end NUMINAMATH_CALUDE_sum_two_angles_greater_90_implies_acute_l1304_130425


namespace NUMINAMATH_CALUDE_sphere_volume_in_cube_l1304_130427

/-- Given a cube with edge length a and two congruent spheres inscribed in opposite trihedral angles
    that touch each other, this theorem states the volume of each sphere. -/
theorem sphere_volume_in_cube (a : ℝ) (a_pos : 0 < a) : 
  ∃ (r : ℝ), r = (3 * a - a * Real.sqrt 3) / 4 ∧ 
              (4 / 3 : ℝ) * Real.pi * r^3 = (4 / 3 : ℝ) * Real.pi * ((3 * a - a * Real.sqrt 3) / 4)^3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_in_cube_l1304_130427


namespace NUMINAMATH_CALUDE_min_perimeter_nine_square_rectangle_l1304_130423

/-- Represents a rectangle divided into nine squares with integer side lengths -/
structure NineSquareRectangle where
  squares : Fin 9 → ℕ
  width : ℕ
  height : ℕ
  is_valid : width = squares 0 + squares 1 + squares 2 ∧
             height = squares 0 + squares 3 + squares 6 ∧
             width = squares 6 + squares 7 + squares 8 ∧
             height = squares 2 + squares 5 + squares 8

/-- The perimeter of a rectangle -/
def perimeter (rect : NineSquareRectangle) : ℕ :=
  2 * (rect.width + rect.height)

/-- Theorem stating that the minimum perimeter of a valid NineSquareRectangle is 52 -/
theorem min_perimeter_nine_square_rectangle :
  ∃ (rect : NineSquareRectangle), perimeter rect = 52 ∧
  ∀ (other : NineSquareRectangle), perimeter other ≥ 52 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_nine_square_rectangle_l1304_130423


namespace NUMINAMATH_CALUDE_lunch_gratuity_percentage_l1304_130435

/-- Given the conditions of a lunch bill, prove the gratuity percentage --/
theorem lunch_gratuity_percentage
  (total_price : ℝ)
  (num_people : ℕ)
  (avg_price_no_gratuity : ℝ)
  (h1 : total_price = 207)
  (h2 : num_people = 15)
  (h3 : avg_price_no_gratuity = 12) :
  (total_price - (↑num_people * avg_price_no_gratuity)) / (↑num_people * avg_price_no_gratuity) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_lunch_gratuity_percentage_l1304_130435


namespace NUMINAMATH_CALUDE_value_of_x_l1304_130477

theorem value_of_x : (2023^2 - 2023 + 1) / 2023 = 2022 + 1/2023 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1304_130477


namespace NUMINAMATH_CALUDE_angle_C_60_not_sufficient_for_similarity_l1304_130424

-- Define triangles ABC and A'B'C'
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angles of a triangle
def angle (t : Triangle) (v : Fin 3) : ℝ :=
  sorry

-- State the given conditions
axiom triangle_ABC : Triangle
axiom triangle_A'B'C' : Triangle

axiom angle_B_is_right : angle triangle_ABC 1 = 90
axiom angle_B'_is_right : angle triangle_A'B'C' 1 = 90
axiom angle_A_is_30 : angle triangle_ABC 0 = 30

-- Define triangle similarity
def similar (t1 t2 : Triangle) : Prop :=
  sorry

-- State the theorem
theorem angle_C_60_not_sufficient_for_similarity :
  ¬(∀ (ABC A'B'C' : Triangle),
    angle ABC 1 = 90 →
    angle A'B'C' 1 = 90 →
    angle ABC 0 = 30 →
    angle ABC 2 = 60 →
    similar ABC A'B'C') :=
  sorry

end NUMINAMATH_CALUDE_angle_C_60_not_sufficient_for_similarity_l1304_130424


namespace NUMINAMATH_CALUDE_symmetry_properties_l1304_130472

def Point := ℝ × ℝ

def symmetricAboutXAxis (p : Point) : Point :=
  (p.1, -p.2)

def symmetricAboutYAxis (p : Point) : Point :=
  (-p.1, p.2)

theorem symmetry_properties (x y : ℝ) :
  let A : Point := (x, y)
  (symmetricAboutXAxis A = (x, -y)) ∧
  (symmetricAboutYAxis A = (-x, y)) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_properties_l1304_130472


namespace NUMINAMATH_CALUDE_even_function_c_value_f_increasing_on_interval_l1304_130444

def f (x : ℝ) : ℝ := x^2 + 4*x + 3

def g (x c : ℝ) : ℝ := f x + c*x

theorem even_function_c_value :
  (∀ x, g x (-4) = g (-x) (-4)) ∧ 
  (∀ c, (∀ x, g x c = g (-x) c) → c = -4) :=
sorry

theorem f_increasing_on_interval :
  ∀ x₁ x₂, -2 ≤ x₁ → x₁ < x₂ → f x₁ < f x₂ :=
sorry

end NUMINAMATH_CALUDE_even_function_c_value_f_increasing_on_interval_l1304_130444


namespace NUMINAMATH_CALUDE_circles_coaxial_system_l1304_130420

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  a : Point2D
  b : Point2D
  c : Point2D

/-- Checks if three circles form a coaxial system -/
def areCoaxial (c1 c2 c3 : Circle) : Prop :=
  sorry

/-- Constructs a circle with diameter as the given line segment -/
def circleDiameterSegment (p1 p2 : Point2D) : Circle :=
  sorry

/-- Finds the intersection point of a line and a triangle side -/
def lineTriangleIntersection (l : Line) (t : Triangle) : Point2D :=
  sorry

/-- Main theorem: Given a triangle intersected by a line, 
    the circles constructed on the resulting segments form a coaxial system -/
theorem circles_coaxial_system 
  (t : Triangle) 
  (l : Line) : 
  let a1 := lineTriangleIntersection l t
  let b1 := lineTriangleIntersection l t
  let c1 := lineTriangleIntersection l t
  let circleA := circleDiameterSegment t.a a1
  let circleB := circleDiameterSegment t.b b1
  let circleC := circleDiameterSegment t.c c1
  areCoaxial circleA circleB circleC :=
by
  sorry

end NUMINAMATH_CALUDE_circles_coaxial_system_l1304_130420


namespace NUMINAMATH_CALUDE_farm_animals_l1304_130494

theorem farm_animals (pigs : ℕ) (cows : ℕ) (goats : ℕ) : 
  cows = 2 * pigs - 3 →
  goats = cows + 6 →
  pigs + cows + goats = 50 →
  pigs = 10 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l1304_130494


namespace NUMINAMATH_CALUDE_min_correct_answers_l1304_130498

/-- Represents the scoring system and conditions of the IQ test -/
structure IQTest where
  total_questions : ℕ
  correct_points : ℕ
  wrong_points : ℕ
  unanswered : ℕ
  min_score : ℕ

/-- Calculates the score based on the number of correct answers -/
def calculate_score (test : IQTest) (correct_answers : ℕ) : ℤ :=
  (correct_answers : ℤ) * test.correct_points - 
  ((test.total_questions - test.unanswered - correct_answers) : ℤ) * test.wrong_points

/-- Theorem stating the minimum number of correct answers needed to achieve the minimum score -/
theorem min_correct_answers (test : IQTest) : 
  test.total_questions = 20 ∧ 
  test.correct_points = 5 ∧ 
  test.wrong_points = 2 ∧ 
  test.unanswered = 2 ∧ 
  test.min_score = 60 →
  (∀ x : ℕ, x < 14 → calculate_score test x < test.min_score) ∧
  calculate_score test 14 ≥ test.min_score := by
  sorry


end NUMINAMATH_CALUDE_min_correct_answers_l1304_130498


namespace NUMINAMATH_CALUDE_dining_bill_share_l1304_130478

theorem dining_bill_share (total_bill : ℝ) (num_people : ℕ) (tip_percentage : ℝ) :
  total_bill = 139 ∧ num_people = 7 ∧ tip_percentage = 0.1 →
  (total_bill * (1 + tip_percentage)) / num_people = 21.84 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_share_l1304_130478


namespace NUMINAMATH_CALUDE_gcd_digit_bound_l1304_130451

theorem gcd_digit_bound (a b : ℕ) : 
  (10^6 ≤ a ∧ a < 10^7) →
  (10^6 ≤ b ∧ b < 10^7) →
  (10^12 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^13) →
  Nat.gcd a b < 10^2 :=
sorry

end NUMINAMATH_CALUDE_gcd_digit_bound_l1304_130451


namespace NUMINAMATH_CALUDE_kate_emily_hair_ratio_l1304_130463

/-- The ratio of hair lengths -/
def hair_length_ratio (kate_length emily_length : ℕ) : ℚ :=
  kate_length / emily_length

/-- Theorem stating the ratio of Kate's hair length to Emily's hair length -/
theorem kate_emily_hair_ratio :
  let logan_length : ℕ := 20
  let emily_length : ℕ := logan_length + 6
  let kate_length : ℕ := 7
  hair_length_ratio kate_length emily_length = 7 / 26 := by
sorry

end NUMINAMATH_CALUDE_kate_emily_hair_ratio_l1304_130463


namespace NUMINAMATH_CALUDE_marble_box_problem_l1304_130408

theorem marble_box_problem :
  ∀ (red blue : ℕ),
  red = blue →
  20 + red + blue - 2 * (20 - blue) = 40 →
  20 + red + blue = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_box_problem_l1304_130408


namespace NUMINAMATH_CALUDE_square_of_negative_half_a_squared_b_l1304_130446

theorem square_of_negative_half_a_squared_b (a b : ℝ) :
  (- (1/2 : ℝ) * a^2 * b)^2 = (1/4 : ℝ) * a^4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_half_a_squared_b_l1304_130446


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_circles_l1304_130462

/-- The area of the shaded region formed by the intersection of a rectangle and two circles -/
theorem shaded_area_rectangle_circles (rectangle_width : ℝ) (rectangle_height : ℝ) (circle_radius : ℝ) : 
  rectangle_width = 12 →
  rectangle_height = 10 →
  circle_radius = 3 →
  let rectangle_area := rectangle_width * rectangle_height
  let circle_area := π * circle_radius^2
  let shaded_area := rectangle_area - 2 * circle_area
  shaded_area = 120 - 18 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_circles_l1304_130462


namespace NUMINAMATH_CALUDE_number_of_provinces_l1304_130413

theorem number_of_provinces (P T : ℕ) (n : ℕ) : 
  T = (3 * P) / 4 →  -- The fraction of traditionalists is 0.75
  (∃ k : ℕ, T = k * (P / 12)) →  -- Each province has P/12 traditionalists
  n = T / (P / 12) →  -- Definition of n
  n = 9 :=
by sorry

end NUMINAMATH_CALUDE_number_of_provinces_l1304_130413


namespace NUMINAMATH_CALUDE_fiftieth_islander_is_knight_l1304_130436

/-- Represents the type of an islander: either a knight or a liar -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents the statement made by an islander about their right neighbor -/
inductive Statement
  | Knight
  | Liar

/-- The number of islanders around the table -/
def n : ℕ := 50

/-- Function that returns the statement made by the islander at a given position -/
def statement (pos : ℕ) : Statement :=
  if pos % 2 = 1 then Statement.Knight else Statement.Liar

/-- Function that determines the actual type of the islander at a given position -/
def islanderType (firstType : IslanderType) (pos : ℕ) : IslanderType :=
  sorry

/-- Theorem stating that the 50th islander must be a knight -/
theorem fiftieth_islander_is_knight (firstType : IslanderType) :
  islanderType firstType n = IslanderType.Knight :=
  sorry

end NUMINAMATH_CALUDE_fiftieth_islander_is_knight_l1304_130436


namespace NUMINAMATH_CALUDE_sum_f_half_integers_l1304_130421

def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem sum_f_half_integers (f : ℝ → ℝ) 
  (h1 : is_even (λ x ↦ f (2*x + 2)))
  (h2 : is_odd (λ x ↦ f (x + 1)))
  (h3 : ∃ a b : ℝ, ∀ x ∈ Set.Icc 0 1, f x = a * x + b)
  (h4 : f 4 = 1) :
  (f (3/2) + f (5/2) + f (7/2)) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sum_f_half_integers_l1304_130421


namespace NUMINAMATH_CALUDE_michelle_final_crayons_l1304_130406

/-- 
Given:
- Michelle initially has x crayons
- Janet initially has y crayons
- Both Michelle and Janet receive z more crayons each
- Janet gives all of her crayons to Michelle

Prove that Michelle will have x + y + 2z crayons in total.
-/
theorem michelle_final_crayons (x y z : ℕ) : x + z + (y + z) = x + y + 2*z :=
by sorry

end NUMINAMATH_CALUDE_michelle_final_crayons_l1304_130406


namespace NUMINAMATH_CALUDE_age_problem_l1304_130480

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 27 →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1304_130480


namespace NUMINAMATH_CALUDE_lukas_average_points_l1304_130437

/-- Given a basketball player's total points and number of games, 
    calculate their average points per game. -/
def average_points_per_game (total_points : ℕ) (num_games : ℕ) : ℚ :=
  (total_points : ℚ) / (num_games : ℚ)

/-- Theorem: A player who scores 60 points in 5 games averages 12 points per game. -/
theorem lukas_average_points : 
  average_points_per_game 60 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_lukas_average_points_l1304_130437


namespace NUMINAMATH_CALUDE_positive_abc_l1304_130432

theorem positive_abc (a b c : ℝ) 
  (sum_pos : a + b + c > 0) 
  (sum_prod_pos : a * b + b * c + c * a > 0) 
  (prod_pos : a * b * c > 0) : 
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_abc_l1304_130432


namespace NUMINAMATH_CALUDE_no_ab_term_in_polynomial_l1304_130448

theorem no_ab_term_in_polynomial (m : ℝ) : 
  (∀ a b : ℝ, (a^2 + 2*a*b - b^2) - (a^2 + m*a*b + 2*b^2) = (-3:ℝ)*b^2) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_ab_term_in_polynomial_l1304_130448


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l1304_130405

/-- Represents an ellipse in a 2D Cartesian coordinate system -/
structure Ellipse where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  passing_point : ℝ × ℝ

/-- The standard equation of an ellipse -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem: Given an ellipse with specific properties, prove its standard equation -/
theorem ellipse_standard_equation (e : Ellipse) 
  (h_center : e.center = (0, 0))
  (h_left_focus : e.left_focus = (-Real.sqrt 3, 0))
  (h_passing_point : e.passing_point = (2, 0)) :
  ∀ x y : ℝ, standard_equation 4 1 x y :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l1304_130405


namespace NUMINAMATH_CALUDE_draw_balls_theorem_l1304_130491

/-- The number of ways to draw balls from a bag under specific conditions -/
def draw_balls_count (total_white : ℕ) (total_red : ℕ) (total_black : ℕ) 
                     (draw_count : ℕ) (min_white : ℕ) (max_white : ℕ) 
                     (min_red : ℕ) (max_red : ℕ) (max_black : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to draw balls under given conditions -/
theorem draw_balls_theorem : 
  draw_balls_count 9 5 6 10 3 7 2 5 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_draw_balls_theorem_l1304_130491


namespace NUMINAMATH_CALUDE_composite_sum_of_powers_l1304_130467

theorem composite_sum_of_powers (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ (a^1984 + b^1984 + c^1984 + d^1984 = x * y) := by
  sorry

end NUMINAMATH_CALUDE_composite_sum_of_powers_l1304_130467


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1304_130419

theorem smallest_solution_of_equation :
  ∃ (x : ℝ), x = 39/8 ∧
  x ≠ 0 ∧ x ≠ 3 ∧
  (3*x)/(x-3) + (3*x^2-27)/x = 14 ∧
  ∀ (y : ℝ), y ≠ 0 → y ≠ 3 → (3*y)/(y-3) + (3*y^2-27)/y = 14 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1304_130419


namespace NUMINAMATH_CALUDE_circle_equation_m_range_l1304_130474

/-- 
Given an equation x^2 + y^2 + mx - 2y + 4 = 0 that represents a circle,
prove that m must be in the range (-∞, -2√3) ∪ (2√3, +∞).
-/
theorem circle_equation_m_range :
  ∀ m : ℝ, 
  (∃ x y : ℝ, x^2 + y^2 + m*x - 2*y + 4 = 0) →
  (m < -2 * Real.sqrt 3 ∨ m > 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_m_range_l1304_130474


namespace NUMINAMATH_CALUDE_platform_length_l1304_130434

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 20 seconds to cross a signal pole, the length of the platform is 285 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 39)
  (h3 : time_pole = 20) :
  let speed := train_length / time_pole
  let platform_length := speed * time_platform - train_length
  platform_length = 285 := by sorry

end NUMINAMATH_CALUDE_platform_length_l1304_130434


namespace NUMINAMATH_CALUDE_division_of_mixed_number_by_fraction_l1304_130414

theorem division_of_mixed_number_by_fraction :
  (2 + 1 / 4 : ℚ) / (2 / 3 : ℚ) = 27 / 8 := by sorry

end NUMINAMATH_CALUDE_division_of_mixed_number_by_fraction_l1304_130414


namespace NUMINAMATH_CALUDE_potato_bag_weights_l1304_130460

/-- Represents the weights of three bags of potatoes -/
structure BagWeights where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Calculates the new weights after adjustments -/
def adjustedWeights (w : BagWeights) : BagWeights :=
  { A := w.A - 0.1 * w.C
  , B := w.B + 0.15 * w.A
  , C := w.C }

/-- Theorem stating the result of the potato bag weight problem -/
theorem potato_bag_weights :
  ∀ w : BagWeights,
    w.A = 12 + 1/2 * w.B →
    w.B = 8 + 1/3 * w.C →
    w.C = 20 + 2 * w.A →
    let new_w := adjustedWeights w
    (new_w.A + new_w.B + new_w.C) = 139.55 := by
  sorry


end NUMINAMATH_CALUDE_potato_bag_weights_l1304_130460


namespace NUMINAMATH_CALUDE_negation_of_implication_l1304_130470

theorem negation_of_implication (A B : Set α) :
  ¬(A ∪ B = A → A ∩ B = B) ↔ (A ∪ B = A ∧ A ∩ B ≠ B) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1304_130470


namespace NUMINAMATH_CALUDE_kevins_age_l1304_130441

theorem kevins_age (vanessa_age : ℕ) (future_years : ℕ) (ratio : ℕ) :
  vanessa_age = 2 →
  future_years = 5 →
  ratio = 3 →
  ∃ kevin_age : ℕ, kevin_age + future_years = ratio * (vanessa_age + future_years) ∧ kevin_age = 16 :=
by sorry

end NUMINAMATH_CALUDE_kevins_age_l1304_130441


namespace NUMINAMATH_CALUDE_min_apples_count_l1304_130404

theorem min_apples_count : ∃ n : ℕ, n > 0 ∧
  n % 4 = 1 ∧
  n % 5 = 2 ∧
  n % 9 = 7 ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 9 = 7 → n ≤ m) ∧
  n = 97 := by
sorry

end NUMINAMATH_CALUDE_min_apples_count_l1304_130404


namespace NUMINAMATH_CALUDE_solution_equality_l1304_130486

theorem solution_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.sqrt 2 * x + 1 / (Real.sqrt 2 * x) +
   Real.sqrt 2 * y + 1 / (Real.sqrt 2 * y) +
   Real.sqrt 2 * z + 1 / (Real.sqrt 2 * z) =
   6 - 2 * Real.sqrt (2 * x) * |y - z| -
   Real.sqrt (2 * y) * (x - z)^2 -
   Real.sqrt (2 * z) * Real.sqrt |x - y|) ↔
  (x = 1 / Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 ∧ z = 1 / Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_equality_l1304_130486


namespace NUMINAMATH_CALUDE_pigs_joined_l1304_130496

theorem pigs_joined (initial_pigs final_pigs : ℕ) 
  (h1 : initial_pigs = 64)
  (h2 : final_pigs = 86) :
  final_pigs - initial_pigs = 22 :=
by sorry

end NUMINAMATH_CALUDE_pigs_joined_l1304_130496


namespace NUMINAMATH_CALUDE_x_plus_y_equals_five_l1304_130495

theorem x_plus_y_equals_five (x y : ℝ) (h : (x + 1)^2 + |y - 6| = 0) : x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_five_l1304_130495


namespace NUMINAMATH_CALUDE_hurricane_damage_in_cad_l1304_130407

/-- Converts American dollars to Canadian dollars given a conversion rate -/
def convert_usd_to_cad (usd : ℝ) (rate : ℝ) : ℝ := usd * rate

/-- The damage caused by the hurricane in American dollars -/
def damage_usd : ℝ := 60000000

/-- The conversion rate from American dollars to Canadian dollars -/
def usd_to_cad_rate : ℝ := 1.25

/-- Theorem stating the equivalent damage in Canadian dollars -/
theorem hurricane_damage_in_cad :
  convert_usd_to_cad damage_usd usd_to_cad_rate = 75000000 := by
  sorry

end NUMINAMATH_CALUDE_hurricane_damage_in_cad_l1304_130407


namespace NUMINAMATH_CALUDE_f_negative_a_eq_zero_l1304_130402

/-- Given a real-valued function f(x) = x³ + x + 1 and a real number a such that f(a) = 2,
    prove that f(-a) = 0. -/
theorem f_negative_a_eq_zero (a : ℝ) (h : a^3 + a + 1 = 2) :
  (-a)^3 + (-a) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_eq_zero_l1304_130402


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1304_130459

theorem quadratic_two_distinct_roots (c : ℝ) (h : c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + c = 0 ∧ x₂^2 + 2*x₂ + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1304_130459


namespace NUMINAMATH_CALUDE_triangle_angle_c_l1304_130417

theorem triangle_angle_c (A B C : ℝ) (h1 : A + B = 110) : C = 70 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l1304_130417


namespace NUMINAMATH_CALUDE_units_digit_17_pow_2023_l1304_130438

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of raising a number to a power, considering only the units digit -/
def powerMod10 (base : ℕ) (exp : ℕ) : ℕ :=
  (base ^ exp) % 10

theorem units_digit_17_pow_2023 :
  unitsDigit (powerMod10 17 2023) = 3 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_2023_l1304_130438


namespace NUMINAMATH_CALUDE_exponential_function_values_l1304_130411

-- Define the exponential function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_function_values 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : f a 3 = 8) : 
  f a 4 = 16 ∧ f a (-4) = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_values_l1304_130411


namespace NUMINAMATH_CALUDE_point_not_on_line_l1304_130440

/-- Given real numbers m and b where mb < 0, the point (1, 2001) does not lie on the line y = m(x^2) + b -/
theorem point_not_on_line (m b : ℝ) (h : m * b < 0) : 
  ¬(2001 = m * (1 : ℝ)^2 + b) := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_line_l1304_130440


namespace NUMINAMATH_CALUDE_staircase_extension_l1304_130493

def toothpicks_for_step (n : ℕ) : ℕ := 12 + 2 * (n - 5)

theorem staircase_extension : 
  (toothpicks_for_step 5) + (toothpicks_for_step 6) = 26 :=
by sorry

end NUMINAMATH_CALUDE_staircase_extension_l1304_130493


namespace NUMINAMATH_CALUDE_contrapositive_odd_product_l1304_130473

theorem contrapositive_odd_product (a b : ℤ) : 
  (((a % 2 = 1 ∧ b % 2 = 1) → (a * b) % 2 = 1) ↔ 
   ((a * b) % 2 ≠ 1 → (a % 2 ≠ 1 ∨ b % 2 ≠ 1))) ∧
  (∀ a b : ℤ, (a * b) % 2 ≠ 1 → (a % 2 ≠ 1 ∨ b % 2 ≠ 1)) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_odd_product_l1304_130473


namespace NUMINAMATH_CALUDE_elderly_arrangements_proof_l1304_130452

def number_of_arrangements (n_volunteers : ℕ) (n_elderly : ℕ) : ℕ :=
  (n_volunteers.factorial) * 
  (n_volunteers - 1) * 
  (n_elderly.factorial)

theorem elderly_arrangements_proof :
  number_of_arrangements 4 2 = 144 :=
by sorry

end NUMINAMATH_CALUDE_elderly_arrangements_proof_l1304_130452


namespace NUMINAMATH_CALUDE_trigonometric_properties_l1304_130429

open Real

-- Define the concept of terminal side of an angle
def sameSide (α β : ℝ) : Prop := sorry

-- Define the set of angles with terminal side on x-axis
def xAxisAngles : Set ℝ := { α | ∃ k : ℤ, α = k * π }

-- Define the quadrants
def inFirstOrSecondQuadrant (α : ℝ) : Prop := 0 < α ∧ α < π

theorem trigonometric_properties :
  (∀ α β : ℝ, sameSide α β → sin α = sin β ∧ cos α = cos β) ∧
  (xAxisAngles ≠ { α | ∃ k : ℤ, α = 2 * k * π }) ∧
  (∃ α : ℝ, sin α > 0 ∧ ¬inFirstOrSecondQuadrant α) ∧
  (∃ α β : ℝ, sin α = sin β ∧ ¬(∃ k : ℤ, α = 2 * k * π + β)) := by sorry

end NUMINAMATH_CALUDE_trigonometric_properties_l1304_130429


namespace NUMINAMATH_CALUDE_phones_to_repair_per_person_l1304_130410

theorem phones_to_repair_per_person
  (initial_phones : ℕ)
  (repaired_phones : ℕ)
  (new_phones : ℕ)
  (h1 : initial_phones = 15)
  (h2 : repaired_phones = 3)
  (h3 : new_phones = 6)
  (h4 : repaired_phones ≤ initial_phones) :
  (initial_phones - repaired_phones + new_phones) / 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_phones_to_repair_per_person_l1304_130410


namespace NUMINAMATH_CALUDE_train_length_calculation_l1304_130475

/-- Calculates the length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length_calculation (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 → platform_length = 280 → crossing_time = 26 →
  (train_speed * 1000 / 3600) * crossing_time - platform_length = 240 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l1304_130475


namespace NUMINAMATH_CALUDE_sticker_collection_total_l1304_130442

/-- The number of stickers Karl has -/
def karl_stickers : ℕ := 25

/-- The number of stickers Ryan has -/
def ryan_stickers : ℕ := karl_stickers + 20

/-- The number of stickers Ben has -/
def ben_stickers : ℕ := ryan_stickers - 10

/-- The total number of stickers placed in the book -/
def total_stickers : ℕ := karl_stickers + ryan_stickers + ben_stickers

theorem sticker_collection_total :
  total_stickers = 105 := by sorry

end NUMINAMATH_CALUDE_sticker_collection_total_l1304_130442


namespace NUMINAMATH_CALUDE_base_2016_remainder_l1304_130412

theorem base_2016_remainder (N A B C k : ℕ) : 
  (N = A * 2016^2 + B * 2016 + C) →
  (A < 2016 ∧ B < 2016 ∧ C < 2016) →
  (1 ≤ k ∧ k ≤ 2015) →
  (N - (A + B + C + k)) % 2015 = 2015 - k := by
  sorry

end NUMINAMATH_CALUDE_base_2016_remainder_l1304_130412


namespace NUMINAMATH_CALUDE_integer_root_theorem_l1304_130483

def polynomial (x b : ℤ) : ℤ := x^3 + 4*x^2 + b*x + 12

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

def valid_b_values : Set ℤ := {-177, -62, -35, -25, -18, -17, 9, 16, 27, 48, 144, 1296}

theorem integer_root_theorem :
  ∀ b : ℤ, has_integer_root b ↔ b ∈ valid_b_values :=
by sorry

end NUMINAMATH_CALUDE_integer_root_theorem_l1304_130483


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l1304_130468

theorem tripled_base_and_exponent (a b : ℝ) (x : ℝ) (hx : x > 0) :
  (3*a)^(3*b) = a^b * x^b → x = 27 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l1304_130468


namespace NUMINAMATH_CALUDE_final_values_l1304_130409

def sequence_operations (a b c : ℕ) : ℕ × ℕ × ℕ :=
  let a' := b
  let b' := c
  let c' := a'
  (a', b', c')

theorem final_values :
  sequence_operations 10 20 30 = (20, 30, 20) := by sorry

end NUMINAMATH_CALUDE_final_values_l1304_130409


namespace NUMINAMATH_CALUDE_exp_greater_than_power_e_l1304_130457

theorem exp_greater_than_power_e (x : ℝ) (h1 : x > 0) (h2 : x ≠ ℯ) : ℯ^x > x^ℯ := by
  sorry

end NUMINAMATH_CALUDE_exp_greater_than_power_e_l1304_130457


namespace NUMINAMATH_CALUDE_interest_rate_difference_l1304_130499

/-- Proves that given a principal of $600 invested for 6 years, if the difference in interest earned between two rates is $144, then the difference between these two rates is 4%. -/
theorem interest_rate_difference (principal : ℝ) (time : ℝ) (interest_diff : ℝ) 
  (h1 : principal = 600)
  (h2 : time = 6)
  (h3 : interest_diff = 144) :
  ∃ (original_rate higher_rate : ℝ),
    (principal * time * higher_rate / 100 - principal * time * original_rate / 100 = interest_diff) ∧
    (higher_rate - original_rate = 4) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l1304_130499


namespace NUMINAMATH_CALUDE_axis_symmetry_implies_equal_coefficients_l1304_130400

theorem axis_symmetry_implies_equal_coefficients 
  (a b : ℝ) (h : a * b ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (2 * x) + b * Real.cos (2 * x)
  (∀ x, f (π/8 + x) = f (π/8 - x)) → a = b := by
  sorry

end NUMINAMATH_CALUDE_axis_symmetry_implies_equal_coefficients_l1304_130400


namespace NUMINAMATH_CALUDE_f_derivative_l1304_130479

noncomputable def f (x : ℝ) : ℝ :=
  (5^x * (2 * Real.sin (2*x) + Real.cos (2*x) * Real.log 5)) / (4 + (Real.log 5)^2)

theorem f_derivative (x : ℝ) :
  deriv f x = 5^x * Real.cos (2*x) :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_l1304_130479


namespace NUMINAMATH_CALUDE_total_books_l1304_130443

theorem total_books (tim_books sam_books : ℕ) 
  (h1 : tim_books = 44) 
  (h2 : sam_books = 52) : 
  tim_books + sam_books = 96 := by
sorry

end NUMINAMATH_CALUDE_total_books_l1304_130443


namespace NUMINAMATH_CALUDE_smith_laundry_loads_l1304_130430

/-- The number of bath towels Kylie uses in one month -/
def kylie_towels : ℕ := 3

/-- The number of bath towels Kylie's daughters use in one month -/
def daughters_towels : ℕ := 6

/-- The number of bath towels Kylie's husband uses in one month -/
def husband_towels : ℕ := 3

/-- The number of bath towels that fit in one load of laundry -/
def towels_per_load : ℕ := 4

/-- The total number of bath towels used by the Smith family in one month -/
def total_towels : ℕ := kylie_towels + daughters_towels + husband_towels

/-- The number of laundry loads required to clean all used towels -/
def required_loads : ℕ := (total_towels + towels_per_load - 1) / towels_per_load

theorem smith_laundry_loads : required_loads = 3 := by
  sorry

end NUMINAMATH_CALUDE_smith_laundry_loads_l1304_130430


namespace NUMINAMATH_CALUDE_courtyard_length_proof_l1304_130454

/-- Proves that the length of a rectangular courtyard is 15 m given specific conditions -/
theorem courtyard_length_proof (width : ℝ) (stone_length : ℝ) (stone_width : ℝ) (total_stones : ℕ) :
  width = 6 →
  stone_length = 3 →
  stone_width = 2 →
  total_stones = 15 →
  (width * (width * total_stones * stone_length * stone_width / width / stone_length / stone_width)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_proof_l1304_130454


namespace NUMINAMATH_CALUDE_tan_405_degrees_l1304_130487

theorem tan_405_degrees : Real.tan (405 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_405_degrees_l1304_130487


namespace NUMINAMATH_CALUDE_triangle_area_l1304_130431

theorem triangle_area (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  A = π/3 ∧
  c = 4 ∧
  b = 2 * Real.sqrt 3 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1304_130431


namespace NUMINAMATH_CALUDE_homework_probability_l1304_130458

theorem homework_probability (p : ℚ) (h : p = 5 / 9) :
  1 - p = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_homework_probability_l1304_130458


namespace NUMINAMATH_CALUDE_greene_nursery_white_roses_l1304_130416

/-- The number of white roses at Greene Nursery -/
def white_roses : ℕ := 6284 - (1491 + 3025)

/-- Theorem stating the number of white roses at Greene Nursery -/
theorem greene_nursery_white_roses :
  white_roses = 1768 :=
by sorry

end NUMINAMATH_CALUDE_greene_nursery_white_roses_l1304_130416


namespace NUMINAMATH_CALUDE_jessica_shells_count_l1304_130492

def seashell_problem (sally_shells tom_shells total_shells : ℕ) : Prop :=
  ∃ jessica_shells : ℕ, 
    sally_shells + tom_shells + jessica_shells = total_shells

theorem jessica_shells_count (sally_shells tom_shells total_shells : ℕ) 
  (h : seashell_problem sally_shells tom_shells total_shells) :
  ∃ jessica_shells : ℕ, jessica_shells = total_shells - (sally_shells + tom_shells) :=
by
  sorry

#check jessica_shells_count 9 7 21

end NUMINAMATH_CALUDE_jessica_shells_count_l1304_130492


namespace NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l1304_130490

theorem perpendicular_line_through_intersection :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, (2 * x + y = 8 ∧ x - 2 * y = -1) → a * x + b * y = c) ∧
    (a * 8 + b * (-6) = 0) ∧
    (a * 4 + b * 3 = 28.8) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l1304_130490


namespace NUMINAMATH_CALUDE_small_circle_radius_l1304_130428

/-- Given a large circle with radius 10 meters and four congruent smaller circles
    touching at its center, prove that the radius of each smaller circle is 5 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) : R = 10 → 2 * r = R → r = 5 := by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l1304_130428


namespace NUMINAMATH_CALUDE_slope_bisecting_line_l1304_130489

/-- The slope of the line passing through the center of a rectangle and the center of a circle cut out from it is 1/5. -/
theorem slope_bisecting_line (rectangle_center : ℝ × ℝ) (circle_center : ℝ × ℝ) : 
  rectangle_center = (50, 25) → 
  circle_center = (75, 30) → 
  (circle_center.2 - rectangle_center.2) / (circle_center.1 - rectangle_center.1) = 1/5 := by
sorry

end NUMINAMATH_CALUDE_slope_bisecting_line_l1304_130489


namespace NUMINAMATH_CALUDE_blue_cap_cost_l1304_130455

/-- The cost of items before applying a discount --/
structure PreDiscountCost where
  tshirt : ℕ
  backpack : ℕ
  bluecap : ℕ

/-- The total cost after applying a discount --/
def total_after_discount (cost : PreDiscountCost) (discount : ℕ) : ℕ :=
  cost.tshirt + cost.backpack + cost.bluecap - discount

/-- The theorem stating the cost of the blue cap --/
theorem blue_cap_cost (cost : PreDiscountCost) (discount : ℕ) :
  cost.tshirt = 30 →
  cost.backpack = 10 →
  discount = 2 →
  total_after_discount cost discount = 43 →
  cost.bluecap = 5 := by
  sorry

#check blue_cap_cost

end NUMINAMATH_CALUDE_blue_cap_cost_l1304_130455


namespace NUMINAMATH_CALUDE_discounted_price_theorem_l1304_130445

def original_price : ℝ := 760
def discount_percentage : ℝ := 75

theorem discounted_price_theorem :
  original_price * (1 - discount_percentage / 100) = 570 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_theorem_l1304_130445


namespace NUMINAMATH_CALUDE_equation_solution_l1304_130461

theorem equation_solution (x : ℝ) :
  (x / 3) / 3 = 9 / (x / 3) → x = 3^(5/2) ∨ x = -(3^(5/2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1304_130461


namespace NUMINAMATH_CALUDE_four_half_planes_theorem_l1304_130476

-- Define a half-plane
def HalfPlane : Type := ℝ × ℝ → Prop

-- Define a set of four half-planes
def FourHalfPlanes : Type := Fin 4 → HalfPlane

-- Define the property of covering the entire plane
def CoversPlane (planes : FourHalfPlanes) : Prop :=
  ∀ (x y : ℝ), ∃ (i : Fin 4), planes i (x, y)

-- Define the property of a subset of three half-planes covering the entire plane
def ThreeCoversPlane (planes : FourHalfPlanes) : Prop :=
  ∃ (i j k : Fin 4) (h : i ≠ j ∧ j ≠ k ∧ i ≠ k),
    ∀ (x y : ℝ), planes i (x, y) ∨ planes j (x, y) ∨ planes k (x, y)

-- The theorem to be proved
theorem four_half_planes_theorem (planes : FourHalfPlanes) :
  CoversPlane planes → ThreeCoversPlane planes :=
by
  sorry

end NUMINAMATH_CALUDE_four_half_planes_theorem_l1304_130476


namespace NUMINAMATH_CALUDE_sin_70_in_terms_of_sin_10_l1304_130418

theorem sin_70_in_terms_of_sin_10 (k : ℝ) (h : Real.sin (10 * π / 180) = k) :
  Real.sin (70 * π / 180) = 1 - 2 * k^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_70_in_terms_of_sin_10_l1304_130418


namespace NUMINAMATH_CALUDE_point_in_quadrant_iv_l1304_130484

/-- Given a system of equations x - y = a and 6x + 5y = -1, where x = 1,
    prove that the point (a, y) is in Quadrant IV -/
theorem point_in_quadrant_iv (a : ℚ) : 
  let x : ℚ := 1
  let y : ℚ := -7/5
  (x - y = a) → (6 * x + 5 * y = -1) → (a > 0 ∧ y < 0) := by
  sorry

#check point_in_quadrant_iv

end NUMINAMATH_CALUDE_point_in_quadrant_iv_l1304_130484


namespace NUMINAMATH_CALUDE_natural_numbers_with_special_last_digit_l1304_130439

def last_digit (n : ℕ) : ℕ := n % 10

def satisfies_condition (n : ℕ) : Prop :=
  n ≠ 0 ∧ n = 2016 * (last_digit n)

theorem natural_numbers_with_special_last_digit :
  {n : ℕ | satisfies_condition n} = {4032, 8064, 12096, 16128} :=
by sorry

end NUMINAMATH_CALUDE_natural_numbers_with_special_last_digit_l1304_130439


namespace NUMINAMATH_CALUDE_negative_power_fourth_l1304_130433

theorem negative_power_fourth (x : ℝ) : (-x^7)^4 = x^28 := by
  sorry

end NUMINAMATH_CALUDE_negative_power_fourth_l1304_130433
