import Mathlib

namespace NUMINAMATH_CALUDE_shaded_region_area_is_zero_l4160_416039

/-- A rectangle with height 8 and width 12 -/
structure Rectangle where
  height : ℝ
  width : ℝ
  height_eq : height = 8
  width_eq : width = 12

/-- A right triangle with base 12 and height 8 -/
structure RightTriangle where
  base : ℝ
  height : ℝ
  base_eq : base = 12
  height_eq : height = 8

/-- The shaded region formed by the segment and parts of the rectangle and triangle -/
def shadedRegion (r : Rectangle) (t : RightTriangle) : ℝ := sorry

/-- The theorem stating that the area of the shaded region is 0 -/
theorem shaded_region_area_is_zero (r : Rectangle) (t : RightTriangle) :
  shadedRegion r t = 0 := by sorry

end NUMINAMATH_CALUDE_shaded_region_area_is_zero_l4160_416039


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l4160_416011

theorem isosceles_triangle_quadratic_roots (a b k : ℝ) : 
  (∃ c : ℝ, c = 4 ∧ 
   (a = b ∧ (a + b = c ∨ a + c = b ∨ b + c = a)) ∧
   a^2 - 12*a + k + 2 = 0 ∧
   b^2 - 12*b + k + 2 = 0) →
  k = 34 ∨ k = 30 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l4160_416011


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l4160_416094

/-- Given a curve y = x^2 + a ln(x) where a > 0, if the minimum value of the slope
    of the tangent line at any point on the curve is 4, then the coordinates of the
    point of tangency at this minimum slope are (1, 1). -/
theorem tangent_point_coordinates (a : ℝ) (h1 : a > 0) :
  (∀ x > 0, 2 * x + a / x ≥ 4) ∧ (∃ x > 0, 2 * x + a / x = 4) →
  ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ y = x^2 + a * Real.log x ∧ 2 * x + a / x = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l4160_416094


namespace NUMINAMATH_CALUDE_circle_area_from_parallel_chords_l4160_416068

-- Define the circle C
def C : Real → Real → Prop := sorry

-- Define the two lines
def line1 (x y : Real) : Prop := x - y - 1 = 0
def line2 (x y : Real) : Prop := x - y - 5 = 0

-- Define the chord length
def chord_length : Real := 10

-- Theorem statement
theorem circle_area_from_parallel_chords 
  (h1 : ∃ (x1 y1 x2 y2 : Real), C x1 y1 ∧ C x2 y2 ∧ line1 x1 y1 ∧ line1 x2 y2)
  (h2 : ∃ (x3 y3 x4 y4 : Real), C x3 y3 ∧ C x4 y4 ∧ line2 x3 y3 ∧ line2 x4 y4)
  (h3 : ∀ (x1 y1 x2 y2 : Real), C x1 y1 ∧ C x2 y2 ∧ line1 x1 y1 ∧ line1 x2 y2 → 
        Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = chord_length)
  (h4 : ∀ (x3 y3 x4 y4 : Real), C x3 y3 ∧ C x4 y4 ∧ line2 x3 y3 ∧ line2 x4 y4 → 
        Real.sqrt ((x3 - x4)^2 + (y3 - y4)^2) = chord_length) :
  (∃ (r : Real), ∀ (x y : Real), C x y ↔ (x - 0)^2 + (y - 0)^2 = r^2) ∧ 
  (∃ (area : Real), area = 27 * Real.pi) :=
sorry

end NUMINAMATH_CALUDE_circle_area_from_parallel_chords_l4160_416068


namespace NUMINAMATH_CALUDE_task_completion_time_l4160_416076

theorem task_completion_time 
  (time_A : ℝ) (time_B : ℝ) (time_C : ℝ) 
  (rest_A : ℝ) (rest_B : ℝ) :
  time_A = 10 →
  time_B = 15 →
  time_C = 15 →
  rest_A = 1 →
  rest_B = 2 →
  (1 - rest_B / time_B - (1 / time_A + 1 / time_B) * rest_A) / 
  (1 / time_A + 1 / time_B + 1 / time_C) + rest_A + rest_B = 29/7 :=
by sorry

end NUMINAMATH_CALUDE_task_completion_time_l4160_416076


namespace NUMINAMATH_CALUDE_certain_number_value_l4160_416084

theorem certain_number_value : ∃ x : ℝ, 0.65 * 40 = (4/5) * x + 6 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l4160_416084


namespace NUMINAMATH_CALUDE_xavier_yasmin_age_ratio_l4160_416088

/-- Proves that the ratio of Xavier's age to Yasmin's age is 2:1 given the conditions -/
theorem xavier_yasmin_age_ratio :
  ∀ (xavier_age yasmin_age : ℕ),
    xavier_age + yasmin_age = 36 →
    xavier_age + 6 = 30 →
    xavier_age / yasmin_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_xavier_yasmin_age_ratio_l4160_416088


namespace NUMINAMATH_CALUDE_sticker_problem_solution_l4160_416070

def sticker_problem (initial_stickers : ℕ) (front_page_stickers : ℕ) (stickers_per_page : ℕ) (remaining_stickers : ℕ) : ℕ :=
  (initial_stickers - remaining_stickers - front_page_stickers) / stickers_per_page

theorem sticker_problem_solution :
  sticker_problem 89 3 7 44 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sticker_problem_solution_l4160_416070


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_axis_of_symmetry_example_l4160_416085

/-- The axis of symmetry of a parabola y = ax^2 + bx + c is x = -b / (2a) -/
theorem parabola_axis_of_symmetry (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  ∃ x₀ : ℝ, x₀ = -b / (2 * a) ∧ ∀ x : ℝ, f (x₀ + x) = f (x₀ - x) :=
sorry

/-- The axis of symmetry of the parabola y = -x^2 + 4x + 1 is the line x = 2 -/
theorem axis_of_symmetry_example :
  let f : ℝ → ℝ := λ x => -x^2 + 4*x + 1
  ∃ x₀ : ℝ, x₀ = 2 ∧ ∀ x : ℝ, f (x₀ + x) = f (x₀ - x) :=
sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_axis_of_symmetry_example_l4160_416085


namespace NUMINAMATH_CALUDE_two_rational_solutions_l4160_416008

-- Define the system of equations
def system (x y z : ℚ) : Prop :=
  x + y + z = 0 ∧ x * y * z + z = 0 ∧ x * y + y * z + x * z + y = 0

-- Theorem stating that there are exactly two rational solutions
theorem two_rational_solutions :
  ∃! (s : Finset (ℚ × ℚ × ℚ)), s.card = 2 ∧ ∀ (x y z : ℚ), (x, y, z) ∈ s ↔ system x y z :=
sorry

end NUMINAMATH_CALUDE_two_rational_solutions_l4160_416008


namespace NUMINAMATH_CALUDE_turtle_difference_is_nine_l4160_416063

/-- Given the number of turtles Kristen has, calculate the difference between Trey's and Kristen's turtle counts. -/
def turtle_difference (kristen_turtles : ℕ) : ℕ :=
  let kris_turtles := kristen_turtles / 4
  let trey_turtles := 7 * kris_turtles
  trey_turtles - kristen_turtles

/-- Theorem stating that the difference between Trey's and Kristen's turtle counts is 9 when Kristen has 12 turtles. -/
theorem turtle_difference_is_nine :
  turtle_difference 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_turtle_difference_is_nine_l4160_416063


namespace NUMINAMATH_CALUDE_intersection_A_B_l4160_416078

-- Define set A
def A : Set ℝ := {x | x - 1 ≤ 0}

-- Define set B
def B : Set ℝ := {0, 1, 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l4160_416078


namespace NUMINAMATH_CALUDE_thirty_five_million_scientific_notation_l4160_416046

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive integer to scientific notation -/
def to_scientific_notation (n : ℕ+) : ScientificNotation :=
  sorry

theorem thirty_five_million_scientific_notation :
  to_scientific_notation 35000000 = ScientificNotation.mk 3.5 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_thirty_five_million_scientific_notation_l4160_416046


namespace NUMINAMATH_CALUDE_partition_theorem_l4160_416040

theorem partition_theorem (a b : ℝ) (ha : 1 < a) (hab : a < 2) (hb : 2 < b) :
  (¬ ∃ (A₀ A₁ : Set ℕ), (A₀ ∪ A₁ = Set.univ) ∧ (A₀ ∩ A₁ = ∅) ∧
    (∀ (j : Fin 2) (m n : ℕ), m ∈ (if j = 0 then A₀ else A₁) → n ∈ (if j = 0 then A₀ else A₁) →
      (n / m : ℝ) < a ∨ (n / m : ℝ) > b)) ∧
  ((∃ (A₀ A₁ A₂ : Set ℕ), (A₀ ∪ A₁ ∪ A₂ = Set.univ) ∧ (A₀ ∩ A₁ = ∅) ∧ (A₀ ∩ A₂ = ∅) ∧ (A₁ ∩ A₂ = ∅) ∧
    (∀ (j : Fin 3) (m n : ℕ),
      m ∈ (if j = 0 then A₀ else if j = 1 then A₁ else A₂) →
      n ∈ (if j = 0 then A₀ else if j = 1 then A₁ else A₂) →
        (n / m : ℝ) < a ∨ (n / m : ℝ) > b)) ↔ b ≤ a^2) :=
by sorry

end NUMINAMATH_CALUDE_partition_theorem_l4160_416040


namespace NUMINAMATH_CALUDE_initial_songs_count_l4160_416082

/-- 
Given an album where:
- Each song is 3 minutes long
- Adding 10 more songs will make the total listening time 105 minutes
Prove that the initial number of songs in the album is 25.
-/
theorem initial_songs_count (song_duration : ℕ) (additional_songs : ℕ) (total_duration : ℕ) :
  song_duration = 3 →
  additional_songs = 10 →
  total_duration = 105 →
  ∃ (initial_songs : ℕ), song_duration * (initial_songs + additional_songs) = total_duration ∧ initial_songs = 25 :=
by sorry

end NUMINAMATH_CALUDE_initial_songs_count_l4160_416082


namespace NUMINAMATH_CALUDE_pencil_case_cost_solution_l4160_416045

/-- Calculates the amount spent on a pencil case given the initial amount,
    the amount spent on a toy truck, and the remaining amount. -/
def pencil_case_cost (initial : ℝ) (toy_truck : ℝ) (remaining : ℝ) : ℝ :=
  initial - toy_truck - remaining

theorem pencil_case_cost_solution :
  pencil_case_cost 10 3 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_case_cost_solution_l4160_416045


namespace NUMINAMATH_CALUDE_inequality_of_exponential_l4160_416030

theorem inequality_of_exponential (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (1/3 : ℝ)^a < (1/3 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_exponential_l4160_416030


namespace NUMINAMATH_CALUDE_sum_of_coordinates_A_l4160_416095

/-- Given three points A, B, and C in a 2D plane satisfying certain conditions,
    prove that the sum of the coordinates of A is 8.5. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (C.2 - A.2) / (B.2 - A.2) = 1/3 →
  B = (2, 8) →
  C = (5, 14) →
  A.1 + A.2 = 8.5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_A_l4160_416095


namespace NUMINAMATH_CALUDE_equation_solution_l4160_416023

theorem equation_solution :
  ∃! x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4160_416023


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_three_pi_eighths_l4160_416025

theorem cos_squared_minus_sin_squared_three_pi_eighths :
  Real.cos (3 * Real.pi / 8) ^ 2 - Real.sin (3 * Real.pi / 8) ^ 2 = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_three_pi_eighths_l4160_416025


namespace NUMINAMATH_CALUDE_percentage_increase_l4160_416034

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 400 → final = 480 → (final - initial) / initial * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l4160_416034


namespace NUMINAMATH_CALUDE_subset_iff_m_le_three_l4160_416001

-- Define the sets A and B
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2*m - 1 }

-- State the theorem
theorem subset_iff_m_le_three (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_iff_m_le_three_l4160_416001


namespace NUMINAMATH_CALUDE_project_duration_l4160_416054

theorem project_duration (x : ℝ) : 
  (1 / (x - 6) = 1.4 * (1 / x)) → x = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_project_duration_l4160_416054


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l4160_416066

-- Define the triangle ABC
structure Triangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :=
  (A B C : α)

-- Define an isosceles triangle
def IsIsosceles {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : Prop :=
  ‖t.A - t.B‖ = ‖t.A - t.C‖

-- Define the incenter
def Incenter {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : α :=
  sorry  -- Definition of incenter omitted for brevity

-- Define the distance from a point to a line segment
def DistanceToSegment {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (P : α) (A B : α) : ℝ :=
  sorry  -- Definition of distance to segment omitted for brevity

-- Theorem statement
theorem isosceles_triangle_side_length 
  {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (t : Triangle α) (I : α) :
  IsIsosceles t →
  I = Incenter t →
  ‖t.A - I‖ = 3 →
  DistanceToSegment I t.B t.C = 2 →
  ‖t.B - t.C‖ = 4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l4160_416066


namespace NUMINAMATH_CALUDE_divisor_greater_than_remainder_l4160_416092

theorem divisor_greater_than_remainder (a b q r : ℕ) : 
  a = b * q + r → r = 8 → b > 8 := by sorry

end NUMINAMATH_CALUDE_divisor_greater_than_remainder_l4160_416092


namespace NUMINAMATH_CALUDE_ladder_velocity_l4160_416012

theorem ladder_velocity (l a τ : ℝ) (hl : l > 0) (ha : a > 0) (hτ : τ > 0) :
  let α := Real.arcsin (a * τ^2 / (2 * l))
  let v₁ := a * τ
  let v₂ := (a^2 * τ^3) / Real.sqrt (4 * l^2 - a^2 * τ^4)
  v₁ * Real.sin α = v₂ * Real.cos α :=
by sorry

end NUMINAMATH_CALUDE_ladder_velocity_l4160_416012


namespace NUMINAMATH_CALUDE_identical_coordinate_point_exists_l4160_416026

/-- Represents a 2D rectangular coordinate system -/
structure CoordinateSystem :=
  (origin : ℝ × ℝ)
  (xAxis : ℝ × ℝ)
  (yAxis : ℝ × ℝ)
  (unitLength : ℝ)

/-- Theorem: Existence of a point with identical coordinates in two different coordinate systems -/
theorem identical_coordinate_point_exists 
  (cs1 cs2 : CoordinateSystem) 
  (h1 : cs1.origin ≠ cs2.origin) 
  (h2 : ¬ (∃ k : ℝ, cs1.xAxis = k • cs2.xAxis)) 
  (h3 : cs1.unitLength ≠ cs2.unitLength) : 
  ∃ p : ℝ × ℝ, ∃ x y : ℝ, 
    (x, y) = p ∧ 
    (∃ x' y' : ℝ, (x', y') = p ∧ x = x' ∧ y = y') :=
sorry

end NUMINAMATH_CALUDE_identical_coordinate_point_exists_l4160_416026


namespace NUMINAMATH_CALUDE_problem_solution_l4160_416029

theorem problem_solution (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 4 = d + 2 * Real.sqrt (a + b + c - d)) : 
  d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4160_416029


namespace NUMINAMATH_CALUDE_compute_expression_l4160_416017

theorem compute_expression : 6^3 - 5*7 + 2^4 = 197 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l4160_416017


namespace NUMINAMATH_CALUDE_inequality_solution_range_l4160_416027

theorem inequality_solution_range (m : ℝ) : 
  (∀ x : ℕ+, (x = 1 ∨ x = 2 ∨ x = 3) ↔ 3 * (x : ℝ) - m ≤ 0) →
  (9 ≤ m ∧ m < 12) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l4160_416027


namespace NUMINAMATH_CALUDE_scarlet_earrings_cost_l4160_416096

/-- Calculates the cost of earrings given initial savings, necklace cost, and remaining money --/
def earrings_cost (initial_savings : ℕ) (necklace_cost : ℕ) (remaining : ℕ) : ℕ :=
  initial_savings - necklace_cost - remaining

/-- Proves that the cost of earrings is 23 given the problem conditions --/
theorem scarlet_earrings_cost :
  let initial_savings : ℕ := 80
  let necklace_cost : ℕ := 48
  let remaining : ℕ := 9
  earrings_cost initial_savings necklace_cost remaining = 23 := by
  sorry

end NUMINAMATH_CALUDE_scarlet_earrings_cost_l4160_416096


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l4160_416083

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- The sum of interior angles of a polygon with n sides -/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: Each interior angle of a regular octagon measures 135 degrees -/
theorem regular_octagon_interior_angle :
  (interior_angle_sum octagon_sides) / octagon_sides = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l4160_416083


namespace NUMINAMATH_CALUDE_difference_of_sixes_in_7669_l4160_416006

/-- Given a natural number n, returns the digit at the i-th place (0-indexed from right) -/
def digit_at_place (n : ℕ) (i : ℕ) : ℕ := 
  (n / (10 ^ i)) % 10

/-- Given a natural number n, returns the place value of the digit at the i-th place -/
def place_value (n : ℕ) (i : ℕ) : ℕ := 
  digit_at_place n i * (10 ^ i)

theorem difference_of_sixes_in_7669 : 
  place_value 7669 2 - place_value 7669 1 = 540 := by sorry

end NUMINAMATH_CALUDE_difference_of_sixes_in_7669_l4160_416006


namespace NUMINAMATH_CALUDE_subtracted_number_l4160_416009

theorem subtracted_number (x y : ℤ) : x = 48 → 5 * x - y = 102 → y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l4160_416009


namespace NUMINAMATH_CALUDE_roller_coaster_tickets_l4160_416020

theorem roller_coaster_tickets : ∃ x : ℕ, 
  (∀ y : ℕ, y = 3 → 7 * x + 4 * y = 47) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_tickets_l4160_416020


namespace NUMINAMATH_CALUDE_current_intensity_bound_l4160_416074

/-- Given a voltage and a minimum resistance, the current intensity is bounded above. -/
theorem current_intensity_bound (U R : ℝ) (hU : U = 200) (hR : R ≥ 62.5) :
  let I := U / R
  I ≤ 3.2 := by
  sorry

end NUMINAMATH_CALUDE_current_intensity_bound_l4160_416074


namespace NUMINAMATH_CALUDE_wood_measurement_equations_l4160_416019

/-- Represents the wood measurement problem from "The Mathematical Classic of Sunzi" -/
def wood_measurement_problem (x y : ℝ) : Prop :=
  (y - x = 4.5) ∧ (y / 2 = x - 1)

/-- The correct system of equations for the wood measurement problem -/
theorem wood_measurement_equations :
  ∃ x y : ℝ,
    (x > 0) ∧  -- Length of wood is positive
    (y > 0) ∧  -- Length of rope is positive
    (y > x) ∧  -- Rope is longer than wood
    wood_measurement_problem x y :=
sorry

end NUMINAMATH_CALUDE_wood_measurement_equations_l4160_416019


namespace NUMINAMATH_CALUDE_evaluate_polynomial_at_negative_two_l4160_416086

theorem evaluate_polynomial_at_negative_two :
  let y : ℤ := -2
  y^3 - y^2 + 2*y + 4 = -12 := by
sorry

end NUMINAMATH_CALUDE_evaluate_polynomial_at_negative_two_l4160_416086


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l4160_416044

theorem ferris_wheel_capacity (total_seats broken_seats people_riding : ℕ) 
  (h1 : total_seats = 18)
  (h2 : broken_seats = 10)
  (h3 : people_riding = 120) :
  people_riding / (total_seats - broken_seats) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l4160_416044


namespace NUMINAMATH_CALUDE_tan_value_proof_l4160_416002

theorem tan_value_proof (α : Real) 
  (h1 : Real.sin α - Real.cos α = 1/5)
  (h2 : 0 < α ∧ α < Real.pi) : 
  Real.tan α = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_proof_l4160_416002


namespace NUMINAMATH_CALUDE_june_bird_eggs_l4160_416065

/-- The number of eggs in each nest in the first tree -/
def eggs_per_nest_tree1 : ℕ := 5

/-- The number of nests in the first tree -/
def nests_in_tree1 : ℕ := 2

/-- The number of eggs in the nest in the second tree -/
def eggs_in_tree2 : ℕ := 3

/-- The number of eggs in the nest in the front yard -/
def eggs_in_front_yard : ℕ := 4

/-- The total number of bird eggs June found -/
def total_eggs : ℕ := nests_in_tree1 * eggs_per_nest_tree1 + eggs_in_tree2 + eggs_in_front_yard

theorem june_bird_eggs : total_eggs = 17 := by
  sorry

end NUMINAMATH_CALUDE_june_bird_eggs_l4160_416065


namespace NUMINAMATH_CALUDE_f_properties_l4160_416099

noncomputable def f (x : ℝ) := Real.exp (-x) * Real.sin x

theorem f_properties :
  let a := -Real.pi
  let b := Real.pi
  ∃ (max_val min_val : ℝ),
    (∀ x ∈ Set.Icc a b, f x ≤ max_val) ∧
    (∃ x ∈ Set.Icc a b, f x = max_val) ∧
    (∀ x ∈ Set.Icc a b, min_val ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min_val) ∧
    (StrictMonoOn f (Set.Ioo (-3*Real.pi/4) (Real.pi/4))) ∧
    (StrictAntiOn f (Set.Ioc a (-3*Real.pi/4))) ∧
    (StrictAntiOn f (Set.Ico (Real.pi/4) b)) ∧
    max_val = (Real.sqrt 2 / 2) * Real.exp (-Real.pi/4) ∧
    min_val = -(Real.sqrt 2 / 2) * Real.exp (3*Real.pi/4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4160_416099


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l4160_416033

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₂ = -5 and the common difference is 3,
    prove that a₁ = -8 -/
theorem arithmetic_sequence_first_term
  (a : ℕ → ℝ)
  (h_arith : IsArithmeticSequence a)
  (h_a2 : a 2 = -5)
  (h_d : ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  a 1 = -8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l4160_416033


namespace NUMINAMATH_CALUDE_increase_averages_possible_l4160_416064

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem increase_averages_possible :
  ∃ g ∈ group1,
    average (group1.filter (· ≠ g)) > average group1 ∧
    average (g :: group2) > average group2 := by
  sorry

end NUMINAMATH_CALUDE_increase_averages_possible_l4160_416064


namespace NUMINAMATH_CALUDE_rectangle_circle_equality_l4160_416021

/-- Given a rectangle with sides a and b, where a = 2b, and a circle with radius 3,
    if the perimeter of the rectangle equals the circumference of the circle,
    then a = 2π and b = π. -/
theorem rectangle_circle_equality (a b : ℝ) :
  a = 2 * b →
  2 * (a + b) = 2 * π * 3 →
  a = 2 * π ∧ b = π := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_equality_l4160_416021


namespace NUMINAMATH_CALUDE_dot_product_range_l4160_416014

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse x^2 + y^2/9 = 1 -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 + p.y^2/9 = 1

/-- Checks if two points are symmetric about the origin -/
def areSymmetric (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = -q.y

/-- Calculates the dot product of vectors CA and CB -/
def dotProduct (a b c : Point) : ℝ :=
  (a.x - c.x) * (b.x - c.x) + (a.y - c.y) * (b.y - c.y)

theorem dot_product_range :
  ∀ (a b : Point),
    isOnEllipse a →
    isOnEllipse b →
    areSymmetric a b →
    let c := Point.mk 5 5
    41 ≤ dotProduct a b c ∧ dotProduct a b c ≤ 49 := by
  sorry


end NUMINAMATH_CALUDE_dot_product_range_l4160_416014


namespace NUMINAMATH_CALUDE_habitable_earth_surface_fraction_l4160_416000

theorem habitable_earth_surface_fraction :
  let total_surface : ℚ := 1
  let water_covered_fraction : ℚ := 2/3
  let land_fraction : ℚ := 1 - water_covered_fraction
  let inhabitable_land_fraction : ℚ := 2/3
  inhabitable_land_fraction * land_fraction = 2/9 :=
by sorry

end NUMINAMATH_CALUDE_habitable_earth_surface_fraction_l4160_416000


namespace NUMINAMATH_CALUDE_inequality_solution_l4160_416097

theorem inequality_solution (c : ℝ) : 
  (4 * c / 3 ≤ 8 + 4 * c ∧ 8 + 4 * c < -3 * (1 + c)) ↔ 
  (c ≥ -3 ∧ c < -11/7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4160_416097


namespace NUMINAMATH_CALUDE_negation_of_exp_gt_ln_proposition_l4160_416057

open Real

theorem negation_of_exp_gt_ln_proposition :
  (¬ ∀ x : ℝ, x > 0 → exp x > log x) ↔ (∃ x : ℝ, x > 0 ∧ exp x ≤ log x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exp_gt_ln_proposition_l4160_416057


namespace NUMINAMATH_CALUDE_second_train_length_problem_l4160_416003

/-- Calculates the length of the second train given the conditions of the problem -/
def second_train_length (first_train_length : ℝ) (first_train_speed : ℝ) (second_train_speed : ℝ) (time_to_cross : ℝ) : ℝ :=
  let relative_speed := first_train_speed - second_train_speed
  let total_distance := relative_speed * time_to_cross
  total_distance - first_train_length

/-- Theorem stating that given the problem conditions, the length of the second train is 299.9440044796417 m -/
theorem second_train_length_problem :
  let first_train_length : ℝ := 400
  let first_train_speed : ℝ := 72 * 1000 / 3600  -- Convert km/h to m/s
  let second_train_speed : ℝ := 36 * 1000 / 3600 -- Convert km/h to m/s
  let time_to_cross : ℝ := 69.99440044796417
  second_train_length first_train_length first_train_speed second_train_speed time_to_cross = 299.9440044796417 := by
  sorry

end NUMINAMATH_CALUDE_second_train_length_problem_l4160_416003


namespace NUMINAMATH_CALUDE_sasha_remainder_l4160_416052

theorem sasha_remainder (n : ℕ) (a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  n = 103 * c + d ∧ 
  0 ≤ b ∧ b < 102 ∧
  0 ≤ d ∧ d < 103 ∧
  a + d = 20 →
  b = 20 :=
by sorry

end NUMINAMATH_CALUDE_sasha_remainder_l4160_416052


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l4160_416058

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 25) : 
  r - p = 30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l4160_416058


namespace NUMINAMATH_CALUDE_corresponding_sides_of_congruent_triangles_are_equal_l4160_416091

-- Define a triangle as a structure with three points
structure Triangle (α : Type*) :=
  (A B C : α)

-- Define congruence for triangles
def CongruentTriangles {α : Type*} (T1 T2 : Triangle α) : Prop :=
  sorry

-- Define the concept of corresponding sides
def CorrespondingSides {α : Type*} (T1 T2 : Triangle α) (s1 s2 : α × α) : Prop :=
  sorry

-- Define equality of sides
def EqualSides {α : Type*} (s1 s2 : α × α) : Prop :=
  sorry

-- Theorem: Corresponding sides of congruent triangles are equal
theorem corresponding_sides_of_congruent_triangles_are_equal
  {α : Type*} (T1 T2 : Triangle α) :
  CongruentTriangles T1 T2 →
  ∀ s1 s2, CorrespondingSides T1 T2 s1 s2 → EqualSides s1 s2 :=
by
  sorry

end NUMINAMATH_CALUDE_corresponding_sides_of_congruent_triangles_are_equal_l4160_416091


namespace NUMINAMATH_CALUDE_trig_expression_equality_l4160_416093

theorem trig_expression_equality : 
  2 * Real.sin (30 * π / 180) - Real.tan (45 * π / 180) - Real.sqrt ((1 - Real.tan (60 * π / 180))^2) = Real.sqrt 3 - 1 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l4160_416093


namespace NUMINAMATH_CALUDE_election_votes_l4160_416056

theorem election_votes (total_members : ℕ) (winner_percentage : ℚ) (winner_total_percentage : ℚ) :
  total_members = 1600 →
  winner_percentage = 60 / 100 →
  winner_total_percentage = 19.6875 / 100 →
  (↑total_members * winner_total_percentage : ℚ) / winner_percentage = 525 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l4160_416056


namespace NUMINAMATH_CALUDE_bus_interval_theorem_l4160_416072

/-- Given a circular bus route with two buses operating at the same speed with an interval of 21 minutes,
    the interval between three buses operating on the same route at the same speed is 14 minutes. -/
theorem bus_interval_theorem (interval_two_buses : ℕ) (interval_three_buses : ℕ) : 
  interval_two_buses = 21 → interval_three_buses = (2 * interval_two_buses) / 3 := by
  sorry

end NUMINAMATH_CALUDE_bus_interval_theorem_l4160_416072


namespace NUMINAMATH_CALUDE_isosceles_when_root_is_one_right_angled_when_equal_roots_l4160_416049

/-- Triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Quadratic equation associated with the triangle -/
def quadratic_equation (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 - 2 * t.b * x - t.a + t.c

theorem isosceles_when_root_is_one (t : Triangle) :
  quadratic_equation t 1 = 0 → t.b = t.c :=
sorry

theorem right_angled_when_equal_roots (t : Triangle) :
  (∃ x : ℝ, ∀ y : ℝ, quadratic_equation t y = 0 ↔ y = x) →
  t.a^2 + t.b^2 = t.c^2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_when_root_is_one_right_angled_when_equal_roots_l4160_416049


namespace NUMINAMATH_CALUDE_final_distance_is_35_l4160_416048

/-- Represents the movements of the first car -/
structure FirstCarMovement where
  initial_run : ℝ
  right_turn : ℝ
  left_turn : ℝ

/-- Represents the movement of the second car -/
def SecondCarMovement : ℝ := 35

/-- Calculates the final distance between two cars given their movements -/
def finalDistance (initial_distance : ℝ) (first_car : FirstCarMovement) (second_car : ℝ) : ℝ :=
  initial_distance - (first_car.initial_run + 2 * first_car.right_turn + first_car.left_turn) - second_car

/-- Theorem stating that the final distance between the cars is 35 km -/
theorem final_distance_is_35 :
  let first_car : FirstCarMovement := ⟨25, 15, 25⟩
  let second_car : ℝ := SecondCarMovement
  finalDistance 150 first_car second_car = 35 := by
  sorry


end NUMINAMATH_CALUDE_final_distance_is_35_l4160_416048


namespace NUMINAMATH_CALUDE_letter_150_is_B_l4160_416031

def letter_sequence : ℕ → Char
  | n => match n % 4 with
    | 0 => 'A'
    | 1 => 'B'
    | 2 => 'C'
    | _ => 'D'

theorem letter_150_is_B : letter_sequence 149 = 'B' := by
  sorry

end NUMINAMATH_CALUDE_letter_150_is_B_l4160_416031


namespace NUMINAMATH_CALUDE_balloons_bought_at_park_l4160_416041

theorem balloons_bought_at_park (allan_balloons jake_initial_balloons : ℕ) 
  (h1 : allan_balloons = 6)
  (h2 : jake_initial_balloons = 3)
  (h3 : ∃ (x : ℕ), jake_initial_balloons + x = allan_balloons + 1) :
  ∃ (x : ℕ), x = 4 ∧ jake_initial_balloons + x = allan_balloons + 1 := by
sorry

end NUMINAMATH_CALUDE_balloons_bought_at_park_l4160_416041


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4160_416042

theorem complex_magnitude_problem (z : ℂ) : z = (2 + I) / (1 - I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4160_416042


namespace NUMINAMATH_CALUDE_particular_number_problem_l4160_416073

theorem particular_number_problem (x : ℚ) (h : (x + 10) / 5 = 4) : 3 * x - 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_problem_l4160_416073


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l4160_416053

def C : Set Nat := {65, 67, 68, 71, 73}

def hasSmallerPrimeFactor (a b : Nat) : Prop :=
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ p ∣ a ∧ q ∣ b ∧ ∀ r < q, Nat.Prime r → ¬(r ∣ b)

theorem smallest_prime_factor_in_C :
  ∀ n ∈ C, n ≠ 68 → hasSmallerPrimeFactor 68 n :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l4160_416053


namespace NUMINAMATH_CALUDE_pump_calculations_l4160_416071

/-- Ultraflow pump rate in gallons per hour -/
def ultraflow_rate : ℚ := 560

/-- MiniFlow pump rate in gallons per hour -/
def miniflow_rate : ℚ := 220

/-- Convert minutes to hours -/
def minutes_to_hours (minutes : ℚ) : ℚ := minutes / 60

/-- Calculate gallons pumped given rate and time -/
def gallons_pumped (rate : ℚ) (time : ℚ) : ℚ := rate * time

theorem pump_calculations :
  (gallons_pumped ultraflow_rate (minutes_to_hours 75) = 700) ∧
  (gallons_pumped ultraflow_rate (minutes_to_hours 50) + gallons_pumped miniflow_rate (minutes_to_hours 50) = 883) := by
  sorry

end NUMINAMATH_CALUDE_pump_calculations_l4160_416071


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l4160_416035

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, -1 < x ∧ x < 3 → -2 < x ∧ x < 4) ∧
  (∃ x : ℝ, -2 < x ∧ x < 4 ∧ ¬(-1 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l4160_416035


namespace NUMINAMATH_CALUDE_unique_integer_sum_pair_l4160_416004

theorem unique_integer_sum_pair (a : ℕ → ℝ) (h1 : 1 < a 1 ∧ a 1 < 2) 
  (h2 : ∀ k : ℕ, a (k + 1) = a k + k / a k) :
  ∃! (i j : ℕ), i ≠ j ∧ ∃ m : ℤ, (a i + a j : ℝ) = m := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_sum_pair_l4160_416004


namespace NUMINAMATH_CALUDE_m_plus_one_value_l4160_416018

theorem m_plus_one_value (m n : ℕ) 
  (h1 : m * n = 121) 
  (h2 : (m + 1) * (n + 1) = 1000) : 
  m + 1 = 879 - n := by
sorry

end NUMINAMATH_CALUDE_m_plus_one_value_l4160_416018


namespace NUMINAMATH_CALUDE_max_covered_squares_l4160_416075

def checkerboard_width : ℕ := 15
def checkerboard_height : ℕ := 36
def tile_side_1 : ℕ := 7
def tile_side_2 : ℕ := 5

theorem max_covered_squares :
  ∃ (n m : ℕ),
    n * (tile_side_1 ^ 2) + m * (tile_side_2 ^ 2) = checkerboard_width * checkerboard_height ∧
    ∀ (k l : ℕ),
      k * (tile_side_1 ^ 2) + l * (tile_side_2 ^ 2) ≤ checkerboard_width * checkerboard_height →
      k * (tile_side_1 ^ 2) + l * (tile_side_2 ^ 2) ≤ n * (tile_side_1 ^ 2) + m * (tile_side_2 ^ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_max_covered_squares_l4160_416075


namespace NUMINAMATH_CALUDE_euler_line_concurrency_l4160_416080

/-- A point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The Euler line of a triangle -/
def EulerLine (A B C : Point) : Set Point := sorry

/-- The point of concurrency of three lines -/
def Concurrent (l1 l2 l3 : Set Point) : Point := sorry

/-- Predicate to check if a triangle is not obtuse -/
def NotObtuse (A B C : Point) : Prop := sorry

theorem euler_line_concurrency 
  (A B C D : Point) 
  (h1 : NotObtuse A B C) 
  (h2 : NotObtuse B C D) 
  (h3 : NotObtuse C A D) 
  (h4 : NotObtuse D A B) 
  (P : Point) 
  (hP : P = Concurrent (EulerLine A B C) (EulerLine B C D) (EulerLine C A D)) :
  P ∈ EulerLine D A B := by
  sorry

end NUMINAMATH_CALUDE_euler_line_concurrency_l4160_416080


namespace NUMINAMATH_CALUDE_robins_hair_growth_l4160_416059

/-- Calculates the hair growth given initial length, final length, and cut length -/
def hair_growth (initial_length final_length cut_length : ℕ) : ℕ :=
  cut_length + final_length - initial_length

/-- Theorem: Given Robin's hair scenario, the hair growth is 8 inches -/
theorem robins_hair_growth :
  hair_growth 14 2 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_robins_hair_growth_l4160_416059


namespace NUMINAMATH_CALUDE_minimum_cut_length_l4160_416007

theorem minimum_cut_length (longer_strip shorter_strip : ℝ) 
  (h1 : longer_strip = 23)
  (h2 : shorter_strip = 15) : 
  ∃ x : ℝ, x ≥ 7 ∧ ∀ y : ℝ, y ≥ 0 → longer_strip - y ≥ 2 * (shorter_strip - y) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_minimum_cut_length_l4160_416007


namespace NUMINAMATH_CALUDE_max_value_of_function_l4160_416069

/-- The function f(x) = x^2(1-3x) has a maximum value of 1/12 in the interval (0, 1/3) -/
theorem max_value_of_function : 
  ∃ (c : ℝ), c ∈ Set.Ioo 0 (1/3) ∧ 
  (∀ x, x ∈ Set.Ioo 0 (1/3) → x^2 * (1 - 3*x) ≤ c) ∧
  c = 1/12 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l4160_416069


namespace NUMINAMATH_CALUDE_triangle_max_area_l4160_416081

theorem triangle_max_area (a b c : ℝ) (h1 : (a + b - c) * (a + b + c) = 3 * a * b) (h2 : c = 4) :
  ∃ (S : ℝ), S = (4 : ℝ) * Real.sqrt 3 ∧ ∀ (area : ℝ), area = 1/2 * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) → area ≤ S :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l4160_416081


namespace NUMINAMATH_CALUDE_xiaoming_coins_l4160_416098

theorem xiaoming_coins (first_day : ℕ) (second_day : ℕ)
  (h1 : first_day = 22)
  (h2 : second_day = 12) :
  first_day + second_day = 34 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_coins_l4160_416098


namespace NUMINAMATH_CALUDE_people_at_game_l4160_416032

/-- The number of people who came to a little league game -/
theorem people_at_game (total_seats empty_seats : ℕ) 
  (h1 : total_seats = 92)
  (h2 : empty_seats = 45) :
  total_seats - empty_seats = 47 := by
  sorry

end NUMINAMATH_CALUDE_people_at_game_l4160_416032


namespace NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_for_non_intersection_l4160_416043

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define lines in the space
def Line (V : Type*) [NormedAddCommGroup V] := V → Set V

-- Define the property of being skew
def are_skew (l₁ l₂ : Line V) : Prop := sorry

-- Define the property of not intersecting
def do_not_intersect (l₁ l₂ : Line V) : Prop := sorry

-- Theorem statement
theorem skew_lines_sufficient_not_necessary_for_non_intersection :
  (∀ l₁ l₂ : Line V, are_skew l₁ l₂ → do_not_intersect l₁ l₂) ∧
  (∃ l₁ l₂ : Line V, do_not_intersect l₁ l₂ ∧ ¬are_skew l₁ l₂) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_for_non_intersection_l4160_416043


namespace NUMINAMATH_CALUDE_roses_money_proof_l4160_416013

/-- The amount of money Rose already has -/
def roses_money : ℝ := 7.10

/-- The cost of the paintbrush -/
def paintbrush_cost : ℝ := 2.40

/-- The cost of the set of paints -/
def paints_cost : ℝ := 9.20

/-- The cost of the easel -/
def easel_cost : ℝ := 6.50

/-- The additional amount Rose needs -/
def additional_needed : ℝ := 11

theorem roses_money_proof :
  roses_money + additional_needed = paintbrush_cost + paints_cost + easel_cost :=
by sorry

end NUMINAMATH_CALUDE_roses_money_proof_l4160_416013


namespace NUMINAMATH_CALUDE_class_average_problem_l4160_416036

theorem class_average_problem (percent_high : Real) (percent_mid : Real) (percent_low : Real)
  (score_high : Real) (score_low : Real) (overall_average : Real) :
  percent_high = 15 →
  percent_mid = 50 →
  percent_low = 35 →
  score_high = 100 →
  score_low = 63 →
  overall_average = 76.05 →
  (percent_high * score_high + percent_mid * ((percent_high * score_high + percent_mid * X + percent_low * score_low) / 100) + percent_low * score_low) / 100 = overall_average →
  X = 78 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l4160_416036


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l4160_416061

def A (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_equals_one (m : ℝ) :
  B m ⊆ A m → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l4160_416061


namespace NUMINAMATH_CALUDE_conic_not_parabola_l4160_416090

/-- A conic section represented by the equation x^2 + ky^2 = 1 -/
def conic (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + k * p.2^2 = 1}

/-- Definition of a parabola -/
def is_parabola (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ 
  ∀ (x y : ℝ), (x, y) ∈ S ↔ a * x^2 + b * x * y + c * y^2 + d * x + e * y = 0

/-- Theorem: The conic section x^2 + ky^2 = 1 is not a parabola for any real k -/
theorem conic_not_parabola : ∀ (k : ℝ), ¬(is_parabola (conic k)) := by
  sorry

end NUMINAMATH_CALUDE_conic_not_parabola_l4160_416090


namespace NUMINAMATH_CALUDE_sum_a5_a4_l4160_416038

def S (n : ℕ) : ℕ := n^2 + 2*n - 1

def a (n : ℕ) : ℕ := S n - S (n-1)

theorem sum_a5_a4 : a 5 + a 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_a5_a4_l4160_416038


namespace NUMINAMATH_CALUDE_root_equality_condition_l4160_416060

theorem root_equality_condition (m n p : ℕ) 
  (hm : Even m) (hn : Even n) (hp : Even p) 
  (hm_pos : m > 0) (hn_pos : n > 0) (hp_pos : p > 0) :
  (m - p : ℝ) ^ (1 / n) = (n - p : ℝ) ^ (1 / m) ↔ m = n ∧ m ≥ p :=
sorry

end NUMINAMATH_CALUDE_root_equality_condition_l4160_416060


namespace NUMINAMATH_CALUDE_camryn_practice_schedule_l4160_416087

theorem camryn_practice_schedule : Nat.lcm (Nat.lcm 11 3) 7 = 231 := by
  sorry

end NUMINAMATH_CALUDE_camryn_practice_schedule_l4160_416087


namespace NUMINAMATH_CALUDE_negation_of_conditional_l4160_416051

theorem negation_of_conditional (x : ℝ) :
  ¬(x^2 + x - 6 > 0 → x > 2 ∨ x < -3) ↔ (x^2 + x - 6 ≤ 0 → -3 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l4160_416051


namespace NUMINAMATH_CALUDE_gemstones_for_four_sets_l4160_416050

/-- The number of gemstones needed for a given number of earring sets -/
def gemstones_needed (num_sets : ℕ) : ℕ :=
  let magnets_per_earring : ℕ := 2
  let buttons_per_earring : ℕ := magnets_per_earring / 2
  let gemstones_per_earring : ℕ := buttons_per_earring * 3
  let earrings_per_set : ℕ := 2
  num_sets * earrings_per_set * gemstones_per_earring

/-- Theorem: 4 sets of earrings require 24 gemstones -/
theorem gemstones_for_four_sets : gemstones_needed 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gemstones_for_four_sets_l4160_416050


namespace NUMINAMATH_CALUDE_shorter_leg_is_15_l4160_416022

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ  -- shorter leg
  b : ℕ  -- longer leg
  c : ℕ  -- hypotenuse
  right_angle : a ^ 2 + b ^ 2 = c ^ 2
  a_shorter : a ≤ b

/-- The length of the shorter leg in a right triangle with hypotenuse 25 -/
def shorter_leg_length : ℕ := 15

/-- Theorem stating that in a right triangle with integer side lengths and hypotenuse 25, 
    the shorter leg has length 15 -/
theorem shorter_leg_is_15 (t : RightTriangle) (hyp_25 : t.c = 25) : 
  t.a = shorter_leg_length := by
  sorry


end NUMINAMATH_CALUDE_shorter_leg_is_15_l4160_416022


namespace NUMINAMATH_CALUDE_soccer_ball_cost_l4160_416067

theorem soccer_ball_cost (football_cost soccer_cost : ℚ) : 
  (3 * football_cost + soccer_cost = 155) →
  (2 * football_cost + 3 * soccer_cost = 220) →
  soccer_cost = 50 := by
sorry

end NUMINAMATH_CALUDE_soccer_ball_cost_l4160_416067


namespace NUMINAMATH_CALUDE_min_value_problem_l4160_416005

theorem min_value_problem (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + b^2 = 4) (h2 : c * d = 1) :
  (a^2 * c^2 + b^2 * d^2) * (b^2 * c^2 + a^2 * d^2) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l4160_416005


namespace NUMINAMATH_CALUDE_nth_equation_l4160_416047

/-- The product of consecutive integers from n+1 to 2n -/
def leftSide (n : ℕ) : ℕ := Finset.prod (Finset.range n) (λ i => n + i + 1)

/-- The product of odd numbers from 1 to 2n-1 -/
def oddProduct (n : ℕ) : ℕ := Finset.prod (Finset.range n) (λ i => 2 * i + 1)

/-- The nth equation in the pattern -/
theorem nth_equation (n : ℕ) : leftSide n = 2^n * oddProduct n := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l4160_416047


namespace NUMINAMATH_CALUDE_polynomial_remainder_l4160_416010

theorem polynomial_remainder (s : ℝ) : (s^10 + 1) % (s - 2) = 1025 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l4160_416010


namespace NUMINAMATH_CALUDE_square_roots_sum_l4160_416028

theorem square_roots_sum (x y : ℝ) (hx : x^2 = 16) (hy : y^2 = 9) :
  x^2 + y^2 + x - 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_sum_l4160_416028


namespace NUMINAMATH_CALUDE_valuation_problems_l4160_416077

/-- The p-adic valuation of an integer n -/
noncomputable def padic_valuation (p : ℕ) (n : ℤ) : ℕ := sorry

theorem valuation_problems :
  (padic_valuation 3 (2^27 + 1) = 4) ∧
  (padic_valuation 7 (161^14 - 112^14) = 16) ∧
  (padic_valuation 2 (7^20 + 1) = 1) ∧
  (padic_valuation 2 (17^48 - 5^48) = 6) := by sorry

end NUMINAMATH_CALUDE_valuation_problems_l4160_416077


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l4160_416089

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 240) : x + y = 680 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l4160_416089


namespace NUMINAMATH_CALUDE_trig_identity_l4160_416055

open Real

theorem trig_identity (α : ℝ) : 
  (1 / sin (-α) - sin (π + α)) / (1 / cos (3*π - α) + cos (2*π - α)) = 1 / tan α^3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l4160_416055


namespace NUMINAMATH_CALUDE_min_value_of_f_l4160_416016

-- Define the function f
def f (x : ℝ) : ℝ := 3*x - 4*x^3

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧
  (∀ y ∈ Set.Icc 0 1, f y ≥ f x) ∧
  f x = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l4160_416016


namespace NUMINAMATH_CALUDE_line_plane_relationship_l4160_416015

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpendicular : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the intersection relation between a line and a plane
variable (intersects : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_relationship 
  (a b : Line) (α : Plane)
  (h1 : perpendicular a b)
  (h2 : parallel_line_plane a α) :
  intersects b α ∨ subset_line_plane b α ∨ parallel_line_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l4160_416015


namespace NUMINAMATH_CALUDE_laptop_price_l4160_416037

theorem laptop_price : ∃ (S : ℝ), S = 733 ∧ 
  (0.80 * S - 120 = 0.65 * S - 10) := by
  sorry

end NUMINAMATH_CALUDE_laptop_price_l4160_416037


namespace NUMINAMATH_CALUDE_flour_for_cake_l4160_416079

theorem flour_for_cake (total_flour : ℚ) (scoop_size : ℚ) (num_scoops : ℕ) : 
  total_flour = 8 →
  scoop_size = 1/4 →
  num_scoops = 8 →
  total_flour - (↑num_scoops * scoop_size) = 6 :=
by sorry

end NUMINAMATH_CALUDE_flour_for_cake_l4160_416079


namespace NUMINAMATH_CALUDE_sum_reciprocals_of_three_integers_l4160_416062

theorem sum_reciprocals_of_three_integers (a b c : ℕ+) :
  a < b ∧ b < c ∧ a + b + c = 11 →
  (1 : ℚ) / a + 1 / b + 1 / c = 31 / 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_of_three_integers_l4160_416062


namespace NUMINAMATH_CALUDE_length_width_difference_approx_l4160_416024

/-- Represents a rectangular field -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  length_gt_width : length > width
  area_eq : area = length * width

/-- The difference between length and width of a rectangular field -/
def length_width_difference (field : RectangularField) : ℝ :=
  field.length - field.width

theorem length_width_difference_approx 
  (field : RectangularField) 
  (h_area : field.area = 171) 
  (h_length : field.length = 19.13) : 
  ∃ ε > 0, |length_width_difference field - 10.19| < ε :=
sorry

end NUMINAMATH_CALUDE_length_width_difference_approx_l4160_416024
