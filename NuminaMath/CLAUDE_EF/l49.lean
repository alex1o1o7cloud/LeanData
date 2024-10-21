import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_correct_l49_4940

-- Define the function f as noncomputable due to its dependence on Real.sqrt
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 3) / Real.sqrt (x^2 - 5*x + 6)

-- Define the domain of f
def domain_of_f : Set ℝ := {x | x ≤ 2 ∨ x ≥ 3}

-- Theorem statement
theorem domain_of_f_is_correct :
  {x : ℝ | IsRegular (f x)} = domain_of_f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_correct_l49_4940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l49_4967

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2) + Real.sqrt (x^2 - 1)

theorem range_of_f :
  Set.range f = {0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l49_4967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_price_decrease_l49_4953

/-- Calculates the percent decrease between two prices -/
noncomputable def percent_decrease (original_price sale_price : ℝ) : ℝ :=
  (original_price - sale_price) / original_price * 100

theorem trouser_price_decrease : percent_decrease 100 10 = 90 := by
  -- Unfold the definition of percent_decrease
  unfold percent_decrease
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Perform the calculation
  norm_num

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_price_decrease_l49_4953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l49_4937

-- Define the geometric sequence and its sum
def S (n : ℕ) (t : ℝ) : ℝ := 3^n + t

-- Define the sequence terms
def a (n : ℕ) (t : ℝ) : ℝ :=
  match n with
  | 0 => 0  -- Add a case for 0
  | 1 => S 1 t
  | n+1 => S (n+1) t - S n t

-- State the theorem
theorem geometric_sequence_sum (t : ℝ) :
  t + a 3 t = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l49_4937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_b_find_c_range_l49_4972

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + x

-- Define the property of being an odd function
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Theorem 1
theorem find_a_b :
  ∀ a b : ℝ,
  is_odd_function (f a b) →
  f a b 1 - f a b (-1) = 4 →
  a = 2 ∧ b = 0 := by sorry

-- Define the simplified function f after finding a and b
def f_simplified (x : ℝ) : ℝ := 2 * x^3 + x

-- Theorem 2
theorem find_c_range :
  ∀ c : ℝ,
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f_simplified x < c^2 - 9*c) →
  c ∈ Set.Ioi 10 ∪ Set.Iio (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_b_find_c_range_l49_4972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_point_exists_l49_4993

/-- Two lines in a plane -/
structure IntersectingLines where
  l₁ : Set (ℝ × ℝ)
  l₂ : Set (ℝ × ℝ)
  intersection : Set (ℝ × ℝ)
  h_intersect : intersection.Nonempty
  h_unique : ∀ p ∈ intersection, ∀ q ∈ intersection, p = q

/-- A point moving with constant speed along a line -/
structure MovingPoint where
  position : ℝ → ℝ × ℝ
  speed : ℝ
  line : Set (ℝ × ℝ)
  h_on_line : ∀ t, position t ∈ line
  h_constant_speed : ∀ t₁ t₂, dist (position t₁) (position t₂) = speed * |t₁ - t₂|

/-- The main theorem -/
theorem equal_distance_point_exists
  (lines : IntersectingLines)
  (P : MovingPoint)
  (Q : MovingPoint)
  (h_lines_P : P.line = lines.l₁)
  (h_lines_Q : Q.line = lines.l₂)
  (h_speed : P.speed = Q.speed) :
  ∃ A : ℝ × ℝ, ∀ t : ℝ, dist A (P.position t) = dist A (Q.position t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_point_exists_l49_4993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l49_4949

/-- Represents a convex quadrilateral ABCD with given side lengths and an angle -/
structure Quadrilateral :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (angleABC : ℝ)

/-- The area of triangle ABC given side lengths AB, BC and angle ABC -/
noncomputable def areaABC (q : Quadrilateral) : ℝ :=
  1/2 * q.AB * q.BC * Real.sin q.angleABC

/-- The area of triangle CAD -/
noncomputable def areaCAD (q : Quadrilateral) : ℝ := sorry

/-- The theorem stating the area of the quadrilateral ABCD -/
theorem quadrilateral_area (q : Quadrilateral) 
  (h1 : q.AB = 5)
  (h2 : q.BC = 7)
  (h3 : q.CD = 15)
  (h4 : q.AD = 13)
  (h5 : q.angleABC = π/3) :
  areaABC q + areaCAD q = (35 * Real.sqrt 3) / 4 + areaCAD q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l49_4949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_values_l49_4955

-- Define the ellipse equation
noncomputable def is_ellipse (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m + y^2 / 4 = 1

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

-- Theorem statement
theorem ellipse_eccentricity_m_values :
  ∀ m : ℝ, m > 0 →
  (∃ a b : ℝ, a > b ∧ b > 0 ∧
    (∀ x y : ℝ, is_ellipse m x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    eccentricity a b = 2) →
  m = 3 ∨ m = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_values_l49_4955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l49_4929

theorem trig_expression_simplification (B : ℝ) (h : Real.cos B ≠ 0 ∧ Real.sin B ≠ 0) :
  (1 - (Real.cos B / Real.sin B) + (1 / Real.sin B)) * (1 + (Real.sin B / Real.cos B) - (1 / Real.cos B)) = -2 * (Real.cos (2 * B) / Real.sin (2 * B)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l49_4929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subtriangle_perimeter_l49_4917

/-- Represents an isosceles triangle with given base and height -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

/-- Calculates the perimeter of a subtriangle when the isosceles triangle is divided into 6 equal parts -/
noncomputable def subtriangle_perimeter (triangle : IsoscelesTriangle) (k : ℝ) : ℝ :=
  2 + (triangle.height^2 + k^2).sqrt + (triangle.height^2 + (k+2)^2).sqrt

/-- Theorem stating the maximum perimeter of subtriangles -/
theorem max_subtriangle_perimeter (triangle : IsoscelesTriangle) 
  (h_base : triangle.base = 12)
  (h_height : triangle.height = 15) :
  ∃ (max_perimeter : ℝ), 
    (∀ k, 0 ≤ k ∧ k ≤ 10 → subtriangle_perimeter triangle k ≤ max_perimeter) ∧
    (abs (max_perimeter - 34.36) < 0.01) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subtriangle_perimeter_l49_4917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_games_per_month_l49_4963

/-- Given a football season with the following properties:
  - The season lasts for 17.0 months
  - A total of 5491 games are played in the season
  - Each month has the same number of games
  Prove that the number of games played in one month is 323 -/
theorem games_per_month (season_length : ℝ) (total_games : ℕ) (h1 : season_length = 17.0)
  (h2 : total_games = 5491) (h3 : ∃ (games_per_month : ℕ), (↑games_per_month : ℝ) * season_length = ↑total_games) :
  ∃ (games_per_month : ℕ), games_per_month = 323 ∧ (↑games_per_month : ℝ) * season_length = ↑total_games :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_games_per_month_l49_4963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hostel_cost_23_days_l49_4945

/-- Calculates the cost of staying at a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekWeekdayRate : ℚ := 20
  let firstWeekWeekendRate : ℚ := 25
  let additionalWeekWeekdayRate : ℚ := 15
  let additionalWeekWeekendRate : ℚ := 20
  let discountRate : ℚ := 1 / 10
  let discountThreshold : ℕ := 15

  let firstWeekCost : ℚ := 5 * firstWeekWeekdayRate + 2 * firstWeekWeekendRate
  let additionalDays : ℕ := days - 7
  let additionalWeeks : ℕ := additionalDays / 7
  let remainingDays : ℕ := additionalDays % 7
  
  let additionalWeeksCost : ℚ := ↑additionalWeeks * (5 * additionalWeekWeekdayRate + 2 * additionalWeekWeekendRate)
  let remainingDaysCost : ℚ := ↑remainingDays * additionalWeekWeekdayRate
  
  let totalCost : ℚ := firstWeekCost + additionalWeeksCost + remainingDaysCost
  
  if days ≥ discountThreshold then
    totalCost * (1 - discountRate)
  else
    totalCost

theorem hostel_cost_23_days :
  hostelCost 23 = 369 := by
  sorry

#eval hostelCost 23

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hostel_cost_23_days_l49_4945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2_value_b_n_formula_a_n_formula_geometric_sequence_inequality_l49_4956

noncomputable def sequence_a : ℕ → ℚ := sorry

noncomputable def sum_S : ℕ → ℚ := sorry

axiom a_1 : sequence_a 1 = 2

axiom sequence_property (n : ℕ) :
  n ≥ 1 → 1 / sequence_a n - 1 / sequence_a (n + 1) = 2 / (4 * sum_S n - 1)

theorem a_2_value : sequence_a 2 = 14 / 3 := by sorry

noncomputable def sequence_b (n : ℕ) : ℚ :=
  sequence_a n / (sequence_a (n + 1) - sequence_a n)

theorem b_n_formula (n : ℕ) :
  n ≥ 1 → sequence_b n = (4 * n - 1) / 4 := by sorry

theorem a_n_formula (n : ℕ) :
  n ≥ 1 → sequence_a n = (8 * n - 2) / 3 := by sorry

theorem geometric_sequence_inequality (m p r : ℕ) :
  m < p ∧ p < r ∧
  (sequence_a p)^2 = sequence_a m * sequence_a r →
  p^2 < m * r := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2_value_b_n_formula_a_n_formula_geometric_sequence_inequality_l49_4956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l49_4943

theorem vector_equation_solution :
  ∃ (u v : ℝ), 
    u = -6/7 ∧ 
    v = 5/7 ∧ 
    (⟨3, -1⟩ : ℝ × ℝ) + u • (⟨8, -6⟩ : ℝ × ℝ) = (⟨0, 1⟩ : ℝ × ℝ) + v • (⟨-3, 4⟩ : ℝ × ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l49_4943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_center_l49_4931

noncomputable def g (z : ℂ) : ℂ := ((1 + Complex.I * Real.sqrt 2) * z + (4 * Real.sqrt 2 - 10 * Complex.I)) / 3

noncomputable def w : ℂ := -2 * Real.sqrt 2 + 4/3 - (10/3) * Complex.I

theorem rotation_center : g w = w := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_center_l49_4931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l49_4980

theorem log_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 20*x
  ∀ x : ℝ, (Real.log (f x) = 3 * Real.log 10) ↔ (x = 10 + Real.sqrt 1100 ∨ x = 10 - Real.sqrt 1100) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l49_4980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l49_4939

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

-- Define the concept of a period for a real function
def is_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- State the theorem
theorem smallest_positive_period_of_f :
  ∃ T > 0, is_period T f ∧ ∀ S, 0 < S → S < T → ¬ is_period S f :=
by
  -- We claim that 2π is the smallest positive period
  use 2 * Real.pi
  
  sorry -- The proof is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l49_4939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_equals_triangle_area_l49_4960

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the length of sides
noncomputable def side_length (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the area of a triangle using Heron's formula
noncomputable def triangle_area (t : Triangle) : ℝ :=
  let a := side_length t.A t.B
  let b := side_length t.B t.C
  let c := side_length t.C t.A
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define reflection of a point across another point
def reflect_point (p center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - p.1, 2 * center.2 - p.2)

-- Theorem statement
theorem intersection_area_equals_triangle_area (t : Triangle) 
  (h1 : side_length t.A t.B = 8)
  (h2 : side_length t.B t.C = 15)
  (h3 : side_length t.C t.A = 17) :
  let H := centroid t
  let A' := reflect_point t.A H
  let B' := reflect_point t.B H
  let C' := reflect_point t.C H
  let t' : Triangle := { A := A', B := B', C := C' }
  triangle_area t = 60 ∧ triangle_area t = triangle_area t' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_equals_triangle_area_l49_4960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_monotone_increasing_interval_l49_4964

theorem cosine_monotone_increasing_interval (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, -Real.pi ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a → Real.cos x₁ < Real.cos x₂) → 
  -Real.pi < a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_monotone_increasing_interval_l49_4964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_problem_l49_4977

-- Define polynomials as functions
noncomputable def A : ℝ → ℝ → ℝ := sorry
def B : ℝ → ℝ → ℝ := fun x y ↦ 3 * x^2 * y - 5 * x * y + x + 7

-- State the theorem
theorem polynomial_problem :
  (∀ x y, A x y + B x y = 12 * x^2 * y + 2 * x * y + 5) →
  (∀ x y, A x y = 9 * x^2 * y + 7 * x * y - x - 2) ∧
  (∃ y, ∀ x, ∃ c, 2 * A x y - (A x y + 3 * B x y) = c) →
  (∃ y, y = 2 / 11 ∧ ∀ x, ∃ c, 2 * A x y - (A x y + 3 * B x y) = c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_problem_l49_4977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_rational_roots_even_coeff_l49_4909

theorem quadratic_rational_roots_even_coeff 
  (a b c : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : ∃ (x : ℚ), a * x^2 + b * x + c = 0) : 
  (∃ k ∈ ({a, b, c} : Set ℤ), Even k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_rational_roots_even_coeff_l49_4909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l49_4934

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area of the region bound by two circles and the x-axis below y=5 that is not inside either circle -/
noncomputable def area_between_circles_and_x_axis (c d : Circle) : ℝ :=
  35 - 17 * Real.pi

/-- Theorem stating the area of the region -/
theorem area_calculation (c d : Circle) 
  (hc : c.center = (3, 5) ∧ c.radius = 3)
  (hd : d.center = (10, 5) ∧ d.radius = 5) :
  area_between_circles_and_x_axis c d = 35 - 17 * Real.pi :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l49_4934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_vertices_l49_4930

-- Define the lines of the triangle
def line1 (x y : ℝ) : Prop := y = x + 1
def line2 (x y : ℝ) : Prop := y = -1/2 * x - 2
def line3 (x y : ℝ) : Prop := y = 3 * x - 9

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 25

-- Define a function to check if a point is on the circle
def is_on_circle (x y : ℝ) : Prop := circle_eq x y

-- Theorem: The circle passes through the vertices of the triangle
theorem circle_passes_through_vertices :
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (line1 x1 y1 ∧ line2 x1 y1) ∧
    (line1 x2 y2 ∧ line3 x2 y2) ∧
    (line2 x3 y3 ∧ line3 x3 y3) ∧
    is_on_circle x1 y1 ∧
    is_on_circle x2 y2 ∧
    is_on_circle x3 y3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_vertices_l49_4930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suresh_walking_speed_l49_4996

/-- Calculates the walking speed of a person given the track circumference, the other person's speed, and the time they meet. -/
noncomputable def calculate_walking_speed (track_circumference : ℝ) (other_speed : ℝ) (meeting_time : ℝ) : ℝ :=
  let other_distance := other_speed * 1000 / 60 * meeting_time
  let person_distance := track_circumference - other_distance
  person_distance / meeting_time * 60 / 1000

/-- Theorem stating that under given conditions, Suresh's walking speed is approximately 4.51 km/hr -/
theorem suresh_walking_speed :
  let track_circumference : ℝ := 726
  let wife_speed : ℝ := 3.75
  let meeting_time : ℝ := 5.28
  let suresh_speed := calculate_walking_speed track_circumference wife_speed meeting_time
  ∃ ε > 0, |suresh_speed - 4.51| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suresh_walking_speed_l49_4996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_books_borrowed_l49_4971

theorem average_books_borrowed (total_students : ℕ) 
  (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 38)
  (h2 : zero_books = 2)
  (h3 : one_book = 12)
  (h4 : two_books = 10)
  (h5 : zero_books + one_book + two_books < total_students) :
  (((0 : ℝ) * zero_books + 1 * one_book + 2 * two_books + 
   3 * (total_students - (zero_books + one_book + two_books))) / total_students) ≥ 1.947 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_books_borrowed_l49_4971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_award_recipients_l49_4947

/-- The number of people initially planned to receive the award -/
def initial_recipients : ℕ := sorry

/-- The initial total amount of the award in rubles -/
def initial_total : ℕ := sorry

/-- The final number of award recipients -/
def final_recipients : ℕ := initial_recipients + 3

/-- The amount each person would receive if the initial total were divided among the final recipients -/
def reduced_amount : ℚ := initial_total / final_recipients

/-- The amount each person actually received after the increase -/
def final_amount : ℕ := 2500

/-- The increase in the total award amount -/
def increase_amount : ℕ := 9000

theorem award_recipients :
  (initial_total / initial_recipients - reduced_amount = 400) →
  (initial_total + increase_amount = final_amount * final_recipients) →
  (final_recipients = 18) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_award_recipients_l49_4947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_plane_intersection_distance_l49_4974

/-- Represents a 3D point -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  vertices : List Point3D

/-- Represents a plane -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Find intersection points of a plane with a cube's edges -/
def findIntersections (cube : Cube) (plane : Plane) : List Point3D :=
  sorry

theorem cube_plane_intersection_distance :
  let cube : Cube := {
    vertices := [
      {x := 0, y := 0, z := 0}, {x := 0, y := 0, z := 5},
      {x := 0, y := 5, z := 0}, {x := 0, y := 5, z := 5},
      {x := 5, y := 0, z := 0}, {x := 5, y := 0, z := 5},
      {x := 5, y := 5, z := 0}, {x := 5, y := 5, z := 5}
    ]
  }
  let p : Point3D := {x := 0, y := 3, z := 0}
  let q : Point3D := {x := 2, y := 0, z := 0}
  let r : Point3D := {x := 2, y := 5, z := 5}
  let plane : Plane := {
    a := -15,
    b := 10,
    c := 4,
    d := 30
  }
  let intersections := findIntersections cube plane
  ∃ (s t : Point3D), s ∈ intersections ∧ t ∈ intersections ∧ s ≠ p ∧ s ≠ q ∧ s ≠ r ∧ t ≠ p ∧ t ≠ q ∧ t ≠ r ∧ distance s t = 3 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_plane_intersection_distance_l49_4974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l49_4926

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.cos x + Real.sqrt (1 + Real.sin x) + Real.sqrt (1 - Real.sin x)

noncomputable def t (x : ℝ) : ℝ := Real.sqrt (1 + Real.sin x) + Real.sqrt (1 - Real.sin x)

noncomputable def g (a : ℝ) (t : ℝ) : ℝ := a / 2 * t^2 + t - a

def I : Set ℝ := Set.Icc (-Real.pi/2) (Real.pi/2)

theorem f_properties (a : ℝ) (h : a < 0) :
  (∀ x ∈ I, t x ∈ Set.Icc (Real.sqrt 2) 2) ∧
  (∀ x ∈ I, f a x = g a (t x)) ∧
  (∀ x ∈ I, f a x ≤ 
    if -1/2 < a then a + 2
    else if a ≤ -Real.sqrt 2 / 2 then -1 / (2*a) - a
    else Real.sqrt 2) ∧
  (a = -1 → 
    ∃ m : ℝ, m ≥ 1/2 ∧ 
    (∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → |f a x₁ - f a x₂| ≤ m) ∧
    (∀ m' : ℝ, (∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → |f a x₁ - f a x₂| ≤ m') → m' ≥ 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l49_4926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_hydroxide_formation_l49_4990

/-- Represents a chemical compound -/
structure Compound where
  name : String
  formula : String
  deriving Inhabited, BEq

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List (Compound × ℕ)
  products : List (Compound × ℕ)

def silver_nitrate : Compound := ⟨"Silver nitrate", "AgNO₃"⟩
def sodium_hydroxide : Compound := ⟨"Sodium hydroxide", "NaOH"⟩
def silver_hydroxide : Compound := ⟨"Silver hydroxide", "AgOH"⟩
def sodium_nitrate : Compound := ⟨"Sodium nitrate", "NaNO₃"⟩

def reaction : Reaction :=
  ⟨[(silver_nitrate, 1), (sodium_hydroxide, 1)],
   [(silver_hydroxide, 1), (sodium_nitrate, 1)]⟩

def available_silver_nitrate : ℕ := 2
def available_sodium_hydroxide : ℕ := 2

/-- 
  Theorem: Given 2 moles of Silver nitrate and 2 moles of Sodium hydroxide,
  and the balanced chemical equation AgNO₃ + NaOH → AgOH + NaNO₃,
  the number of moles of Silver Hydroxide formed is 2.
-/
theorem silver_hydroxide_formation :
  ∃ (n : ℕ), n = 2 ∧
  n = min available_silver_nitrate available_sodium_hydroxide ∧
  (reaction.products.filter (λ p => p.1 == silver_hydroxide)).head!.2 * n =
    (reaction.products.filter (λ p => p.1 == silver_hydroxide)).head!.2 * 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_hydroxide_formation_l49_4990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_legs_is_392_l49_4989

/-- Calculates the remaining legs of furniture after damage -/
def remaining_legs (
  chair_count : ℕ)
  (chair_legs : ℕ)
  (chair_damage_rate : ℚ)
  (round_table_count : ℕ)
  (round_table_legs : ℕ)
  (round_table_damage_rate : ℚ)
  (cabinet_count : ℕ)
  (cabinet_legs : ℕ)
  (cabinet_damage_rate : ℚ)
  (long_table_count : ℕ)
  (long_table_legs : ℕ)
  (long_table_damage_rate : ℚ) : ℕ :=
  let total_legs := 
    chair_count * chair_legs +
    round_table_count * round_table_legs +
    cabinet_count * cabinet_legs +
    long_table_count * long_table_legs
  let damaged_legs := 
    (chair_count * chair_legs * chair_damage_rate).floor +
    (round_table_count * round_table_legs * round_table_damage_rate).floor +
    (cabinet_count * cabinet_legs * cabinet_damage_rate).floor +
    (long_table_count * long_table_legs * long_table_damage_rate).floor
  total_legs - damaged_legs.toNat

/-- Theorem: The remaining legs of furniture after damage is 392 -/
theorem remaining_legs_is_392 : 
  remaining_legs 80 5 (1/2) 20 3 (3/10) 30 4 (1/5) 10 6 (1/10) = 392 := by
  sorry

#eval remaining_legs 80 5 (1/2) 20 3 (3/10) 30 4 (1/5) 10 6 (1/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_legs_is_392_l49_4989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_of_min_f_eq_neg_three_l49_4918

/-- The function f(x) = -x^2 + 2x - 3 -/
noncomputable def f (x : ℝ) : ℝ := -x^2 + 2*x - 3

/-- The minimum value of f(x) on the interval [2a - 1, 2] as a function of a -/
noncomputable def g (a : ℝ) : ℝ := 
  if a ≤ 1/2 
  then f (2*a - 1)
  else -3

/-- Theorem: The maximum value of the minimum values of f(x) on [2a - 1, 2] is -3 -/
theorem max_of_min_f_eq_neg_three : 
  ∀ a : ℝ, g a ≤ -3 ∧ ∃ a₀ : ℝ, g a₀ = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_of_min_f_eq_neg_three_l49_4918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_value_l49_4951

noncomputable def point_A : ℝ := 1
noncomputable def point_B : ℝ := Real.sqrt 5

theorem point_C_value (C : ℝ) (h1 : |C - point_B| = 3 * |point_B - point_A|) :
  C = 4 * Real.sqrt 5 - 3 ∨ C = -2 * Real.sqrt 5 + 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_value_l49_4951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_equation_on_sphere_iff_satisfies_equation_l49_4965

/-- Theorem: Equation of a sphere with center (a, b, c) and radius R -/
theorem sphere_equation (a b c R : ℝ) :
  ∀ (x y z : ℝ), dist (x, y, z) (a, b, c) = R ↔ (x - a)^2 + (y - b)^2 + (z - c)^2 = R^2 :=
by sorry

/-- Definition: A point is on the sphere if its distance from the center is equal to the radius -/
def on_sphere (a b c R : ℝ) (x y z : ℝ) : Prop :=
  dist (x, y, z) (a, b, c) = R

/-- Theorem: A point is on the sphere if and only if it satisfies the sphere equation -/
theorem on_sphere_iff_satisfies_equation (a b c R : ℝ) (x y z : ℝ) :
  on_sphere a b c R x y z ↔ (x - a)^2 + (y - b)^2 + (z - c)^2 = R^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_equation_on_sphere_iff_satisfies_equation_l49_4965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_price_after_discounts_l49_4925

theorem hat_price_after_discounts (original_price : ℝ) (first_discount_rate : ℝ) (second_discount_rate : ℝ) : 
  original_price = 15 →
  first_discount_rate = 0.2 →
  second_discount_rate = 0.25 →
  original_price * (1 - first_discount_rate) * (1 - second_discount_rate) = 9 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  -- The proof is completed automatically by norm_num
  done

#check hat_price_after_discounts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_price_after_discounts_l49_4925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_hyperbola_l49_4901

/-- The mathematical constant e --/
noncomputable def e : ℝ := Real.exp 1

/-- Definition of x coordinate --/
noncomputable def x (t : ℝ) : ℝ := Real.exp t + Real.exp (-t)

/-- Definition of y coordinate --/
noncomputable def y (t : ℝ) : ℝ := 3 * (Real.exp t - Real.exp (-t))

/-- Theorem: The points (x(t), y(t)) lie on a hyperbola --/
theorem points_on_hyperbola (t : ℝ) : (x t)^2 / 4 - (y t)^2 / 36 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_hyperbola_l49_4901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l49_4991

/-- Represents the time needed to complete a work -/
structure WorkTime where
  days : ℚ
  work : ℚ
  complete : work = 1

/-- Represents a person's work rate -/
def WorkRate (wt : WorkTime) : ℚ := wt.work / wt.days

theorem work_completion_time 
  (total_work : ℚ)
  (y_time z_time : WorkTime)
  (combined_work : ℚ)
  (z_remaining_time : ℚ)
  (h1 : y_time.days = 20)
  (h2 : z_time.days = 30)
  (h3 : combined_work = 2 * (total_work / 5 + y_time.work / y_time.days + z_time.work / z_time.days))
  (h4 : z_remaining_time = 13)
  (h5 : combined_work + z_remaining_time * (z_time.work / z_time.days) = total_work)
  : ∃ x_time : WorkTime, x_time.days = 5 ∧ x_time.work = total_work := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l49_4991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_4_50_l49_4903

-- Define the clock parameters
def hours_in_clock : ℕ := 12
def minutes_in_hour : ℕ := 60
def degrees_in_circle : ℕ := 360

-- Define the time
def hour : ℕ := 4
def minute : ℕ := 50

-- Define functions for hand positions
noncomputable def hour_hand_position (h : ℕ) (m : ℕ) : ℝ :=
  (h % hours_in_clock : ℝ) * (degrees_in_circle / hours_in_clock : ℝ) +
  (m : ℝ) * (degrees_in_circle / hours_in_clock : ℝ) / minutes_in_hour

noncomputable def minute_hand_position (m : ℕ) : ℝ :=
  (m : ℝ) * (degrees_in_circle / minutes_in_hour : ℝ)

-- Define the angle between hands
noncomputable def angle_between_hands (h : ℕ) (m : ℕ) : ℝ :=
  let angle := abs (minute_hand_position m - hour_hand_position h m)
  min angle (degrees_in_circle - angle)

-- Theorem statement
theorem clock_angle_at_4_50 :
  angle_between_hands hour minute = 155 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_4_50_l49_4903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_chord_length_l49_4968

/-- A hexagon inscribed in a circle -/
structure InscribedHexagon where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The side lengths of the hexagon -/
  sides : Fin 6 → ℝ
  /-- Ensures that three consecutive sides have length 4 -/
  short_sides : ∃ i : Fin 6, (sides i = 4) ∧ (sides (i + 1) = 4) ∧ (sides (i + 2) = 4)
  /-- Ensures that three consecutive sides have length 6 -/
  long_sides : ∃ i : Fin 6, (sides i = 6) ∧ (sides (i + 1) = 6) ∧ (sides (i + 2) = 6)
  /-- Ensures that the hexagon is inscribed in the circle -/
  inscribed : ∀ i : Fin 6, 2 * radius * Real.sin (π / 3) = sides i

/-- The chord that divides the hexagon into two trapezoids -/
def dividing_chord (h : InscribedHexagon) : ℝ := 2 * h.radius

/-- The theorem stating that the dividing chord has length 10 -/
theorem dividing_chord_length (h : InscribedHexagon) : 
  dividing_chord h = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_chord_length_l49_4968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_de_moivre_for_all_n_up_to_500_l49_4933

noncomputable def complex_exp (t : ℝ) : ℂ := Complex.exp (Complex.I * t)

theorem de_moivre_for_all_n_up_to_500 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 500 →
    ∀ t : ℝ, (complex_exp (-t))^n = complex_exp (-n * t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_de_moivre_for_all_n_up_to_500_l49_4933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_to_coefficients_l49_4970

theorem quadratic_roots_to_coefficients (a : ℝ) :
  ∃ k c : ℝ, (2 * 7^2 + k * 7 + c = 0 ∧ 2 * a^2 + k * a + c = 0) → 
    (k = -2*a - 14 ∧ c = 14*a) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_to_coefficients_l49_4970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_effect_l49_4919

-- Define the original price of 110 mangoes
def original_price_110 : ℚ := 366.67

-- Define the price reduction percentage
def price_reduction : ℚ := 0.10

-- Define Mr. John's spending amount
def john_spending : ℚ := 360

-- Function to calculate the number of mangoes
def calculate_mangoes (price_per_mango : ℚ) : ℕ :=
  (john_spending / price_per_mango).floor.toNat

-- Theorem statement
theorem price_reduction_effect :
  let original_price_per_mango : ℚ := original_price_110 / 110
  let new_price_per_mango : ℚ := original_price_per_mango * (1 - price_reduction)
  let original_mangoes : ℕ := calculate_mangoes original_price_per_mango
  let new_mangoes : ℕ := calculate_mangoes new_price_per_mango
  new_mangoes - original_mangoes = 12 := by
  sorry

#eval calculate_mangoes (366.67 / 110)
#eval calculate_mangoes ((366.67 / 110) * (1 - 0.10))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_effect_l49_4919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_P_representation_l49_4979

def U : Set ℕ := {1, 2, 3, 4, 5}

def CU_P : Set ℕ := {4, 5}

def P : Set ℕ := {x : ℕ | x ∈ U ∧ x ∉ CU_P}

theorem correct_P_representation : P = {x : ℕ | x ∈ U ∧ x < 4} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_P_representation_l49_4979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_150_equals_149_75_l49_4916

/-- The product of terms from 1 to n, where each term is (1 - 1/k) -/
def product (n : ℕ) : ℚ :=
  (List.range (n-1)).foldl (λ acc k => acc * (1 - 1 / (k+2))) 1

/-- The main theorem stating the value of the product for n = 150 -/
theorem product_150_equals_149_75 :
  2 * product 150 = 149 / 75 := by
  sorry

#eval 2 * product 150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_150_equals_149_75_l49_4916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_gp_l49_4986

/-- Represents a geometric progression -/
structure GeometricProgression where
  a : ℕ → ℝ
  q : ℝ
  first_term : a 1 ≠ 0
  progression : ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of first n terms of a geometric progression -/
noncomputable def sum_of_gp (gp : GeometricProgression) (n : ℕ) : ℝ :=
  gp.a 1 * (1 - gp.q^n) / (1 - gp.q)

theorem sum_of_specific_gp :
  ∀ gp : GeometricProgression,
  (∀ n : ℕ, gp.a (n + 1) > gp.a n) →
  gp.a 1 + gp.a 3 = 5 →
  gp.a 1 * gp.a 3 = 4 →
  sum_of_gp gp 6 = 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_gp_l49_4986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_l49_4928

-- Define the class composition
def junior_percentage : ℚ := 1/5
def senior_percentage : ℚ := 4/5

-- Define the average scores
def overall_average : ℚ := 78
def senior_average : ℚ := 75

-- Define the theorem
theorem junior_score (total_students : ℕ) (h1 : total_students > 0) :
  let junior_count : ℕ := (junior_percentage * total_students).num.toNat
  let senior_count : ℕ := total_students - junior_count
  let total_score : ℚ := overall_average * total_students
  let senior_total_score : ℚ := senior_average * senior_count
  let junior_total_score : ℚ := total_score - senior_total_score
  junior_total_score / junior_count = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_l49_4928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l49_4935

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y + 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 6 = 0

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem min_distance_between_circles :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ →
  distance x₁ y₁ x₂ y₂ ≥ 3 - Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l49_4935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_A_value_l49_4982

/-- The speed of person B in kilometers per hour -/
noncomputable def speed_B : ℝ := 7.555555555555555

/-- The time in hours that B walks before overtaking A -/
noncomputable def time_B : ℝ := 1.8

/-- The time in hours that A walks before being overtaken by B -/
noncomputable def time_A : ℝ := 2.3

/-- The speed of person A in kilometers per hour -/
noncomputable def speed_A : ℝ := (speed_B * time_B) / time_A

theorem speed_A_value : 
  ∃ ε > 0, |speed_A - 5.913| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_A_value_l49_4982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_parabola_intersection_l49_4985

noncomputable section

-- Define the ellipse and parabola
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the foci and point P
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)
def P : ℝ × ℝ := (2/3, 2/3 * Real.sqrt 6)

-- Theorem statement
theorem ellipse_and_parabola_intersection 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hP : ellipse a b P.1 P.2 ∧ parabola P.1 P.2) 
  (hPF2 : Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 5/3) :
  (∃ (t : ℝ), 
    ellipse 2 (Real.sqrt 3) P.1 P.2 = ellipse a b P.1 P.2 ∧
    (0 < t ∧ t < 1/4 ↔ 
      ∃ (M N : ℝ × ℝ), 
        ellipse a b M.1 M.2 ∧ 
        ellipse a b N.1 N.2 ∧ 
        (∃ (m : ℝ), M.1 = m * M.2 + 1 ∧ N.1 = m * N.2 + 1) ∧
        Real.sqrt ((t - M.1)^2 + M.2^2) = Real.sqrt ((t - N.1)^2 + N.2^2))) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_parabola_intersection_l49_4985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_parabola_point_l49_4973

/-- The distance from the origin to a point on a parabola -/
theorem distance_origin_to_parabola_point (P : ℝ × ℝ) :
  P.1^2 = 4 * P.2 →  -- P is on the parabola x^2 = 4y
  Real.sqrt ((P.1 - 0)^2 + (P.2 - 1)^2) = 5 →  -- |PF| = 5, where F is (0, 1)
  Real.sqrt (P.1^2 + P.2^2) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_parabola_point_l49_4973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_1003_div_6_l49_4988

noncomputable section

/-- The curve C defined as y = 2006x³ - 12070102x² + ax + b -/
def C (a b : ℝ) (x : ℝ) : ℝ := 2006 * x^3 - 12070102 * x^2 + a * x + b

/-- The derivative of C with respect to x -/
def C' (a : ℝ) (x : ℝ) : ℝ := 6018 * x^2 - 24140204 * x + a

/-- The tangent line of C at x = 2006 -/
def tangent_line (a b : ℝ) (x : ℝ) : ℝ := 
  (C' a 2006) * (x - 2006) + C a b 2006

/-- The area between C and its tangent line at x = 2006 -/
noncomputable def area_between_curve_and_tangent (a b : ℝ) : ℝ :=
  ∫ x in Set.Icc 2005 2006, C a b x - tangent_line a b x

theorem area_is_1003_div_6 (a b : ℝ) :
  area_between_curve_and_tangent a b = 1003 / 6 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_1003_div_6_l49_4988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_walk_distance_l49_4978

/-- A regular octagon with side length 3 km -/
structure RegularOctagon where
  side_length : ℝ
  regular : side_length = 3

/-- Walking 12 km along the perimeter of the octagon -/
def walk_distance : ℝ := 12

/-- Function to calculate the end point after walking a given distance -/
noncomputable def walk_perimeter (o : RegularOctagon) (d : ℝ) : ℝ × ℝ := sorry

/-- The theorem to prove -/
theorem octagon_walk_distance (octagon : RegularOctagon) :
  let end_point := walk_perimeter octagon walk_distance
  ‖end_point‖ = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_walk_distance_l49_4978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_initial_speed_l49_4999

/-- A journey with two phases of driving at different speeds -/
structure Journey where
  total_distance : ℝ
  total_time : ℝ
  initial_time : ℝ
  final_speed : ℝ

/-- Calculate the initial speed of the journey -/
noncomputable def initial_speed (j : Journey) : ℝ :=
  let remaining_time := j.total_time - j.initial_time
  let remaining_distance := j.final_speed * (remaining_time / 60)
  let initial_distance := j.total_distance - remaining_distance
  (initial_distance / j.initial_time) * 60

/-- Theorem stating that for the given journey parameters, the initial speed is 60 mph -/
theorem journey_initial_speed :
  let j : Journey := {
    total_distance := 120,
    total_time := 90,
    initial_time := 30,
    final_speed := 90
  }
  initial_speed j = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_initial_speed_l49_4999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_sum_l49_4941

-- Define the function f(x) = x^2 |x|
def f (x : ℝ) : ℝ := x^2 * abs x

-- State the theorem
theorem inverse_f_sum : 
  ∃ (y₁ y₂ : ℝ), f y₁ = 9 ∧ f y₂ = -27 ∧ y₁ + y₂ = (9 : ℝ) ^ (1/3) - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_sum_l49_4941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l49_4992

noncomputable def is_geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c

noncomputable def eccentricity (m : ℝ) : ℝ := 
  if m > 0 then 
    Real.sqrt (1 - 1/m)
  else
    Real.sqrt (1 + 1/(-m))

theorem conic_section_eccentricity (m : ℝ) :
  is_geometric_sequence 4 m 9 →
  (eccentricity m = Real.sqrt 30 / 6 ∨ eccentricity m = Real.sqrt 7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l49_4992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_transformed_function_l49_4927

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2^x)
def domain_f_2_pow_x : Set ℝ := Set.Icc 1 2

-- State the theorem
theorem domain_of_transformed_function 
  (h : ∀ x ∈ domain_f_2_pow_x, f (2^x) ∈ Set.range f) :
  {x : ℝ | f (x + 1) / (x - 1) ∈ Set.range f ∧ x ≠ 1} = Set.Ioo 1 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_transformed_function_l49_4927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_intersecting_cubes_A_is_sufficient_A_is_necessary_l49_4905

/-- The minimum number of unit cubes needed to intersect all other unit cubes in an n x n x n cube -/
def A (n : ℕ) : ℕ :=
  if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2

/-- Function to get the i-th selected unit cube (implementation details omitted) -/
def ith_selected_cube (n : ℕ) (i : ℕ) : ℕ × ℕ × ℕ :=
  sorry

/-- Theorem stating the minimum number of unit cubes needed to intersect all other unit cubes in an n x n x n cube -/
theorem minimum_intersecting_cubes (n : ℕ) (h : n > 0) :
  A n = if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2 :=
by sorry

/-- Theorem stating that A n is always sufficient to intersect all unit cubes -/
theorem A_is_sufficient (n : ℕ) (h : n > 0) :
  ∀ (x y z : ℕ), x ≤ n ∧ y ≤ n ∧ z ≤ n →
  ∃ (a b c : ℕ), (a ≤ n ∧ b ≤ n ∧ c ≤ n) ∧
  ((a = x ∧ b = y) ∨ (a = x ∧ c = z) ∨ (b = y ∧ c = z)) ∧
  (∃ (i : ℕ), i < A n ∧ (a, b, c) = ith_selected_cube n i) :=
by sorry

/-- Theorem stating that any number less than A n is not sufficient to intersect all unit cubes -/
theorem A_is_necessary (n : ℕ) (h : n > 0) :
  ∀ (k : ℕ), k < A n →
  ∃ (x y z : ℕ), x ≤ n ∧ y ≤ n ∧ z ≤ n ∧
  ∀ (a b c : ℕ), (a ≤ n ∧ b ≤ n ∧ c ≤ n) →
  ((a = x ∧ b = y) ∨ (a = x ∧ c = z) ∨ (b = y ∧ c = z)) →
  (∀ (i : ℕ), i < k → (a, b, c) ≠ ith_selected_cube n i) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_intersecting_cubes_A_is_sufficient_A_is_necessary_l49_4905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_square_and_cube_l49_4913

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → 
    (∃ (y : ℕ), 4 * x = y^2) → 
    (∃ (z : ℕ), 5 * x = z^3) → 
    x ≥ n) ∧
  n = 50 := by
  sorry

#check smallest_n_square_and_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_square_and_cube_l49_4913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l49_4976

noncomputable def train_length : ℝ := 250
noncomputable def bridge_length : ℝ := 300
noncomputable def train_speed_kmph : ℝ := 36

noncomputable def total_distance : ℝ := train_length + bridge_length

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (5/18)

noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

noncomputable def time_to_cross : ℝ := total_distance / train_speed_mps

theorem train_crossing_time :
  time_to_cross = 55 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l49_4976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_l49_4904

/-- Predicate for an isosceles right triangle -/
def IsoscelesRightTriangle (A B C : ℝ × ℝ) : Prop := sorry

/-- Function to calculate the hypotenuse of a triangle -/
def Hypotenuse (A B C : ℝ × ℝ) : ℝ := sorry

/-- Function to calculate the area of a triangle -/
def Area (A B C : ℝ × ℝ) : ℝ := sorry

/-- An isosceles right triangle with hypotenuse 6√2 has an area of 18 square units. -/
theorem isosceles_right_triangle_area : 
  ∀ (A B C : ℝ × ℝ),
  IsoscelesRightTriangle A B C →
  Hypotenuse A B C = 6 * Real.sqrt 2 →
  Area A B C = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_l49_4904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l49_4906

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_difference :
  let seq := arithmetic_sequence (-2) 7
  |seq 3010 - seq 3000| = 70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l49_4906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_HML_l49_4922

/-- Triangle ABC with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The orthocenter of a triangle -/
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- The incenter of a triangle -/
noncomputable def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The area of a triangle given three points -/
noncomputable def area (p q r : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of triangle HML is 21 for the given triangle -/
theorem area_of_HML (t : Triangle) 
  (h : t.a = 13 ∧ t.b = 14 ∧ t.c = 15) : 
  area (orthocenter t) (centroid t) (incenter t) = 21 := by
  sorry

#check area_of_HML

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_HML_l49_4922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_partial_x_mixed_partial_yx_l49_4911

open Real

noncomputable section

variables (x y : ℝ)

noncomputable def z : ℝ → ℝ → ℝ := sorry

-- Assumptions
axiom z_diff_x : 
  deriv (fun x => z x y) x = z x y / (z x y ^ 2 - 1)

axiom z_diff_y : 
  deriv (fun y => z x y) y = -z x y / (y * (z x y ^ 2 - 1))

-- Theorems to prove
theorem second_partial_x : 
  deriv (fun x => deriv (fun x => z x y) x) x = 
    z x y * (z x y ^ 4 - z x y ^ 2 + 2) / (x ^ 2 * (z x y ^ 2 - 1) ^ 3) :=
sorry

theorem mixed_partial_yx : 
  deriv (fun y => deriv (fun x => z x y) x) y = 
    -(z x y ^ 2 + 1) * z x y / (y * x * (z x y ^ 2 - 1) ^ 3) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_partial_x_mixed_partial_yx_l49_4911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_unknown_side_l49_4959

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  knownSide : ℝ
  distance : ℝ
  area : ℝ

/-- Calculates the length of the unknown parallel side of a trapezium -/
noncomputable def unknownSide (t : Trapezium) : ℝ :=
  2 * t.area / t.distance - t.knownSide

theorem trapezium_unknown_side (t : Trapezium) 
  (h1 : t.knownSide = 20)
  (h2 : t.distance = 20)
  (h3 : t.area = 380) :
  unknownSide t = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_unknown_side_l49_4959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_percentage_40_percent_l49_4969

noncomputable def circle_area (radius : ℝ) : ℝ := Real.pi * radius^2

noncomputable def design_black_percentage (n : ℕ) (initial_radius : ℝ) (increment : ℝ) : ℝ :=
  let total_area := circle_area (initial_radius + (n.pred * increment))
  let black_area := Finset.sum (Finset.range ((n + 1) / 2)) (λ i =>
    circle_area (initial_radius + ((2*i + 1) * increment)) - 
    circle_area (initial_radius + (2*i * increment)))
  (black_area / total_area) * 100

theorem black_percentage_40_percent :
  design_black_percentage 5 3 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_percentage_40_percent_l49_4969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l49_4910

/-- Given an angle θ with vertex at the origin, initial side on the positive x-axis,
    and terminal side on the line y = -√3x, prove that sin 2θ = -√3/2 -/
theorem sin_double_angle_special (θ : ℝ) (h1 : Real.tan θ = -Real.sqrt 3) :
  Real.sin (2 * θ) = -(Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l49_4910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arccos_lt_arctan_interval_l49_4924

noncomputable def a : ℝ := sorry

theorem arccos_lt_arctan_interval :
  ∃ a : ℝ, 0 < a ∧ a < 1 ∧
  (∀ x : ℝ, x ∈ Set.Ioo a 1 → Real.arccos x < Real.arctan x) ∧
  Real.arccos a = Real.arctan a ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → Real.arccos x ∈ Set.Icc 0 π) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → Real.arctan x ∈ Set.Ioo (-π/2) (π/2)) ∧
  (∀ x y : ℝ, x < y → x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → Real.arccos y < Real.arccos x) ∧
  (∀ x y : ℝ, x < y → x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → Real.arctan x < Real.arctan y) ∧
  Real.arccos 0 = π / 2 ∧
  Real.arctan 0 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arccos_lt_arctan_interval_l49_4924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_PAB_l49_4936

/-- Given a triangle ABC with area 360 and a point P such that AP = 1/4 * AB + 1/4 * AC,
    the area of triangle PAB is 90. -/
theorem area_triangle_PAB (A B C P : ℝ × ℝ) : 
  let triangle_area (X Y Z : ℝ × ℝ) := abs ((X.1 - Z.1) * (Y.2 - Z.2) - (Y.1 - Z.1) * (X.2 - Z.2)) / 2
  (triangle_area A B C = 360) →
  (∃ t : ℝ × ℝ, t = (1/4, 1/4) ∧ 
    P.1 - A.1 = t.1 * (B.1 - A.1) + t.2 * (C.1 - A.1) ∧
    P.2 - A.2 = t.1 * (B.2 - A.2) + t.2 * (C.2 - A.2)) →
  triangle_area P A B = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_PAB_l49_4936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_independent_variable_l49_4975

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x - 3)) / x

-- Define the domain of the function
def domain (x : ℝ) : Prop := x - 3 ≥ 0 ∧ x ≠ 0

-- Theorem stating the range of the independent variable
theorem range_of_independent_variable :
  ∀ x : ℝ, domain x ↔ x ≥ 3 :=
by
  intro x
  constructor
  · intro h
    cases h with
    | intro h1 h2 =>
      -- From x - 3 ≥ 0, we get x ≥ 3
      linarith
  · intro h
    constructor
    · -- If x ≥ 3, then x - 3 ≥ 0
      linarith
    · -- If x ≥ 3, then x ≠ 0
      linarith
  -- The proof is complete, so we don't need 'sorry' here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_independent_variable_l49_4975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l49_4908

/-- Represents the discount scenario in the store --/
structure DiscountScenario where
  initial_discount : ℝ
  additional_discount : ℝ
  claimed_discount : ℝ

/-- Calculates the actual discount given a DiscountScenario --/
def actual_discount (scenario : DiscountScenario) : ℝ :=
  1 - (1 - scenario.initial_discount) * (1 - scenario.additional_discount)

/-- Calculates the difference between claimed and actual discount --/
def discount_difference (scenario : DiscountScenario) : ℝ :=
  scenario.claimed_discount - actual_discount scenario

/-- Theorem stating the correct discount calculation and difference --/
theorem discount_calculation (scenario : DiscountScenario) 
  (h1 : scenario.initial_discount = 0.25)
  (h2 : scenario.additional_discount = 0.1)
  (h3 : scenario.claimed_discount = 0.35) :
  actual_discount scenario = 0.325 ∧ discount_difference scenario = 0.025 := by
  sorry

-- Remove the #eval statement as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l49_4908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l49_4914

/-- The time it takes for a train to cross a pole -/
noncomputable def time_to_cross_pole (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  train_length / train_speed_ms

/-- Theorem: A train with length 100 meters moving at 100 km/h takes about 3.6 seconds to cross a pole -/
theorem train_crossing_pole_time :
  let ε := 0.1  -- Allowable error
  let calculated_time := time_to_cross_pole 100 100
  (calculated_time - ε ≤ 3.6) ∧ (3.6 ≤ calculated_time + ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l49_4914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l49_4923

theorem log_inequality (a b c : ℝ) 
  (ha : a = Real.log 3 / Real.log π)
  (hb : b = Real.log 4 / Real.log 3)
  (hc : c = Real.log 17 / Real.log 4) : 
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l49_4923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_evaluation_l49_4983

theorem complex_expression_evaluation (n : ℕ) (hn : n = 1990) :
  let series := (Finset.range 996).sum (fun k => (-1 : ℤ)^k * (3 : ℝ)^k * (n.choose (2*k)))
  (1 / (2^n : ℝ)) * series = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_evaluation_l49_4983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_xi_function_l49_4938

noncomputable def is_xi_function (f : ℝ → ℝ) : Prop :=
  ∀ T > 0, ∃ m : ℝ, m ≠ 0 ∧ ∀ x : ℝ, f (x + T) / f x = m

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (3*x - 2)

theorem f_is_xi_function : is_xi_function f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_xi_function_l49_4938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_covered_l49_4907

/-- Represents a person with their walking speed and break times -/
structure Person where
  speed : ℝ
  breaks : List ℝ

/-- Calculates the effective walking time for a person -/
noncomputable def effectiveWalkingTime (p : Person) (totalTime : ℝ) : ℝ :=
  totalTime - (p.breaks.sum / 60)

/-- Calculates the distance covered by a person -/
noncomputable def distanceCovered (p : Person) (totalTime : ℝ) : ℝ :=
  p.speed * (effectiveWalkingTime p totalTime)

/-- Theorem stating the total distance covered by all three people -/
theorem total_distance_covered (totalTime : ℝ) (nadia hannah ethan : Person) 
  (h1 : totalTime = 2)
  (h2 : nadia.speed = 6)
  (h3 : nadia.breaks = [10, 20])
  (h4 : hannah.speed = 4)
  (h5 : hannah.breaks = [15])
  (h6 : ethan.speed = 5)
  (h7 : ethan.breaks = [10, 5, 15])
  : distanceCovered nadia totalTime + distanceCovered hannah totalTime + distanceCovered ethan totalTime = 23.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_covered_l49_4907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_gain_result_l49_4981

/-- Calculates the total percentage gain from selling silk and linen cloth -/
noncomputable def cloth_sale_gain (silk_cost_price silk_meters linen_cost_price linen_meters : ℝ)
  (silk_gain_meters : ℝ) (linen_profit_margin : ℝ) : ℝ :=
  let silk_cost := silk_cost_price * silk_meters
  let silk_gain := silk_cost_price * silk_gain_meters
  let silk_selling_price := silk_cost + silk_gain
  let linen_cost := linen_cost_price * linen_meters
  let linen_profit := linen_cost * linen_profit_margin
  let linen_selling_price := linen_cost + linen_profit
  let total_cost := silk_cost + linen_cost
  let total_selling_price := silk_selling_price + linen_selling_price
  let total_gain := total_selling_price - total_cost
  (total_gain / total_cost) * 100

/-- The total percentage gain from selling silk and linen cloth is approximately 35.56% -/
theorem cloth_sale_gain_result :
  ∃ (ε : ℝ), ε > 0 ∧ |cloth_sale_gain 30 15 45 20 10 0.2 - 35.56| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_gain_result_l49_4981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l49_4987

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

-- Theorem statement
theorem f_increasing_on_interval :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₂ < -2 → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l49_4987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bd_length_bounds_l49_4961

variable (a b : ℝ)
variable (h : a > 0 ∧ b > 0)

/-- Segment AB with length a --/
def segmentAB : ℝ := a

/-- Segment BC with length b --/
def segmentBC : ℝ := b

/-- Equilateral triangle ACD constructed on AC --/
def triangleACD : Prop := sorry

/-- B and D are on opposite sides of AC --/
def oppositePoints : Prop := sorry

/-- The maximum length of BD --/
def maxBD : ℝ := a + b

/-- The minimum length of BD --/
noncomputable def minBD : ℝ := Real.sqrt (a^2 + b^2 - a*b)

/-- Theorem stating the maximum and minimum lengths of BD --/
theorem bd_length_bounds (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi) : 
  let bd_length := sorry
  minBD a b ≤ bd_length ∧ bd_length ≤ maxBD a b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bd_length_bounds_l49_4961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximilian_revenue_l49_4921

/-- Calculates the annual revenue for a rental building -/
def annual_revenue (total_units : ℕ) (occupancy_rate : ℚ) (monthly_rent : ℕ) : ℕ :=
  (↑total_units * occupancy_rate).floor.toNat * monthly_rent * 12

/-- Theorem: The annual revenue for Mr. Maximilian's building is $360,000 -/
theorem maximilian_revenue :
  annual_revenue 100 (3/4) 400 = 360000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximilian_revenue_l49_4921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_limit_of_a_l49_4942

theorem upper_limit_of_a (A : ℕ) : 
  (∀ a b : ℕ, 6 < a ∧ a < A ∧ 3 < b ∧ b < 29 → 
    (abs ((A - 1) / 4 - 7 / 28 : ℚ) : ℝ) = 3.75) → 
  A = 17 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_limit_of_a_l49_4942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l49_4984

theorem trigonometric_equation_solution 
  (x y z : ℝ) 
  (hx : Real.sin x ≠ 0) 
  (hy : Real.cos y ≠ 0) :
  (Real.sin x^2 + 1 / Real.sin x^2)^3 + (Real.cos y^2 + 1 / Real.cos y^2)^3 = 16 * Real.sin z^2 ↔
  ∃ (n m k : ℤ), x = Real.pi/2 + Real.pi * ↑n ∧ y = Real.pi * ↑m ∧ z = Real.pi/2 + Real.pi * ↑k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l49_4984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_motion_acceleration_l49_4946

/-- Particle motion on a circular arc --/
structure ParticleMotion where
  k : ℝ  -- radius of the circular arc
  T : ℝ  -- total time of motion
  θ : ℝ → ℝ  -- angle as a function of time
  h_k_pos : k > 0
  h_T_pos : T > 0
  h_θ_diff : Differentiable ℝ θ
  h_θ_init : θ 0 = 0
  h_θ_final : θ T = (deriv θ) 0
  h_θ_vel_init : (deriv θ) 0 = 0
  h_θ_vel_final : (deriv θ) T = 0

/-- The acceleration is non-zero and at some point purely radial --/
theorem particle_motion_acceleration (p : ParticleMotion) :
  (∀ t ∈ Set.Icc 0 p.T, 
    (p.k * ((deriv p.θ) t)^2)^2 + (p.k * (deriv (deriv p.θ)) t)^2 ≠ 0) ∧
  (∃ t ∈ Set.Ioo 0 p.T, (deriv (deriv p.θ)) t = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_motion_acceleration_l49_4946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_abc_is_ten_l49_4958

noncomputable def f (x : ℝ) (a b c : ℕ) : ℝ :=
  if x > 0 then a * x + 4
  else if x = 0 then a * b
  else b * x + c

theorem sum_of_abc_is_ten (a b c : ℕ) : 
  f 3 a b c = 7 → f 0 a b c = 6 → f (-3) a b c = -15 → a + b + c = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_abc_is_ten_l49_4958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_cubic_polynomial_root_l49_4948

/-- A cubic polynomial with integer coefficients -/
def Q (x : ℝ) : ℝ := x^3 - 6*x^2 + 12*x - 11

/-- The cube root of 3 -/
noncomputable def cubeRoot3 : ℝ := (3 : ℝ) ^ (1/3 : ℝ)

theorem monic_cubic_polynomial_root : 
  (∀ x, Q x = x^3 - 6*x^2 + 12*x - 11) ∧ 
  (Q (cubeRoot3 + 2) = 0) ∧
  (∀ x, Q x = x^3 + (-6 : ℝ) * x^2 + (12 : ℝ) * x + (-11 : ℝ)) ∧
  (∃ a b c d : ℤ, ∀ x, Q x = (a : ℝ) * x^3 + (b : ℝ) * x^2 + (c : ℝ) * x + (d : ℝ)) ∧
  (∃ a : ℤ, ∀ x, Q x = (a : ℝ) * x^3 + Q x - (a : ℝ) * x^3) ∧
  (∃ a : ℤ, a = 1 ∧ ∀ x, Q x = (a : ℝ) * x^3 + Q x - (a : ℝ) * x^3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_cubic_polynomial_root_l49_4948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_surface_area_l49_4954

/-- The total surface area of a right pyramid with a square base --/
noncomputable def pyramid_surface_area (base_side : ℝ) (height : ℝ) : ℝ :=
  let base_area := base_side ^ 2
  let half_diagonal := base_side / 2
  let slant_height := Real.sqrt (height ^ 2 + half_diagonal ^ 2)
  let lateral_area := 2 * base_side * slant_height
  base_area + lateral_area

/-- Theorem: The surface area of a specific right pyramid --/
theorem specific_pyramid_surface_area :
  pyramid_surface_area 8 9 = 64 + 16 * Real.sqrt 97 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_surface_area_l49_4954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_purchase_savings_l49_4902

/-- Represents the discount and tax scenario for a shoe purchase. -/
structure ShoePurchase where
  original_price : ℝ
  first_discount : ℝ := 0.08
  second_discount : ℝ := 0.05
  sales_tax : ℝ := 0.06
  final_price : ℝ := 184

/-- Calculates the amount saved in a shoe purchase scenario. -/
def amount_saved (purchase : ShoePurchase) : ℝ :=
  purchase.original_price - purchase.final_price

/-- Theorem stating that the amount saved is approximately $25.78. -/
theorem shoe_purchase_savings (purchase : ShoePurchase) :
  ∃ ε > 0, |amount_saved purchase - 25.78| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_purchase_savings_l49_4902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_n_expression_l49_4932

/-- The inverse function of f -/
noncomputable def inverse_f (x : ℝ) : ℝ := x / (1 + x)

/-- The function f -/
noncomputable def f (x : ℝ) : ℝ := x / (1 - x)

/-- The recursive definition of f_n -/
noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => λ x => f_n n (f_n n x)

/-- The function g_n -/
noncomputable def g_n (n : ℕ) (x : ℝ) : ℝ := -1 / (f_n n x)

/-- The main theorem -/
theorem g_n_expression (n : ℕ) (x : ℝ) : g_n n x = 2^n - 1 / x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_n_expression_l49_4932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_in_M_sin_accompanying_pairs_l49_4998

/-- The set of functions with an accompanying pair -/
def M : Set (ℝ → ℝ) :=
  {f | ∃ (a k : ℝ), k ≠ 0 ∧ ∀ x, f (a + x) = k * f (a - x)}

/-- The accompanying pair of a function in M -/
def AccompanyingPair (f : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (a, k) := p; k ≠ 0 ∧ ∀ x, f (a + x) = k * f (a - x)}

theorem square_in_M : (fun x ↦ x^2) ∈ M := by
  sorry

theorem sin_accompanying_pairs :
  AccompanyingPair Real.sin = 
    {p : ℝ × ℝ | (∃ n : ℤ, p.1 = n * Real.pi + Real.pi/2 ∧ p.2 = 1) ∨ 
                 (∃ n : ℤ, p.1 = n * Real.pi ∧ p.2 = -1)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_in_M_sin_accompanying_pairs_l49_4998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_with_given_point_l49_4944

theorem trig_identity_with_given_point (θ : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos θ = -3/5 ∧ r * Real.sin θ = 4/5) →
  Real.sin (π/2 + θ) + Real.cos (π - θ) + Real.tan (2*π - θ) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_with_given_point_l49_4944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l49_4994

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 4 - 3 * x)

theorem monotonic_increasing_interval (k : ℤ) :
  StrictMonoOn f (Set.Icc ((2 / 3 : ℝ) * ↑k * Real.pi + Real.pi / 4) ((2 / 3 : ℝ) * ↑k * Real.pi + 7 * Real.pi / 12)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l49_4994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_fixed_point_l49_4962

/-- The complex function f(z) representing a scaling and rotation -/
noncomputable def f (z : ℂ) : ℂ := ((1 - Complex.I * Real.sqrt 2) * z + (3 * Real.sqrt 2 - 5 * Complex.I)) / 4

/-- The fixed point of the function f -/
noncomputable def c : ℂ := (9 * Real.sqrt 2 - 15 * Complex.I * Real.sqrt 2 + 10 - 6 * Complex.I) / 11

/-- Theorem stating that c is the fixed point of f -/
theorem c_is_fixed_point : f c = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_fixed_point_l49_4962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_jogger_in_38_seconds_l49_4900

/-- The time taken for a train to pass a jogger -/
noncomputable def train_passing_time (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (train_length : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- Theorem stating that the time taken for the train to pass the jogger is 38 seconds -/
theorem train_passes_jogger_in_38_seconds :
  train_passing_time 9 45 260 120 = 38 := by
  -- Unfold the definition of train_passing_time
  unfold train_passing_time
  -- Perform the calculation
  -- This is where we would normally prove each step,
  -- but for now we'll use sorry to skip the detailed proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_jogger_in_38_seconds_l49_4900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_existence_and_uniqueness_l49_4920

noncomputable def A : ℝ × ℝ := (4, -3)
noncomputable def B : ℝ × ℝ := (2, -1)

def line_l (x y : ℝ) : Prop := 4 * x + 3 * y - 2 = 0

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |4 * p.1 + 3 * p.2 - 2| / Real.sqrt (4^2 + 3^2)

theorem point_P_existence_and_uniqueness :
  ∃ P : ℝ × ℝ, 
    distance P A = distance P B ∧ 
    distance_to_line P = 2 ↔ 
    P = (1, -4) ∨ P = (27/7, -8/7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_existence_and_uniqueness_l49_4920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l49_4950

theorem trigonometric_identity (α β : Real) 
  (h : (Real.sin β)^4 / (Real.sin α)^2 + (Real.cos β)^4 / (Real.cos α)^2 = 1) :
  (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l49_4950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_72_l49_4952

/-- The capacity of a water tank in gallons. -/
def tank_capacity (c : ℝ) : Prop :=
  c > 0 ∧ 0.9 * c - 0.4 * c = 36

/-- The total capacity of the water tank is 72 gallons. -/
theorem tank_capacity_is_72 : ∃ c, tank_capacity c ∧ c = 72 := by
  -- We'll use 72 as our witness for the existential quantifier
  use 72
  -- Now we need to prove that tank_capacity 72 holds and that 72 = 72
  apply And.intro
  · -- Prove tank_capacity 72
    apply And.intro
    · -- Prove 72 > 0
      norm_num
    · -- Prove 0.9 * 72 - 0.4 * 72 = 36
      norm_num
  · -- Prove 72 = 72
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_is_72_l49_4952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3x_period_l49_4957

/-- The period of tan(3x) is π/3 -/
theorem tan_3x_period : ∃ (T : ℝ), T > 0 ∧ 
  (∀ (x : ℝ), Real.tan (3 * x) = Real.tan (3 * (x + T))) ∧ 
  (∀ (S : ℝ), 0 < S ∧ S < T → ∃ (y : ℝ), Real.tan (3 * y) ≠ Real.tan (3 * (y + S))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3x_period_l49_4957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_statement2_correct_l49_4912

-- Define the statements
def statement1 : Prop := ({0} : Set ℕ) = {0}
def statement2 : Prop := ({1, 2, 3} : Set ℕ) = {3, 2, 1}
def statement3 : Prop := (∃ s : Set ℝ, s = {1, 1, 2} ∧ ∀ x, x ∈ s ↔ (x - 1)^2 * (x - 2) = 0)
noncomputable def statement4 : Prop := Set.Finite {x : ℝ | 4 < x ∧ x < 5}

-- Theorem stating that only statement2 is correct
theorem only_statement2_correct :
  ¬statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_statement2_correct_l49_4912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_max_values_l49_4997

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  a : ℝ
  A : Point
  B : Point
  C : Point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about maximum values in an equilateral triangle -/
theorem equilateral_triangle_max_values (triangle : EquilateralTriangle) :
  ∃ (P : Point),
    (∀ (Q : Point), distance triangle.A Q + distance triangle.B Q + distance triangle.C Q ≤ 2 * triangle.a) ∧
    (∀ (Q : Point), distance triangle.A Q * distance triangle.B Q * distance triangle.C Q ≤ (Real.sqrt 3 / 8) * triangle.a^3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_max_values_l49_4997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l49_4995

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := |x + 1 - 2*a| + |x - a^2|
noncomputable def g (x : ℝ) : ℝ := x^2 - 2*x - 4 + 4 / (x - 1)^2

-- Theorem for part I
theorem part_one (a : ℝ) :
  f a (2*a^2 - 1) > 4 * |a - 1| ↔ a < -5/3 ∨ a > 1 := by
  sorry

-- Theorem for part II
theorem part_two (a : ℝ) :
  (∃ x y : ℝ, f a x + g y ≤ 0) ↔ 0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l49_4995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_specific_l49_4966

/-- The volume of a cone with given slant height and height -/
noncomputable def cone_volume (l h : ℝ) : ℝ :=
  let r := Real.sqrt (l^2 - h^2)
  (1/3) * Real.pi * r^2 * h

/-- Theorem stating the volume of a cone with slant height 5 and height 4 -/
theorem cone_volume_specific : cone_volume 5 4 = 12 * Real.pi := by
  -- Unfold the definition of cone_volume
  unfold cone_volume
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_specific_l49_4966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_time_l49_4915

-- Define the lawn dimensions
def lawn_length : ℝ := 100
def lawn_width : ℝ := 140

-- Define the mower characteristics
def mower_swath_inches : ℝ := 30
def mower_overlap_inches : ℝ := 6

-- Define Sam's mowing speed
def mowing_speed : ℝ := 4500

-- Theorem statement
theorem lawn_mowing_time :
  ∃ (time : ℝ),
    (abs (time - 1.6) < 0.1) ∧
    (time = (lawn_length * lawn_width) /
            (((mower_swath_inches - mower_overlap_inches) / 12) * mowing_speed)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_mowing_time_l49_4915
