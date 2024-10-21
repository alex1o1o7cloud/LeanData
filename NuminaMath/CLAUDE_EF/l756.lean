import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l756_75683

/-- The ellipse defined by x^2/9 + y^2/4 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 4 = 1}

/-- The line defined by x + 2y - 10 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + 2 * p.2 - 10 = 0}

/-- The distance function between a point and a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) : ℝ :=
  |p.1 + 2 * p.2 - 10| / Real.sqrt 5

theorem min_distance_ellipse_to_line :
  ∃ (d : ℝ), d = Real.sqrt 5 ∧
  ∀ (p : ℝ × ℝ), p ∈ Ellipse → distanceToLine p ≥ d ∧
  ∃ (q : ℝ × ℝ), q ∈ Ellipse ∧ distanceToLine q = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l756_75683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_regular_hours_example_l756_75686

/-- Represents a worker's pay structure and hours worked -/
structure WorkerPay where
  regularRate : ℚ
  overtimeRate : ℚ
  totalPay : ℚ
  overtimeHours : ℚ

/-- Calculates the maximum regular hours worked given the worker's pay structure -/
def maxRegularHours (w : WorkerPay) : ℚ :=
  (w.totalPay - w.overtimeRate * w.overtimeHours) / w.regularRate

/-- Theorem stating the maximum regular hours worked for the given conditions -/
theorem max_regular_hours_example :
  let w : WorkerPay := {
    regularRate := 3,
    overtimeRate := 6,
    totalPay := 180,
    overtimeHours := 10
  }
  maxRegularHours w = 40 := by
  -- Proof goes here
  sorry

#eval maxRegularHours {
  regularRate := 3,
  overtimeRate := 6,
  totalPay := 180,
  overtimeHours := 10
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_regular_hours_example_l756_75686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_remaining_numbers_l756_75681

theorem average_of_remaining_numbers 
  (total : ℕ) 
  (avg_all : ℚ) 
  (avg_first_two : ℚ) 
  (avg_next_two : ℚ) 
  (h_total : total = 6) 
  (h_avg_all : avg_all = 395/100) 
  (h_avg_first_two : avg_first_two = 42/10) 
  (h_avg_next_two : avg_next_two = 385/100) : 
  (total * avg_all - (2 * avg_first_two + 2 * avg_next_two)) / 2 = 38/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_remaining_numbers_l756_75681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_implies_a_geq_one_l756_75628

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 - x^2 + x

-- State the theorem
theorem monotonic_increasing_implies_a_geq_one :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 0 2, Monotone (fun x => f a x)) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_implies_a_geq_one_l756_75628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounds_l756_75665

theorem sequence_bounds (n : ℕ) (hn : n > 0) : 
  ∃ a : ℕ → ℚ, 
    a 0 = 1 / 2 ∧
    (∀ k, a (k + 1) = a k + (1 / n) * (a k)^2) ∧
    1 - 1 / n < a n ∧ a n < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounds_l756_75665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_integer_is_32_l756_75682

theorem one_integer_is_32 
  (a b c d e : ℚ) 
  (h1 : (a + b + c + d) / 4 + e = 44)
  (h2 : (a + b + c + e) / 4 + d = 38)
  (h3 : (a + b + d + e) / 4 + c = 35)
  (h4 : (a + c + d + e) / 4 + b = 30)
  (h5 : (b + c + d + e) / 4 + a = 29)
  (pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) :
  a = 32 ∨ b = 32 ∨ c = 32 ∨ d = 32 ∨ e = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_integer_is_32_l756_75682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_equation_plane_l756_75623

/-- Given a constant c, the set of points (r, θ, z) in cylindrical coordinates
    satisfying θ = c + π/4 forms a plane -/
theorem cylindrical_equation_plane (c : ℝ) :
  {p : ℝ × ℝ × ℝ | p.2.1 = c + π/4} = {p : ℝ × ℝ × ℝ | ∃ (r z : ℝ), p = (r, c + π/4, z)} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_equation_plane_l756_75623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l756_75670

noncomputable section

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop := 4 * x - 3 * y + 6 * z = 24

/-- The point we're measuring distance from -/
def A : Fin 3 → ℝ := ![2, 1, 4]

/-- The point claimed to be closest to A on the plane -/
def P : Fin 3 → ℝ := ![102/61, 76/61, 214/61]

/-- Distance between two points in 3D space -/
def distance (p1 p2 : Fin 3 → ℝ) : ℝ :=
  Real.sqrt ((p1 0 - p2 0)^2 + (p1 1 - p2 1)^2 + (p1 2 - p2 2)^2)

theorem closest_point_on_plane :
  plane_equation (P 0) (P 1) (P 2) ∧
  ∀ Q : Fin 3 → ℝ, plane_equation (Q 0) (Q 1) (Q 2) → distance A P ≤ distance A Q :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l756_75670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l756_75677

/-- A triangular pyramid with mutually perpendicular lateral faces -/
structure TriangularPyramid where
  a : ℝ
  b : ℝ
  c : ℝ
  lateral_face_area_1 : ℝ := a^2
  lateral_face_area_2 : ℝ := b^2
  lateral_face_area_3 : ℝ := c^2
  lateral_faces_perpendicular : True

/-- The volume of a triangular pyramid with mutually perpendicular lateral faces -/
noncomputable def volume (p : TriangularPyramid) : ℝ :=
  (1/3) * p.a * p.b * p.c * Real.sqrt 2

/-- Theorem: The volume of a triangular pyramid with mutually perpendicular lateral faces
    of areas a², b², and c² is equal to (1/3) * a * b * c * √2 -/
theorem triangular_pyramid_volume (p : TriangularPyramid) :
  volume p = (1/3) * p.a * p.b * p.c * Real.sqrt 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l756_75677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l756_75631

def sequence_term (n : ℕ) : ℚ :=
  (2187000 : ℚ) / (3 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ z : ℤ, q = z

theorem last_integer_in_sequence :
  ∃ k : ℕ, (∀ i < k, is_integer (sequence_term i)) ∧
           is_integer (sequence_term k) ∧
           ∀ j > k, ¬ is_integer (sequence_term j) ∧
           sequence_term k = 1000 := by
  sorry

#eval sequence_term 7  -- Should output 1000
#eval sequence_term 8  -- Should output a non-integer value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_l756_75631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_initial_state_after_even_rounds_l756_75676

/-- Represents the state of the game as a triple of natural numbers -/
def GameState := Fin 3 → ℕ

/-- The initial state of the game where each player has 1 dollar -/
def initialState : GameState := fun _ => 1

/-- Checks if a given state is valid according to the game rules -/
def isValidState (state : GameState) : Prop :=
  (state 0 + state 1 + state 2 = 3) ∧ 
  (∀ i, state i ≤ 2) ∧
  (∀ i, state i = 0 → ∃ j, j ≠ i ∧ state j > 1)

/-- Defines the transition probability from one state to another -/
noncomputable def transitionProb (fromState toState : GameState) : ℝ :=
  sorry

/-- The probability of being in the initial state after n rounds -/
noncomputable def probInitialStateAfterNRounds (n : ℕ) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem prob_initial_state_after_even_rounds (n : ℕ) (h : Even n) :
  probInitialStateAfterNRounds n = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_initial_state_after_even_rounds_l756_75676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l756_75646

/-- The circle with center (15, 2) and radius 5 -/
def circle_eq (x y : ℝ) : Prop := (x - 15)^2 + (y - 2)^2 = 25

/-- A line passing through the point (0, 7) -/
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k * x + 7

/-- The line is tangent to the circle -/
def is_tangent (k : ℝ) : Prop :=
  ∃! x y, circle_eq x y ∧ line_through_A k x y

/-- The only lines passing through (0, 7) tangent to the circle are y = 7 and y = -3/4x + 7 -/
theorem tangent_lines_to_circle :
  (∀ k, is_tangent k ↔ k = 0 ∨ k = -3/4) := by
  sorry

#check tangent_lines_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l756_75646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_theorem_l756_75617

/-- Triangle DEF with side lengths a, b, c -/
structure Triangle (a b c : ℝ) where
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0

/-- Rectangle WXYZ inscribed in triangle DEF -/
structure InscribedRectangle (t : Triangle a b c) (l : ℝ) where
  length : l > 0
  width : ℝ
  area : ℝ → ℝ → ℝ
  area_eq : area = fun γ δ ↦ γ * l - δ * l^2

/-- Helper function to calculate triangle area using Heron's formula -/
noncomputable def area (t : Triangle a b c) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The main theorem -/
theorem inscribed_rectangle_area_theorem 
  (t : Triangle 15 36 39) 
  (r : InscribedRectangle t 18) 
  (h_area : r.area r.width r.width = (area t) / 2) :
  ∃ (γ δ : ℝ), γ = 9720/1278 ∧ δ = 45/7668 ∧ r.area γ δ = (area t) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_theorem_l756_75617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l756_75616

/-- 
Given real numbers a, b, and c, where a ≠ 0, the expression 
a * cos θ + b * sin θ + c * tan θ is maximized when 
cos θ = a / sqrt(a² + b²), sin θ = b / sqrt(a² + b²), and tan θ = b / a. 
The maximum value is sqrt(a² + b²) + c * b / a.
-/
theorem max_value_trig_expression (a b c : ℝ) (ha : a ≠ 0) :
  ∃ θ : ℝ, 
    Real.cos θ = a / Real.sqrt (a^2 + b^2) ∧ 
    Real.sin θ = b / Real.sqrt (a^2 + b^2) ∧ 
    Real.tan θ = b / a ∧
    ∀ φ : ℝ, a * Real.cos φ + b * Real.sin φ + c * Real.tan φ ≤ Real.sqrt (a^2 + b^2) + c * b / a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l756_75616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l756_75650

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2*x + 4 else 2^x

theorem f_range (a : ℝ) : f (f a) > f (f a + 1) → -5/2 < a ∧ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l756_75650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_specific_draw_l756_75637

-- Define the number of slips and the range of numbers
def total_slips : ℕ := 60
def number_range : ℕ := 20
def slips_per_number : ℕ := 3
def drawn_slips : ℕ := 5

-- Define the probability r
def r : ℚ := 30810 / 5461512

-- State the theorem
theorem probability_of_specific_draw : 
  (number_range * 1 * Nat.choose (number_range - 1) 2 * (slips_per_number ^ 2) : ℚ) / 
  (Nat.choose total_slips drawn_slips : ℚ) = r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_specific_draw_l756_75637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l756_75655

theorem expression_equality : 
  Real.sin (30 * Real.pi / 180) + (1/2 : Real) - (1 : Real) + |(-2)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l756_75655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_and_y_product_of_x_and_y_expression_one_expression_two_l756_75652

-- Define x and y as noncomputable
noncomputable def x : ℝ := (Real.sqrt 11 / 2) + (Real.sqrt 7 / 2)
noncomputable def y : ℝ := (Real.sqrt 11 / 2) - (Real.sqrt 7 / 2)

-- Theorem statements
theorem sum_of_x_and_y : x + y = Real.sqrt 11 := by sorry

theorem product_of_x_and_y : x * y = 1 := by sorry

theorem expression_one : x^2 * y + x * y^2 = Real.sqrt 11 := by sorry

theorem expression_two : y / x + x / y = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_and_y_product_of_x_and_y_expression_one_expression_two_l756_75652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_extreme_points_iff_a_in_range_l756_75615

/-- The function f(x) = xe^x - ae^(2x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * Real.exp (2 * x)

/-- f'(x) = e^x * (x + 1 - 2ae^x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x + 1 - 2 * a * Real.exp x)

/-- g(x) = x + 1 - 2ae^x -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x + 1 - 2 * a * Real.exp x

/-- g'(x) = 1 - 2ae^x -/
noncomputable def g_derivative (a : ℝ) (x : ℝ) : ℝ := 1 - 2 * a * Real.exp x

theorem f_has_two_extreme_points_iff_a_in_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    f_derivative a x₁ = 0 ∧ 
    f_derivative a x₂ = 0 ∧ 
    (∀ x : ℝ, x ≠ x₁ ∧ x ≠ x₂ → f_derivative a x ≠ 0)) ↔ 
  (0 < a ∧ a < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_extreme_points_iff_a_in_range_l756_75615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l756_75669

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- Define the theorem
theorem triangle_theorem (t : Triangle) :
  (t.b * Real.cos t.C = 3 * t.a * Real.cos t.B - t.c * Real.cos t.B) →
  (Real.cos t.B = 1/3) ∧
  ((t.b * Real.cos t.C = 3 * t.a * Real.cos t.B - t.c * Real.cos t.B) →
   (t.a * t.c * Real.cos t.B = 2) →
   (t.b = 2 * Real.sqrt 2) →
   (t.a = Real.sqrt 6 ∧ t.c = Real.sqrt 6)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l756_75669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l756_75633

/-- Calculates the speed of a train given its length, the time it takes to pass a person
    moving in the opposite direction, and the person's speed. -/
noncomputable def train_speed (train_length : ℝ) (passing_time : ℝ) (person_speed : ℝ) : ℝ :=
  (train_length / 1000) / (passing_time / 3600) - person_speed

/-- Theorem stating that a train of length 110 m passing a man running at 9 km/h
    in the opposite direction in 4 seconds has a speed of 90 km/h. -/
theorem train_speed_calculation :
  train_speed 110 4 9 = 90 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l756_75633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_difference_positive_l756_75642

theorem exponential_difference_positive (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (2 : ℝ)^x - (2 : ℝ)^y > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_difference_positive_l756_75642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l756_75609

def my_sequence (x : ℕ → ℚ) : Prop :=
  ∀ k : ℕ, k ≥ 2 → x k = 1 / (1 - x (k - 1))

theorem sequence_property (x : ℕ → ℚ) (h : my_sequence x) :
  x 2 = 5 → x 7 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l756_75609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l756_75644

/-- Calculates the average speed of a train given two segments of its journey. -/
noncomputable def average_speed (x : ℝ) : ℝ :=
  let distance1 := x
  let speed1 := (65 : ℝ)
  let distance2 := 2 * x
  let speed2 := (20 : ℝ)
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  total_distance / total_time

/-- Theorem stating that the average speed of the train is 26 kmph. -/
theorem train_average_speed (x : ℝ) (h : x > 0) :
  average_speed x = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l756_75644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_eq_one_set_l756_75626

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 1 else x - 3

-- Define the set of x values for which f(f(x)) = 1
def S : Set ℝ := {x | f (f x) = 1}

-- Theorem statement
theorem f_f_eq_one_set : S = Set.Icc 0 1 ∪ Set.Icc 3 4 ∪ {7} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_eq_one_set_l756_75626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l756_75699

-- Define the sets M and N
def M : Set ℝ := {x | (x - 2) / (x - 3) < 0}
def N : Set ℝ := {x | Real.log (x - 2) / Real.log (1/2) ≥ 1}

-- State the theorem
theorem set_intersection_theorem :
  M ∩ N = Set.Ioo 2 (5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l756_75699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_share_after_tax_b_share_is_1050_l756_75649

/-- Represents an employee's salary share --/
structure SalaryShare where
  amount : ℚ
  deriving Repr

/-- Represents the salary distribution --/
structure SalaryDistribution where
  a : SalaryShare
  b : SalaryShare
  c : SalaryShare
  d : SalaryShare
  deriving Repr

/-- Calculates the tax on a salary --/
def calculateTax (salary : ℚ) : ℚ :=
  if salary > 1500 then (salary - 1500) * 15 / 100 else 0

/-- Theorem stating B's share after tax deduction --/
theorem b_share_after_tax (dist : SalaryDistribution) : ℚ :=
  dist.b.amount - calculateTax dist.b.amount

/-- Main theorem to prove --/
theorem b_share_is_1050 (dist : SalaryDistribution) : 
  (b_share_after_tax dist = 1050) ∧ 
  (dist.a.amount + dist.b.amount + dist.c.amount + dist.d.amount > 0) ∧
  (dist.a.amount : ℚ) / (dist.b.amount : ℚ) = 2 / 3 ∧
  (dist.b.amount : ℚ) / (dist.c.amount : ℚ) = 3 / 4 ∧
  (dist.c.amount : ℚ) / (dist.d.amount : ℚ) = 4 / 6 ∧
  dist.d.amount - dist.c.amount = 700 ∧
  dist.a.amount ≥ 1000 ∧ dist.b.amount ≥ 1000 ∧ 
  dist.c.amount ≥ 1000 ∧ dist.d.amount ≥ 1000 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_share_after_tax_b_share_is_1050_l756_75649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l756_75697

noncomputable section

-- Define the curves
def curve1 (x : ℝ) : ℝ := 1 / x
def curve2 (x : ℝ) : ℝ := x^2

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 1)

-- Define the tangent lines
def tangent1 (x : ℝ) : ℝ := -x + 2
def tangent2 (x : ℝ) : ℝ := 2*x - 1

-- Define the x-intercepts of the tangent lines
def x_intercept1 : ℝ := 2
def x_intercept2 : ℝ := 1/2

-- Define a function to calculate the area of a triangle given its vertices
def area_of_triangle (vertices : List (ℝ × ℝ)) : ℝ :=
  match vertices with
  | [(x1, y1), (x2, y2), (x3, y3)] => 
    abs ((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2)
  | _ => 0  -- Return 0 for invalid input

-- State the theorem
theorem triangle_area : 
  let vertices := [intersection_point, (x_intercept1, 0), (x_intercept2, 0)]
  area_of_triangle vertices = 3/4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l756_75697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l756_75603

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := (x - 2) / (x - 1)
def g (m x : ℝ) : ℝ := m * x + 1 - m

-- Define the intersection points A and B
def A (m : ℝ) : ℝ × ℝ := (1 - Real.sqrt (-1/m), 1 - m * Real.sqrt (-1/m))
def B (m : ℝ) : ℝ × ℝ := (1 + Real.sqrt (-1/m), 1 + m * Real.sqrt (-1/m))

-- Define the moving point P
def P : ℝ × ℝ → Prop := fun _ => True

-- Theorem statement
theorem trajectory_of_P (m : ℝ) (h : m < 0) :
  ∀ x y : ℝ, P (x, y) ↔ (x - 1)^2 + (y - 1)^2 = 1 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l756_75603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_at_a_half_f_three_zeros_iff_l756_75659

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - a * x^2 + a / x^2

-- Theorem 1
theorem f_positive_at_a_half (a : ℝ) (h : 0 < a ∧ a < 1) : f a (a/2) > 0 := by
  sorry

-- Theorem 2
theorem f_three_zeros_iff (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ 0 < a ∧ a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_at_a_half_f_three_zeros_iff_l756_75659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l756_75653

-- Define the circle M
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

noncomputable def chord_length (c : Circle) : ℝ :=
  2 * Real.sqrt (c.radius ^ 2 - c.center.2 ^ 2)

noncomputable def is_tangent_to_line (c : Circle) : Prop :=
  let d := abs (3 * c.center.1 + c.center.2) / Real.sqrt 10
  d = c.radius

-- State the theorem
theorem circle_equation (M : Circle) 
  (h1 : is_in_first_quadrant M.center)
  (h2 : chord_length M = 6)
  (h3 : is_tangent_to_line M) :
  ∀ (x y : ℝ), (x - 3)^2 + (y - 1)^2 = 10 ↔ 
    (x - M.center.1)^2 + (y - M.center.2)^2 = M.radius^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l756_75653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturer_profit_percentage_is_approximately_18_percent_l756_75636

-- Define the given values
noncomputable def manufacturer_cost : ℝ := 17
noncomputable def wholesaler_profit_rate : ℝ := 0.20
noncomputable def retailer_profit_rate : ℝ := 0.25
noncomputable def customer_price : ℝ := 30.09

-- Define the calculated values
noncomputable def retailer_cost : ℝ := customer_price / (1 + retailer_profit_rate)
noncomputable def wholesaler_price : ℝ := retailer_cost / (1 + wholesaler_profit_rate)
noncomputable def manufacturer_profit : ℝ := wholesaler_price - manufacturer_cost
noncomputable def manufacturer_profit_rate : ℝ := manufacturer_profit / manufacturer_cost

-- Theorem statement
theorem manufacturer_profit_percentage_is_approximately_18_percent :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |manufacturer_profit_rate - 0.18| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturer_profit_percentage_is_approximately_18_percent_l756_75636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_of_2_36_equals_8_y_l756_75675

theorem eighth_of_2_36_equals_8_y (y : ℝ) : (1 / 8) * (2^36 : ℝ) = 8^y → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_of_2_36_equals_8_y_l756_75675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_implies_a_range_l756_75651

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 1) * x + 1 else x^2 - 2 * a * x + 6

theorem function_range_implies_a_range (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_implies_a_range_l756_75651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_of_4_equals_2_l756_75600

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x

-- State the theorem
theorem inverse_f_of_4_equals_2 :
  ∃ (f_inv : ℝ → ℝ), (∀ x, f (f_inv x) = x ∧ f_inv (f x) = x) → f_inv 4 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_of_4_equals_2_l756_75600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_quadratic_inequality_l756_75694

theorem range_of_a_for_quadratic_inequality (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ 
  a ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) := 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_quadratic_inequality_l756_75694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_perpendicular_chord_l756_75654

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-1, 0)

-- Define a line passing through a point
def line_through (p : ℝ × ℝ) (t : ℝ) (x y : ℝ) : Prop :=
  x = t * y + p.1

-- Define perpendicularity of two vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ :=
  sorry -- Implementation details omitted for brevity

-- Theorem statement
theorem distance_to_perpendicular_chord (A B : ℝ × ℝ) (t : ℝ) :
  ellipse A.1 A.2 →
  ellipse B.1 B.2 →
  line_through left_focus t A.1 A.2 →
  line_through left_focus t B.1 B.2 →
  perpendicular A B →
  distance_point_to_line (0, 0) (line_through left_focus t) = Real.sqrt 6 / 3 :=
by
  sorry -- Proof omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_perpendicular_chord_l756_75654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_negative_solution_set_l756_75634

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

-- State the theorem
theorem f_derivative_negative_solution_set :
  {x : ℝ | x > 0 ∧ (deriv f) x < 0} = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_negative_solution_set_l756_75634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_p_eq_n_l756_75645

noncomputable def p (S : Finset Nat) : Rat :=
  (S.prod (λ x => x : Nat → Rat))⁻¹

def subsets (n : Nat) : Finset (Finset Nat) :=
  (Finset.range n).powerset.filter (λ S => S.Nonempty)

theorem sum_p_eq_n (n : Nat) : 
  (subsets n).sum p = n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_p_eq_n_l756_75645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_with_seven_percentage_l756_75601

/-- A palindrome is an integer that reads the same forward and backward. -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- Checks if a number contains the digit 7. -/
def containsSeven (n : ℕ) : Prop := sorry

/-- The set of palindromes between 1000 and 2000. -/
def palindromeSet : Finset ℕ := sorry

/-- The set of palindromes between 1000 and 2000 that contain at least one digit 7. -/
def palindromeWithSevenSet : Finset ℕ := sorry

theorem palindrome_with_seven_percentage :
  (palindromeWithSevenSet.card : ℚ) / (palindromeSet.card : ℚ) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_with_seven_percentage_l756_75601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_one_l756_75661

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log (x + Real.sqrt (a + x^2))

-- State the theorem
theorem f_even_implies_a_eq_one (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_one_l756_75661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magician_disappearing_act_l756_75678

/-- The magician's disappearing act problem -/
theorem magician_disappearing_act 
  (total_performances : ℕ) 
  (no_reappearance_prob : ℚ) 
  (total_reappeared : ℕ) 
  (h1 : total_performances = 100)
  (h2 : no_reappearance_prob = 1/10)
  (h3 : total_reappeared = 110)
  : (total_reappeared - total_performances * (1 - no_reappearance_prob)) / total_performances = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magician_disappearing_act_l756_75678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_containing_triangle_l756_75660

/-- A set of n distinct points on a plane -/
def Plane := ℝ × ℝ

/-- The area of a triangle formed by three points -/
def area (A B C : Plane) : ℝ := sorry

/-- Check if a point is inside or on the boundary of a triangle -/
def point_in_triangle (P : Plane) (T : Plane × Plane × Plane) : Prop := sorry

/-- The theorem to be proved -/
theorem exists_containing_triangle (n : ℕ) (P : Fin n → Plane) 
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → k ≠ i → area (P i) (P j) (P k) ≤ 1) :
  ∃ (T : Plane × Plane × Plane), 
    (area T.1 T.2.1 T.2.2 ≤ 4) ∧ (∀ i : Fin n, point_in_triangle (P i) T) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_containing_triangle_l756_75660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l756_75657

noncomputable section

open Real

theorem triangle_properties :
  ∀ (a b c A B C : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  a / sin A = b / sin B ∧
  b / sin B = c / sin C →
  (sin (15 * π / 180) + cos (15 * π / 180) = sqrt 6 / 2) ∧
  (A > C → sin A > sin C) :=
by
  intros a b c A B C h
  have h1 : sin (15 * π / 180) + cos (15 * π / 180) = sqrt 6 / 2 := by sorry
  have h2 : A > C → sin A > sin C := by sorry
  exact ⟨h1, h2⟩

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l756_75657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_inequality_l756_75630

theorem lcm_inequality (n : ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ i < n, 0 < a i)
  (h2 : ∀ i j, i < j → i < n → j < n → a i ≤ a j)
  (h3 : ∀ i < n, a i ≤ 2*n)
  (h4 : ∀ i j, i ≠ j → i < n → j < n → Nat.lcm (a i) (a j) ≥ 2*n) :
  3 * a 0 > 2 * n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_inequality_l756_75630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_l756_75632

/-- Given a circle C with equation (x+1)^2 + (y-2)^2 = 4 and a line with slope 1 passing through 
    the center of C, prove that the equation of this line is x - y + 3 = 0 -/
theorem line_through_circle_center (x y : ℝ) : 
  ((x+1)^2 + (y-2)^2 = 4) → (y - 2 = x + 1) → x - y + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_l756_75632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_condition_count_l756_75629

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Checks if a number satisfies the divisor condition -/
def satisfies_condition (n : ℕ) : Bool :=
  num_divisors (10 * n) = 3 * num_divisors n

/-- The count of numbers satisfying the condition -/
def count_satisfying (max : ℕ) : ℕ :=
  (Finset.range max).filter (fun n => satisfies_condition n) |>.card

/-- The main theorem -/
theorem divisor_condition_count :
  count_satisfying 101 = 28 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_condition_count_l756_75629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_for_specific_trapezoid_l756_75691

/-- An isosceles trapezoid with specific dimensions and circles -/
structure IsoscelesTrapezoidWithCircles where
  /-- Length of the base (AB) -/
  base : ℝ
  /-- Length of the top (CD) -/
  top : ℝ
  /-- Length of the leg (BC or DA) -/
  leg : ℝ
  /-- Radius of circles centered at A and B -/
  large_radius : ℝ
  /-- Radius of circles centered at C and D -/
  small_radius : ℝ

/-- The radius of the inner circle tangent to all four outer circles -/
noncomputable def inner_circle_radius (t : IsoscelesTrapezoidWithCircles) : ℝ :=
  (-130 + 120 * Real.sqrt 2) / 35

/-- Theorem stating the radius of the inner circle for the given trapezoid -/
theorem inner_circle_radius_for_specific_trapezoid :
  let t : IsoscelesTrapezoidWithCircles :=
    { base := 8
      top := 2
      leg := 3
      large_radius := 4
      small_radius := 1 }
  inner_circle_radius t = (-130 + 120 * Real.sqrt 2) / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_for_specific_trapezoid_l756_75691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l756_75696

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6)

theorem f_properties :
  (∀ x, f (5 * Real.pi / 12 + x) = -f (5 * Real.pi / 12 - x)) ∧
  (∀ x, f x = Real.sin (2 * x + Real.pi / 6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l756_75696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2014_l756_75666

def mySequence : ℕ → ℤ
  | 0 => -4
  | 1 => 0
  | 2 => 4
  | 3 => 1
  | n + 4 => mySequence n

theorem mySequence_2014 : mySequence 2014 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2014_l756_75666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l756_75612

-- Define higher_order_terms as a function
def higher_order_terms (x : ℝ) : ℝ := 0

theorem coefficient_x_cubed_in_expansion : 
  let f := fun x => (1 - 2*x)^6 * (1 + x)
  ∃ a b c d e : ℝ, ∀ x : ℝ, f x = a*x^3 + b*x^4 + c*x^5 + d*x^6 + e*x^7 + higher_order_terms x
  ∧ a = -100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l756_75612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_three_fifth_power_seven_l756_75607

theorem cube_root_of_three_fifth_power_seven (x : ℝ) :
  x = (5^7 + 5^7 + 5^7)^(1/3) → x = 225 * 15^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_three_fifth_power_seven_l756_75607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_unique_root_when_a_negative_l756_75624

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

-- Theorem for the tangent line equation
theorem tangent_line_equation (a : ℝ) :
  ∃ (m : ℝ), m = (a * Real.exp 1 - 1) / Real.exp 1 ∧
  ∀ (x : ℝ), x > 0 → HasDerivAt (f a) (m * x) x := by sorry

-- Theorem for the unique root when a < 0
theorem unique_root_when_a_negative (a : ℝ) (h : a < 0) :
  ∃! (x : ℝ), x > 0 ∧ f a x + a * x^2 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_unique_root_when_a_negative_l756_75624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_unity_quadratic_equation_l756_75604

theorem roots_of_unity_quadratic_equation :
  ∀ z : ℂ, (∃ n : ℕ, n > 0 ∧ z^n = 1) →
  (∃ a : ℤ, z^2 + a*z - 2 = 0) →
  (z = 1 ∨ z = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_unity_quadratic_equation_l756_75604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diff_extreme_points_l756_75679

noncomputable section

open Real

/-- The function g(x) = x - 1/x + a*ln(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x - 1/x + a * log x

/-- Theorem: The minimum value of g(x₁) - g(x₂) is -4/e -/
theorem min_diff_extreme_points (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), 
    0 < x₁ ∧ x₁ ≤ exp 1 ∧
    (∀ x, x > 0 → (deriv (g a)) x = 0 → x = x₁ ∨ x = x₂) →
    g a x₁ - g a x₂ ≥ -4 / exp 1 ∧
    ∃ (y₁ y₂ : ℝ), 0 < y₁ ∧ y₁ ≤ exp 1 ∧ 
      (∀ x, x > 0 → (deriv (g a)) x = 0 → x = y₁ ∨ x = y₂) ∧
      g a y₁ - g a y₂ = -4 / exp 1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diff_extreme_points_l756_75679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_change_l756_75687

/-- Proves that given the original ratio of milk to water is 6:3, the original quantity of milk is 45 liters, and 10 liters of water is added, the new ratio of milk to water is 18:13. -/
theorem milk_water_ratio_change (original_milk : ℚ) (original_water : ℚ) (added_water : ℚ) :
  original_milk / original_water = 6 / 3 →
  original_milk = 45 →
  added_water = 10 →
  let new_water := original_water + added_water
  original_milk / new_water = 18 / 13 := by
  sorry

#check milk_water_ratio_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_change_l756_75687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_problem_solution_l756_75606

/-- Calculates the number of innings played initially given the conditions of the cricket problem -/
noncomputable def initial_innings (initial_average : ℝ) (next_innings_runs : ℝ) (average_increase : ℝ) : ℝ :=
  (next_innings_runs - initial_average - average_increase) / average_increase

theorem cricket_problem_solution :
  let initial_average : ℝ := 42
  let next_innings_runs : ℝ := 86
  let average_increase : ℝ := 4
  initial_innings initial_average next_innings_runs average_increase = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_problem_solution_l756_75606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_correct_l756_75672

/-- Represents the taxi fare structure and ride details -/
structure TaxiRide where
  baseFare : ℚ  -- Base fare for the first 3/4 mile
  baseMiles : ℚ  -- Miles covered in base fare
  additionalRate : ℚ  -- Rate per 0.1 mile after base miles
  tipAmount : ℚ  -- Amount of tip
  totalAmount : ℚ  -- Total amount available for the ride

/-- Calculates the maximum distance that can be traveled given the fare structure and total amount -/
def maxDistance (ride : TaxiRide) : ℚ :=
  ride.baseMiles + 
  (ride.totalAmount - ride.tipAmount - ride.baseFare) / 
  (10 * ride.additionalRate)

/-- Theorem stating that the maximum distance for the given conditions is 4.35 miles -/
theorem max_distance_is_correct (ride : TaxiRide) 
  (h1 : ride.baseFare = 3)
  (h2 : ride.baseMiles = 3/4)
  (h3 : ride.additionalRate = 1/4)
  (h4 : ride.tipAmount = 3)
  (h5 : ride.totalAmount = 15) :
  maxDistance ride = 87/20 := by
  sorry

/-- Example calculation -/
def exampleRide : TaxiRide := {
  baseFare := 3,
  baseMiles := 3/4,
  additionalRate := 1/4,
  tipAmount := 3,
  totalAmount := 15
}

#eval maxDistance exampleRide

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_correct_l756_75672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pinning_l756_75674

-- Define a square as a pair of points (bottom-left and top-right corners)
def Square := ℝ × ℝ × ℝ × ℝ

-- Define a function to check if a point is inside a square
def pointInSquare (p : ℝ × ℝ) (s : Square) : Prop :=
  let (x, y) := p
  let (x1, y1, x2, y2) := s
  x1 ≤ x ∧ x ≤ x2 ∧ y1 ≤ y ∧ y ≤ y2

-- Theorem statement
theorem square_pinning 
  (squares : Finset Square) 
  (h_nonempty : squares.Nonempty) 
  (h_identical : ∀ s1 s2, s1 ∈ squares → s2 ∈ squares → 
    ∃ a b, s1 = (a, b, a + 1, b + 1) ∧ s2 = (a, b, a + 1, b + 1)) 
  (h_parallel : ∀ s, s ∈ squares → ∃ a b, s = (a, b, a + 1, b + 1)) :
  ∃ pins : Finset (ℝ × ℝ), 
    (∀ s, s ∈ squares → ∃! p, p ∈ pins ∧ pointInSquare p s) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pinning_l756_75674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_gravity_shift_center_of_gravity_shift_60_l756_75610

/-- Represents a uniform straight rod -/
structure UniformRod where
  length : ℝ

/-- The center of gravity of a uniform rod -/
noncomputable def centerOfGravity (rod : UniformRod) : ℝ :=
  rod.length / 2

/-- The new rod after cutting off a piece -/
def cutRod (rod : UniformRod) (s : ℝ) : UniformRod :=
  { length := rod.length - s }

/-- Theorem: The shift in center of gravity after cutting a piece of length s -/
theorem center_of_gravity_shift (rod : UniformRod) (s : ℝ) 
    (h1 : 0 < s) (h2 : s < rod.length) :
    |centerOfGravity rod - centerOfGravity (cutRod rod s)| = s / 2 := by
  sorry

/-- Corollary: For s = 60, the center of gravity shifts by 30 -/
theorem center_of_gravity_shift_60 (rod : UniformRod) 
    (h : 60 < rod.length) :
    |centerOfGravity rod - centerOfGravity (cutRod rod 60)| = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_gravity_shift_center_of_gravity_shift_60_l756_75610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_second_derivative_equals_function_l756_75640

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_second_derivative_equals_function :
  ∀ x : ℝ, (deriv (deriv f)) x = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_second_derivative_equals_function_l756_75640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_a_and_b_l756_75638

theorem smallest_sum_of_a_and_b : 
  ∀ a b : ℕ, 
  a > 0 → b > 0 →
  (3^10 * 5^3 * 2^6 : ℕ) = a^b → 
  (∀ c d : ℕ, c > 0 → d > 0 → (3^10 * 5^3 * 2^6 : ℕ) = c^d → a + b ≤ c + d) → 
  a + b = 77762 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_a_and_b_l756_75638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_speed_inner_ring_optimal_train_distribution_l756_75684

/-- Represents a circular subway line with inner and outer rings -/
structure SubwayLine where
  ring_length : ℝ
  inner_speed : ℝ
  outer_speed : ℝ
  total_trains : ℕ

/-- Calculates the maximum waiting time for a ring given its speed and number of trains -/
noncomputable def max_waiting_time (ring_length : ℝ) (speed : ℝ) (num_trains : ℕ) : ℝ :=
  (ring_length / (speed * (num_trains : ℝ))) * 60

/-- Theorem for the minimum speed required on the inner ring -/
theorem min_speed_inner_ring (sl : SubwayLine) (inner_trains : ℕ) (max_wait : ℝ) :
  max_waiting_time sl.ring_length sl.inner_speed inner_trains ≤ max_wait →
  sl.inner_speed ≥ (sl.ring_length * 60) / (max_wait * (inner_trains : ℝ)) := by
  sorry

/-- Theorem for the optimal distribution of trains between inner and outer rings -/
theorem optimal_train_distribution (sl : SubwayLine) (inner_trains : ℕ) :
  sl.ring_length = 30 ∧
  sl.inner_speed = 25 ∧
  sl.outer_speed = 30 ∧
  sl.total_trains = 18 ∧
  inner_trains = 10 →
  let outer_trains := sl.total_trains - inner_trains
  |max_waiting_time sl.ring_length sl.inner_speed inner_trains -
   max_waiting_time sl.ring_length sl.outer_speed outer_trains| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_speed_inner_ring_optimal_train_distribution_l756_75684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shaded_triangles_l756_75664

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℕ

/-- Represents a division of a larger triangle into smaller triangles -/
structure TriangleDivision where
  large_triangle : EquilateralTriangle
  small_triangle_side : ℕ

/-- Calculates the number of vertices in a triangle division -/
def num_vertices (td : TriangleDivision) : ℕ :=
  let n := td.large_triangle.side_length / td.small_triangle_side + 1
  n * (n + 1) / 2

/-- Helper function to represent the set of vertices in a triangle division -/
def set_of_vertices (td : TriangleDivision) : Set (ℕ × ℕ) := sorry

/-- Helper function to check if a vertex is covered by a number of shaded triangles -/
def v_covered_by (n : ℕ) (v : ℕ × ℕ) : Prop := sorry

/-- Theorem: Minimum number of shaded triangles in a specific triangle division -/
theorem min_shaded_triangles (td : TriangleDivision) 
  (h1 : td.large_triangle.side_length = 8)
  (h2 : td.small_triangle_side = 1) :
  ∃ (n : ℕ), n = 15 ∧ 
    (∀ m : ℕ, m < n → ¬(∀ v, v ∈ set_of_vertices td → v_covered_by m v)) ∧
    (∀ v, v ∈ set_of_vertices td → v_covered_by n v) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shaded_triangles_l756_75664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l756_75663

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Conjugate hyperbola -/
def conjugate (h : Hyperbola) : Hyperbola where
  a := h.b
  b := h.a
  h_pos_a := h.h_pos_b
  h_pos_b := h.h_pos_a

/-- Focal distance of a hyperbola -/
noncomputable def focalDistance (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

theorem hyperbola_properties (h : Hyperbola) :
  (eccentricity h * eccentricity (conjugate h) ≥ 2) ∧
  (∃ (r : ℝ), r > 0 ∧ focalDistance h = r ∧ focalDistance (conjugate h) = r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l756_75663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_l756_75605

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 10 + y^2 = 1

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 8 - y^2 = 1

-- Define the common foci
def are_common_foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ 
  (∀ x y, is_on_ellipse x y → (x - F₁.1)^2 + (y - F₁.2)^2 + (x - F₂.1)^2 + (y - F₂.2)^2 = 4 * c^2) ∧
  (∀ x y, is_on_hyperbola x y → |(x - F₁.1)^2 + (y - F₁.2)^2 - ((x - F₂.1)^2 + (y - F₂.2)^2)| = 4 * c^2)

-- Define the intersection point
def is_intersection_point (P : ℝ × ℝ) : Prop :=
  is_on_ellipse P.1 P.2 ∧ is_on_hyperbola P.1 P.2

-- Define the circumradius function
def circumradius (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem circumcircle_radius 
  (F₁ F₂ P : ℝ × ℝ) 
  (h₁ : are_common_foci F₁ F₂) 
  (h₂ : is_intersection_point P) : 
  circumradius F₁ F₂ P = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_l756_75605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ratio_theorem_l756_75688

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (2*x)

noncomputable def f_iter : ℕ → (ℝ → ℝ)
| 0 => id
| (n+1) => f ∘ (f_iter n)

theorem f_ratio_theorem (n : ℕ) (x : ℝ) 
  (hx : x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1) : 
  (f_iter n x) / (f_iter (n+1) x) = 
    1 + 1 / (f (((x+1)/(x-1))^(2^n))) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ratio_theorem_l756_75688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_m_l756_75685

/-- Given a hyperbola with equation x²/16 - y²/m = 1 and eccentricity 5/4, prove that m = 9 -/
theorem hyperbola_eccentricity_m (m : ℝ) :
  (∀ x y : ℝ, x^2/16 - y^2/m = 1) →  -- Hyperbola equation
  (let a := 4;
   let c := (5/4) * a;
   c^2 = a^2 + m) →  -- Eccentricity condition
  m = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_m_l756_75685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l756_75602

-- Define the side length of the square
noncomputable def side_length : ℝ := 12

-- Define the area of the square
noncomputable def square_area : ℝ := side_length ^ 2

-- Define the radius of each quarter circle
noncomputable def quarter_circle_radius : ℝ := side_length / 2

-- Define the area of one full circle
noncomputable def full_circle_area : ℝ := Real.pi * quarter_circle_radius ^ 2

-- Define the area covered by three quarter circles
noncomputable def quarter_circles_area : ℝ := 3 * (full_circle_area / 4)

-- Theorem: The shaded area is equal to 144 - 27π
theorem shaded_area_calculation :
  square_area - quarter_circles_area = 144 - 27 * Real.pi := by
  -- Expand definitions
  unfold square_area quarter_circles_area full_circle_area quarter_circle_radius side_length
  -- Simplify expressions
  simp [Real.pi]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l756_75602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_Q_l756_75627

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 15 = 0

noncomputable def C₂ (α : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos α, Real.sqrt 2 * Real.sin α)

-- Define the point Q
noncomputable def Q : ℝ × ℝ := C₂ (3 * Real.pi / 4)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem min_distance_C₁_Q :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 13 - 1 ∧
  ∀ (p : ℝ × ℝ), C₁ p.1 p.2 → distance p Q ≥ min_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_Q_l756_75627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l756_75613

theorem differential_equation_solution (y : ℝ → ℝ) :
  (∀ x, (deriv (deriv y)) x + 4 * (deriv y x) + 5 * (y x) = 0) →
  y 0 = -3 →
  (deriv y) 0 = 0 →
  ∃ C₁ C₂ : ℝ, ∀ x, y x = Real.exp (-2 * x) * (C₁ * Real.cos x + C₂ * Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l756_75613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_diff_gt_one_is_seven_eighths_l756_75619

-- Define the coin flip process
inductive CoinFlip
| HH
| HT
| Uniform (x : ℝ)

-- Define the probability measure on the coin flip process
noncomputable def P : CoinFlip → ℝ
| CoinFlip.HH => 1/4
| CoinFlip.HT => 1/4
| CoinFlip.Uniform _ => 1/2

-- Define the random variable X
noncomputable def X : CoinFlip → ℝ
| CoinFlip.HH => 0
| CoinFlip.HT => 2
| CoinFlip.Uniform x => x

-- Define the joint distribution of (X, Y)
def JointDist := CoinFlip × CoinFlip

-- Define the probability that |X - Y| > 1
noncomputable def prob_diff_gt_one : ℝ :=
  sorry -- Placeholder for the actual probability calculation

-- The theorem to prove
theorem prob_diff_gt_one_is_seven_eighths : 
  prob_diff_gt_one = 7/8 := by
  sorry -- Placeholder for the proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_diff_gt_one_is_seven_eighths_l756_75619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_technical_personnel_max_m_value_l756_75618

/-- Represents the original average annual investment per person in thousands of yuan -/
def a : ℝ := sorry

/-- Assumption that a is positive -/
axiom ha : a > 0

/-- Represents the number of technical personnel after adjustment -/
def x : ℝ := sorry

/-- Represents the adjustment factor for technical personnel's investment -/
def m : ℝ := sorry

/-- Theorem 1: The maximum value of x that satisfies the inequality is 75 -/
theorem max_technical_personnel :
  (∀ x, 0 ≤ x → x ≤ 100 → (100 - x) * (1 + 4 * x / 100) * a ≥ 100 * a) →
  (∀ x, x > 75 → (100 - x) * (1 + 4 * x / 100) * a < 100 * a) :=
by sorry

/-- Theorem 2: The maximum value of m is 7 and occurs when x = 50 -/
theorem max_m_value :
  (∀ x, 0 ≤ x → x ≤ 100 → (100 - x) * (1 + x / 25) ≥ (m - 2 * x / 25) * x) →
  m ≤ 7 ∧ (m = 7 → x = 50) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_technical_personnel_max_m_value_l756_75618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_symmetry_l756_75693

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The property that P(m) + P(n) = 0 has infinitely many integer solutions -/
def HasInfinitelySolutions (P : RealPolynomial) : Prop :=
  ∀ k : ℕ, ∃ m n : ℤ, m ≠ n ∧ P (m : ℝ) + P (n : ℝ) = 0 ∧ m > k ∧ n > k

/-- The property that the graph of P has a center of symmetry -/
def HasCenterOfSymmetry (P : RealPolynomial) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, P (a + x) = -P (a - x)

/-- Main theorem: If a real polynomial has infinitely many integer solutions
    to P(m) + P(n) = 0, then its graph has a center of symmetry -/
theorem polynomial_symmetry (P : RealPolynomial) 
  (h : HasInfinitelySolutions P) : HasCenterOfSymmetry P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_symmetry_l756_75693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_dist_variance_and_std_dev_l756_75639

/-- Exponential distribution with parameter λ > 0 -/
structure ExpDist where
  lambda : ℝ
  lambda_pos : lambda > 0

/-- Probability density function for exponential distribution -/
noncomputable def pdf (d : ExpDist) (x : ℝ) : ℝ :=
  if x ≥ 0 then d.lambda * Real.exp (-d.lambda * x) else 0

/-- Variance of exponential distribution -/
noncomputable def variance (d : ExpDist) : ℝ := 1 / d.lambda^2

/-- Standard deviation of exponential distribution -/
noncomputable def std_dev (d : ExpDist) : ℝ := 1 / d.lambda

theorem exp_dist_variance_and_std_dev (d : ExpDist) :
  variance d = 1 / d.lambda^2 ∧ std_dev d = 1 / d.lambda := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_dist_variance_and_std_dev_l756_75639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_sum_l756_75635

theorem complex_fraction_sum (a b : ℝ) : (Complex.I - 2) / (1 + Complex.I) = Complex.ofReal a + Complex.I * Complex.ofReal b → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_sum_l756_75635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_expression_value_l756_75622

/-- Given that ax³ + bx + 5 = -9 when x = -2, prove that ax³ + bx + 5 = 19 when x = 2 -/
theorem algebraic_expression_value (a b : ℝ) : 
  (fun x ↦ a * x^3 + b * x + 5) (-2) = -9 → (fun x ↦ a * x^3 + b * x + 5) 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_expression_value_l756_75622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_two_digit_multiples_of_eight_l756_75662

theorem arithmetic_mean_two_digit_multiples_of_eight :
  let multiples := (List.range 100).filter (fun n => n ≥ 10 ∧ n % 8 = 0)
  (multiples.sum / multiples.length : ℚ) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_two_digit_multiples_of_eight_l756_75662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l756_75641

/-- A point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The original point P -/
noncomputable def P (m : ℝ) : Point :=
  ⟨2 - m, 1/2 * m⟩

/-- The symmetric point of P about the x-axis -/
noncomputable def P_sym (m : ℝ) : Point :=
  ⟨2 - m, -(1/2 * m)⟩

/-- Condition: P_sym is in the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem: The range of m for the given conditions -/
theorem m_range :
  ∀ m : ℝ, in_fourth_quadrant (P_sym m) ↔ 0 < m ∧ m < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l756_75641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_f_eq_g_l756_75689

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - x) * Real.exp x - 1

noncomputable def g (t m : ℝ) (x : ℝ) : ℝ := (x - t)^2 + (Real.log x - m / t)^2

theorem min_m_for_f_eq_g :
  ∃ (m : ℝ), m = -1 / Real.exp 1 ∧
  ∀ (m' : ℝ), (∃ (x₁ : ℝ) (x₂ : ℝ), x₂ > 0 ∧ f 1 x₁ = g t m' x₂) →
  m' ≥ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_f_eq_g_l756_75689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_distance_for_average_speed_l756_75647

/-- Calculates the additional distance required to achieve a desired average speed -/
noncomputable def additionalDistance (initialDist : ℝ) (initialSpeed : ℝ) (secondSpeed : ℝ) (desiredAvgSpeed : ℝ) : ℝ :=
  let initialTime := initialDist / initialSpeed
  let t := (desiredAvgSpeed * initialTime - initialDist) / (secondSpeed - desiredAvgSpeed)
  secondSpeed * t

theorem additional_distance_for_average_speed :
  additionalDistance 24 16 64 40 = 96 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_distance_for_average_speed_l756_75647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_bending_l756_75648

/-- Given a wire that can be bent into either a circle or a square, 
    if the area of the square is 7737.769850454057 cm², 
    then the radius of the circle formed by the same wire is approximately 56 cm. -/
theorem wire_bending (square_area : ℝ) (circle_radius : ℝ) : 
  square_area = 7737.769850454057 → 
  abs (circle_radius - 56) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_bending_l756_75648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_propositions_l756_75614

noncomputable section

open Real

def f (x : ℝ) := sin (2 * x + π / 3)
def g (x : ℝ) := cos (x + π / 3)
def h (x : ℝ) := tan (x + π / 3)
def k (x : ℝ) := 3 * sin (2 * x + π / 3)
def m (x : ℝ) := 3 * sin (2 * x)

theorem trigonometric_propositions :
  (∃ (S : Finset (Fin 4)), S.card = 2 ∧
    (1 ∈ S ↔ ∀ x y, -π/3 < x ∧ x < y ∧ y < π/6 → f x < f y) ∧
    (2 ∈ S ↔ ∀ x, g (π/3 - x) = g (π/3 + x)) ∧
    (3 ∈ S ↔ ∀ x, h (π/3 - x) = h (π/3 + x)) ∧
    (4 ∈ S ↔ ∀ x, k (x - π/6) = m x)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_propositions_l756_75614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_fraction_of_grid_l756_75698

noncomputable def A : ℝ × ℝ := (2, 2)
noncomputable def B : ℝ × ℝ := (6, 2)
noncomputable def C : ℝ × ℝ := (5, 6)

noncomputable def grid_width : ℝ := 7
noncomputable def grid_height : ℝ := 6

noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

theorem triangle_fraction_of_grid :
  (triangle_area A B C) / (grid_width * grid_height) = 4/21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_fraction_of_grid_l756_75698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_professors_son_age_l756_75690

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The theorem statement -/
theorem professors_son_age 
  (f : IntPolynomial) 
  (A : ℕ+) 
  (P : ℕ) 
  (h1 : f.eval (A : ℤ) = A) 
  (h2 : f.eval 0 = P) 
  (h3 : Nat.Prime P) 
  (h4 : P > A) : 
  A = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_professors_son_age_l756_75690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l756_75658

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  (1 - 2*x) / Real.sqrt (x*(1-x)) + 
  (1 - 2*y) / Real.sqrt (y*(1-y)) + 
  (1 - 2*z) / Real.sqrt (z*(1-z)) ≥ 
  Real.sqrt (x/(1-x)) + Real.sqrt (y/(1-y)) + Real.sqrt (z/(1-z)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l756_75658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_real_solution_cosine_odd_iff_phase_l756_75671

/-- Quadratic inequality has ℝ as its solution set iff a > 0 and discriminant ≤ 0 -/
theorem quadratic_inequality_real_solution (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4*a*c ≤ 0) := by sorry

/-- Cosine function is odd iff its phase is π/2 + kπ for some integer k -/
theorem cosine_odd_iff_phase (A ω φ : ℝ) (hA : A ≠ 0) :
  (∀ x, A * Real.cos (ω * x + φ) = -A * Real.cos (ω * (-x) + φ)) ↔
  ∃ k : ℤ, φ = Real.pi / 2 + k * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_real_solution_cosine_odd_iff_phase_l756_75671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_digit_of_one_seventh_l756_75620

/-- The decimal representation of 1/7 as a sequence of digits -/
def oneSeventhDecimal (n : ℕ) : ℕ :=
  match n % 6 with
  | 0 => 1
  | 1 => 4
  | 2 => 2
  | 3 => 8
  | 4 => 5
  | _ => 7

/-- The 100th digit after the decimal point in the decimal representation of 1/7 is 8 -/
theorem hundredth_digit_of_one_seventh : oneSeventhDecimal 99 = 8 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_digit_of_one_seventh_l756_75620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_four_plus_eight_n_plus_eleven_consecutive_product_l756_75643

/-- A function that represents the expression n^4 + 8n + 11 -/
def f (n : ℤ) : ℤ := n^4 + 8*n + 11

/-- A predicate that checks if a number is a product of two or more consecutive integers -/
def is_product_of_consecutive (m : ℤ) : Prop :=
  ∃ (k : ℤ) (l : ℕ), l ≥ 2 ∧ m = (Finset.range l).prod (λ i => k + i)

/-- Theorem stating that n^4 + 8n + 11 is a product of two or more consecutive integers 
    if and only if n = 1 -/
theorem n_four_plus_eight_n_plus_eleven_consecutive_product (n : ℤ) :
  is_product_of_consecutive (f n) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_four_plus_eight_n_plus_eleven_consecutive_product_l756_75643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumscribed_circle_diameter_l756_75611

open Real

-- Define a triangle
structure Triangle where
  side : ℝ
  opposite_angle : ℝ

-- Define the diameter of the circumscribed circle
noncomputable def circumscribed_circle_diameter (t : Triangle) : ℝ :=
  t.side / (sin t.opposite_angle)

-- Theorem statement
theorem triangle_circumscribed_circle_diameter 
  (t : Triangle) 
  (h1 : t.side = 14)
  (h2 : t.opposite_angle = π/4) : 
  circumscribed_circle_diameter t = 14 * sqrt 2 := by
  -- Unfold the definition of circumscribed_circle_diameter
  unfold circumscribed_circle_diameter
  -- Rewrite using the hypotheses
  rw [h1, h2]
  -- Simplify the expression
  simp [sin_pi_div_four]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumscribed_circle_diameter_l756_75611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_problem_l756_75695

/-- Prove that if vectors a = (1,1) and b = (4,x), and c = a + b and d = 2a + b,
    and c is parallel to d, then x = 4. -/
theorem parallel_vectors_problem (x : ℝ) : 
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (4, x)
  let c : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
  let d : ℝ × ℝ := (2 * a.1 + b.1, 2 * a.2 + b.2)
  (∃ (k : ℝ), c = k • d) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_problem_l756_75695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_finite_moves_with_determined_winner_l756_75625

/-- Represents the state of a coin (Heads or Tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents the game state as a list of coin states -/
def GameState := List CoinState

/-- Represents a player (FirstPlayer or SecondPlayer) -/
inductive Player
| FirstPlayer
| SecondPlayer

/-- Returns the opposite player -/
def Player.opposite : Player → Player
| FirstPlayer => SecondPlayer
| SecondPlayer => FirstPlayer

/-- Represents a single move in the game -/
structure Move where
  index : Nat
  newState : GameState

/-- Determines if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  move.index < state.length ∧
  state.get? move.index = some CoinState.Heads ∧
  move.newState.length = state.length

/-- Determines if the game is over (no more valid moves) -/
def isGameOver (state : GameState) : Prop :=
  ∀ i, i < state.length → state.get? i = some CoinState.Tails

/-- The main theorem stating that the game ends in finite moves and the winner is determined by the last coin -/
theorem game_ends_finite_moves_with_determined_winner (initialState : GameState) :
  ∃ (finalState : GameState) (moves : Nat) (winner : Player),
    isGameOver finalState ∧
    (moves < 2^initialState.length) ∧
    (winner = Player.FirstPlayer ↔ initialState.get? (initialState.length - 1) = some CoinState.Heads) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_finite_moves_with_determined_winner_l756_75625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l756_75667

theorem cos_minus_sin_value (x : ℝ) 
  (h1 : Real.sin x * Real.cos x = 1/8)
  (h2 : π/4 < x)
  (h3 : x < π/2) :
  Real.cos x - Real.sin x = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l756_75667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l756_75680

/-- The volume of a cylindrical tube formed from a rectangular sheet -/
noncomputable def cylinderVolume (width : ℝ) (height : ℝ) (tubeHeight : ℝ) : ℝ :=
  (width^2 + height^2) * tubeHeight / (4 * Real.pi)

/-- The difference in volumes of two cylindrical tubes -/
noncomputable def volumeDifference (width : ℝ) (height : ℝ) : ℝ :=
  |cylinderVolume width height height - cylinderVolume width height width|

theorem cylinder_volume_difference (width height : ℝ) 
  (h1 : width = 7) (h2 : height = 10) :
  Real.pi * volumeDifference width height = 111.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l756_75680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_undefined_inverse_l756_75608

theorem smallest_undefined_inverse (b : ℕ) : 
  (∀ m : ℕ, m < b → (∃ x : ℕ, x * m % 84 = 1 ∨ ∃ y : ℕ, y * m % 90 = 1)) → 
  (∀ x : ℕ, x * b % 84 ≠ 1) → 
  (∀ y : ℕ, y * b % 90 ≠ 1) → 
  b = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_undefined_inverse_l756_75608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_correct_l756_75668

noncomputable def calculate_total_spent (initial_backpack_price initial_binder_price backpack_increase binder_decrease num_binders discount : ℚ) : ℚ :=
  let new_backpack_price := initial_backpack_price + backpack_increase
  let new_binder_price := initial_binder_price - binder_decrease
  let binders_to_pay := min num_binders (num_binders / 2 + 1)
  let subtotal := new_backpack_price + binders_to_pay * new_binder_price
  subtotal * (1 - discount / 100)

theorem total_spent_correct :
  calculate_total_spent 50 20 5 2 3 10 = 819/10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_correct_l756_75668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_proof_l756_75673

/-- The capacity of a tank with a leak and an inlet pipe --/
noncomputable def tank_capacity (leak_empty_time inlet_rate combined_empty_time : ℝ) : ℝ :=
  5760 / 7

/-- Theorem stating the capacity of the tank given specific conditions --/
theorem tank_capacity_proof 
  (leak_empty_time : ℝ) 
  (inlet_rate : ℝ) 
  (combined_empty_time : ℝ) 
  (h1 : leak_empty_time = 6)
  (h2 : inlet_rate = 4)
  (h3 : combined_empty_time = 8) :
  tank_capacity leak_empty_time inlet_rate combined_empty_time = 5760 / 7 :=
by
  -- Unfold the definition of tank_capacity
  unfold tank_capacity
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_proof_l756_75673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_coincidence_range_l756_75656

noncomputable section

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + x + a else -1/x

-- Define the derivative of f(x)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2*x + 1 else 1/x^2

-- Theorem statement
theorem tangent_coincidence_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧
    (f' a x₁ = f' a x₂) ∧
    (f a x₁ - f' a x₁ * x₁ = f a x₂ - f' a x₂ * x₂)) →
  -2 < a ∧ a < 1/4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_coincidence_range_l756_75656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l756_75621

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 1/a)*x - Real.log x

-- State the theorem
theorem f_lower_bound (a : ℝ) (h : a < 0) :
  ∀ x > 0, f a x ≥ (1 - 2*a)*(a + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l756_75621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_range_l756_75692

noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + abs x

theorem g_inequality_range :
  {x : ℝ | g (2 * x - 1) < g 3} = Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_range_l756_75692
