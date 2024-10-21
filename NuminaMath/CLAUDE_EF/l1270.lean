import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l1270_127063

/-- The circle on which point M lies -/
def circle_eq (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9

/-- The line to which we're calculating the distance -/
def line_eq (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0

/-- The shortest distance from a point on the circle to the line is 2 -/
theorem shortest_distance_circle_to_line :
  ∀ (M : ℝ × ℝ), circle_eq M.1 M.2 → 
    (∃ (d : ℝ), d = 2 ∧ 
      ∀ (P : ℝ × ℝ), line_eq P.1 P.2 → 
        d ≤ Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)) :=
by
  sorry

#check shortest_distance_circle_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l1270_127063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1270_127007

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 * Real.cos x - Real.sin x) * Real.sin x

def is_smallest_positive_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, T' < T → ∃ x, f (x + T') ≠ f x)

def is_monotonic_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_properties :
  (is_smallest_positive_period Real.pi f) ∧
  (∀ k : ℤ, is_monotonic_increasing_on f (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≤ 1 / 2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≥ 0) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x = 1 / 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1270_127007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pete_bottles_to_return_l1270_127094

/-- The number of bottles Pete needs to return to pay off his bike -/
def bottles_to_return (owed : ℚ) (twenty_bills : ℕ) (ten_bills : ℕ) (bottle_value : ℚ) : ℕ :=
  let current_money := (twenty_bills * 20 + ten_bills * 10 : ℚ)
  let remaining := owed - current_money
  (remaining / bottle_value).ceil.toNat

/-- Proof that Pete needs to return 20 bottles -/
theorem pete_bottles_to_return :
  bottles_to_return 90 2 4 (1/2) = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pete_bottles_to_return_l1270_127094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_fourth_l1270_127047

theorem sin_alpha_plus_pi_fourth (α : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (2 * α) = 1 / 2) : 
  Real.sin (α + π / 4) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_fourth_l1270_127047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_completes_work_in_4_days_l1270_127045

/-- The number of days it takes A to complete the work -/
noncomputable def days_A : ℝ := sorry

/-- B can complete the work in 8 days -/
noncomputable def work_rate_B : ℝ := 1 / 8

/-- C can complete the work in 8 days -/
noncomputable def work_rate_C : ℝ := 1 / 8

/-- A, B, and C together can complete the work in 2 days -/
noncomputable def combined_work_rate : ℝ := 1 / 2

/-- Theorem stating that A can complete the work in 4 days -/
theorem A_completes_work_in_4_days : days_A = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_completes_work_in_4_days_l1270_127045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_area_l1270_127032

/-- Represents a right, square-based pyramid -/
structure SquarePyramid where
  base_edge : ℝ
  lateral_edge : ℝ

/-- Calculates the total area of the four triangular faces of a square pyramid -/
noncomputable def total_triangular_area (p : SquarePyramid) : ℝ :=
  let altitude := Real.sqrt (p.lateral_edge ^ 2 - (p.base_edge / 2) ^ 2)
  4 * (1 / 2 * p.base_edge * altitude)

/-- Theorem stating that the total area of the four triangular faces of a right,
    square-based pyramid with base edges of 6 units and lateral edges of 5 units
    is equal to 48 square units -/
theorem square_pyramid_area :
  let p : SquarePyramid := { base_edge := 6, lateral_edge := 5 }
  total_triangular_area p = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_area_l1270_127032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_marbles_count_l1270_127095

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := sorry

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have -/
def total_marbles : ℕ := 12

/-- Theorem: Mary's yellow marbles equals the total minus Joan's marbles -/
theorem mary_marbles_count : mary_marbles = total_marbles - joan_marbles := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_marbles_count_l1270_127095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_satisfies_conditions_l1270_127031

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents the equation of a parabola in general form -/
structure ParabolaEquation where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- Checks if a point lies on the parabola -/
def lieOnParabola (p : Point2D) (eq : ParabolaEquation) : Prop :=
  eq.a * p.x^2 + eq.b * p.x * p.y + eq.c * p.y^2 + eq.d * p.x + eq.e * p.y + eq.f = 0

/-- Checks if the axis of symmetry is parallel to the y-axis -/
def axisParallelToY (eq : ParabolaEquation) : Prop :=
  eq.b = 0 ∧ eq.c = 0

/-- Checks if the vertex lies on the x-axis -/
def vertexOnXAxis (eq : ParabolaEquation) : Prop :=
  ∃ x : ℝ, eq.a * x^2 + eq.d * x + eq.f = 0

/-- Theorem: The given equation represents a parabola satisfying all conditions -/
theorem parabola_satisfies_conditions : ∃ (eq : ParabolaEquation),
  eq.a = 8 ∧ eq.b = 0 ∧ eq.c = 0 ∧ eq.d = -80 ∧ eq.e = -9 ∧ eq.f = 200 ∧
  lieOnParabola ⟨2, 8⟩ eq ∧
  axisParallelToY eq ∧
  vertexOnXAxis eq ∧
  eq.a > 0 ∧
  Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd (eq.a.natAbs) (eq.b.natAbs)) (eq.c.natAbs)) (eq.d.natAbs)) (eq.e.natAbs)) (eq.f.natAbs) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_satisfies_conditions_l1270_127031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_circle_outside_rectangle_l1270_127041

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a circle with radius -/
structure Circle where
  radius : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  r.width * r.height

/-- Calculates the area of a circle -/
noncomputable def circleArea (c : Circle) : ℝ :=
  Real.pi * c.radius^2

/-- Theorem: Area of region inside circle but outside rectangle -/
theorem area_inside_circle_outside_rectangle
  (rect : Rectangle)
  (circ : Circle)
  (h1 : rect.width = 6)
  (h2 : rect.height = 5)
  (h3 : circ.radius^2 = rect.width^2 + rect.height^2) :
  circleArea circ - rectangleArea rect = 61 * Real.pi - 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_circle_outside_rectangle_l1270_127041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_fourth_quadrant_exp_shift_function_fixed_point_l1270_127001

-- Define a power function
noncomputable def powerFunction (n : ℝ) (x : ℝ) : ℝ := x^n

-- Define the exponential function with vertical shift
noncomputable def expShiftFunction (a : ℝ) (x : ℝ) : ℝ := a^(x+1) - 2

-- Theorem for power function property
theorem power_function_not_in_fourth_quadrant (n : ℝ) (x : ℝ) (h1 : n ≠ 0) :
  ¬(x > 0 ∧ powerFunction n x < 0) := by
  sorry

-- Theorem for exponential function with vertical shift passing through (-1, -1)
theorem exp_shift_function_fixed_point (a : ℝ) (h2 : a > 0) (h3 : a ≠ 1) :
  expShiftFunction a (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_not_in_fourth_quadrant_exp_shift_function_fixed_point_l1270_127001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1270_127043

def f (x : ℝ) : ℝ := sorry

theorem domain_of_f :
  {x : ℝ | x ≤ 0 ∧ x ≠ -1} = {x : ℝ | ∃ y, f x = y} :=
by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1270_127043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l1270_127030

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.a * r.b

/-- The diagonal length of a rectangle -/
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ := Real.sqrt (r.a^2 + r.b^2)

/-- Two rectangles are similar if their side ratios are equal -/
def similar (r1 r2 : Rectangle) : Prop :=
  r1.a / r1.b = r2.a / r2.b

theorem rectangle_area_theorem (r1 r2 : Rectangle) :
  r1.a = 4 ∧ r1.area = 32 ∧ similar r1 r2 ∧ r2.diagonal = 20 →
  r2.area = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_theorem_l1270_127030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_difference_is_42_l1270_127070

-- Define the points for lines l and m
def l_point1 : ℝ × ℝ := (0, 4)
def l_point2 : ℝ × ℝ := (2, 0)
def m_point1 : ℝ × ℝ := (0, 1)
def m_point2 : ℝ × ℝ := (5, 0)

-- Define the slope of a line given two points
noncomputable def line_slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the y-intercept of a line given a point and slope
noncomputable def y_intercept (p : ℝ × ℝ) (m : ℝ) : ℝ :=
  p.2 - m * p.1

-- Define the x-coordinate of a line at a given y-value
noncomputable def x_at_y (m b y : ℝ) : ℝ :=
  (y - b) / m

-- Theorem statement
theorem x_coordinate_difference_is_42 :
  let l_slope := line_slope l_point1 l_point2
  let m_slope := line_slope m_point1 m_point2
  let l_intercept := y_intercept l_point1 l_slope
  let m_intercept := y_intercept m_point1 m_slope
  let l_x := x_at_y l_slope l_intercept 10
  let m_x := x_at_y m_slope m_intercept 10
  |l_x - m_x| = 42 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_difference_is_42_l1270_127070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l1270_127008

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

def is_acute_triangle (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

def satisfies_conditions (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A ∧
  t.c = Real.sqrt 7 ∧
  (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2

-- State the theorem
theorem triangle_side_sum (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : is_acute_triangle t)
  (h3 : satisfies_conditions t) :
  t.a + t.b = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l1270_127008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_number_l1270_127021

def count_occurrences (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).count d

theorem existence_of_special_number : ∃ n : ℕ, 
  (n % 2020 = 0) ∧ 
  (∀ d₁ d₂ : Fin 10, (count_occurrences n d₁.val = count_occurrences n d₂.val)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_number_l1270_127021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_pi_over_16_l1270_127016

/-- The probability of a point being within 2 units of the origin when randomly selected from a square with vertices at (±4, ±4) -/
noncomputable def probability_within_circle (square_side : ℝ) (circle_radius : ℝ) : ℝ :=
  (Real.pi * circle_radius^2) / (square_side^2)

theorem probability_is_pi_over_16 :
  probability_within_circle 8 2 = Real.pi / 16 := by
  unfold probability_within_circle
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_pi_over_16_l1270_127016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l1270_127003

noncomputable def f (x : ℝ) := 2^(x-1) + x - 5

theorem zero_point_in_interval :
  ∃! x₀ : ℝ, x₀ ∈ Set.Ioo 2 3 ∧ f x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l1270_127003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l1270_127015

open Real

theorem max_value_trig_expression :
  (∀ x y z : ℝ,
    (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) ≤ 4.5) ∧
  (∃ x y z : ℝ,
    (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) = 4.5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l1270_127015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_properties_l1270_127080

/-- The function f(x) as defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m / x + (1 / 2) * Real.log x - 1

/-- Theorem stating the range of m and the inequality for the zeros of f -/
theorem function_zeros_properties (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : f m x₁ = 0) 
  (h₂ : f m x₂ = 0) 
  (h₃ : x₁ < x₂) :
  (0 < m ∧ m < Real.exp 1 / 2) ∧ 
  (1 / x₁ + 1 / x₂ > 2 / Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_properties_l1270_127080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_RSUTV_value_l1270_127035

-- Define points and distances
noncomputable def P : ℝ × ℝ := sorry
noncomputable def Q : ℝ × ℝ := sorry
noncomputable def R : ℝ × ℝ := sorry
noncomputable def S : ℝ × ℝ := sorry
noncomputable def T : ℝ × ℝ := sorry
noncomputable def U : ℝ × ℝ := sorry
noncomputable def V : ℝ × ℝ := sorry

-- Define the radius of the circles
def radius : ℝ := 3

-- Define the distance PR
noncomputable def PR : ℝ := 3 * Real.sqrt 2

-- State that R is the midpoint of PQ
axiom R_midpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- State that RS is tangent to circle P
axiom RS_tangent_P : sorry

-- State that RT is tangent to circle Q
axiom RT_tangent_Q : sorry

-- State that UV is a common tangent to both circles
axiom UV_common_tangent : sorry

-- Define the area of the shaded region RSUTV
noncomputable def area_RSUTV : ℝ := sorry

-- Theorem statement
theorem area_RSUTV_value : 
  area_RSUTV = 36 * Real.sqrt 2 - 9 - 9 * Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_RSUTV_value_l1270_127035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_sqrt3_over_2_l1270_127002

theorem arcsin_sqrt3_over_2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_sqrt3_over_2_l1270_127002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_distribution_l1270_127097

variable (n m r : ℕ)

-- Define Stirling number of the second kind
def stirling_second (r n : ℕ) : ℕ := sorry

-- Define the function for the number of ways to distribute books
def number_of_ways_to_distribute_books (n m r : ℕ) : ℕ := sorry

-- Theorem statement
theorem library_book_distribution (h1 : n ≤ r) (h2 : r ≤ m) :
  number_of_ways_to_distribute_books n m r = n! * stirling_second r n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_distribution_l1270_127097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_composites_l1270_127053

/-- A sequence of positive integers -/
def IncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1)

/-- Property that every term after a certain point is a sum of two previous terms -/
def SumProperty (a : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, ∀ n, n > k → ∃ i j, i < n ∧ j < n ∧ a n = a i + a j

/-- A number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- The main theorem -/
theorem infinitely_many_composites
  (a : ℕ → ℕ)
  (h_incr : IncreasingSequence a)
  (h_sum : SumProperty a) :
  ∀ N : ℕ, ∃ n > N, IsComposite (a n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_composites_l1270_127053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1270_127081

noncomputable def f (x : ℝ) := Real.exp x - 1 / Real.exp x + x

theorem range_of_a (a : ℝ) :
  f (Real.log 2 * a) - f (Real.log (1/2) * a) ≤ 2 * f 1 →
  0 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1270_127081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_domain_and_oddness_l1270_127071

def α_set : Set ℚ := {-1, 1, 2, 3/5, 7/2}

def has_domain_reals (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, f x = y

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def power_function (α : ℚ) : ℝ → ℝ :=
  λ x ↦ Real.rpow x (α : ℝ)

theorem power_function_domain_and_oddness :
  ∀ α ∈ α_set, (has_domain_reals (power_function α) ∧ is_odd_function (power_function α)) ↔ α ∈ ({1, 3/5} : Set ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_domain_and_oddness_l1270_127071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_l1270_127073

noncomputable def f (A ω b x : ℝ) : ℝ := A * Real.sin (ω * x) + b

theorem sum_of_f (A ω b : ℝ) (h_A : A > 0) (h_ω : ω > 0) 
  (h_max : ∀ x, f A ω b x ≤ 2)
  (h_min : ∀ x, f A ω b x ≥ 0)
  (h_sym : ω = Real.pi / 2) :
  Finset.sum (Finset.range 2008) (λ i => f A ω b (i + 1 : ℝ)) = 2008 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_l1270_127073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_touch_points_four_circles_l1270_127017

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The maximum number of intersection points between a line and a circle is 2 -/
axiom max_intersections_line_circle : ∀ (l : Line) (c : Circle), ∃ (n : ℕ), n ≤ 2

/-- Four coplanar circles -/
noncomputable def four_circles : Finset Circle :=
  sorry

/-- A line passing through all four circles -/
noncomputable def passing_line : Line :=
  sorry

/-- The number of points where the line touches the circles -/
noncomputable def touch_points : ℕ :=
  sorry

/-- Theorem: The maximum number of points where a line can touch four coplanar circles is 8 -/
theorem max_touch_points_four_circles : 
  touch_points ≤ 8 ∧ ∃ (l : Line) (cs : Finset Circle), Finset.card cs = 4 ∧ touch_points = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_touch_points_four_circles_l1270_127017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_owners_without_scooters_l1270_127087

theorem bike_owners_without_scooters 
  (total : ℕ) 
  (bike_owners : ℕ) 
  (scooter_owners : ℕ)
  (set_of_adults : Finset ℕ)
  (bike_owners_set : Finset ℕ)
  (scooter_owners_set : Finset ℕ)
  (h_total : total = 400)
  (h_bike : bike_owners = 370)
  (h_scooter : scooter_owners = 75)
  (h_all_own : ∀ a ∈ set_of_adults, a ∈ bike_owners_set ∨ a ∈ scooter_owners_set)
  (h_bike_count : bike_owners_set.card = bike_owners)
  (h_scooter_count : scooter_owners_set.card = scooter_owners)
  (h_total_count : set_of_adults.card = total) :
  (bike_owners_set \ scooter_owners_set).card = 325 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_owners_without_scooters_l1270_127087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_regular_hours_limit_l1270_127090

/-- Represents the compensation structure and work hours of a bus driver --/
structure BusDriverCompensation where
  regularRate : ℚ
  overtimeRateMultiplier : ℚ
  totalCompensation : ℚ
  totalHours : ℚ

/-- Calculates the limit of regular hours given the compensation structure --/
noncomputable def calculateRegularHoursLimit (comp : BusDriverCompensation) : ℚ :=
  (comp.totalCompensation - comp.totalHours * comp.regularRate * comp.overtimeRateMultiplier) /
  (comp.regularRate * (1 - comp.overtimeRateMultiplier))

/-- Theorem stating that the limit of regular hours is 40 for the given conditions --/
theorem bus_driver_regular_hours_limit :
  let comp : BusDriverCompensation := {
    regularRate := 16,
    overtimeRateMultiplier := 7/4,
    totalCompensation := 976,
    totalHours := 52
  }
  calculateRegularHoursLimit comp = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_regular_hours_limit_l1270_127090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_period_is_four_l1270_127074

/-- Given a principal amount, interest rate, and interest earned, 
    calculate the number of years the money was invested. -/
noncomputable def investment_years (principal : ℝ) (rate : ℝ) (interest : ℝ) : ℝ :=
  (interest * 100) / (principal * rate)

/-- Theorem stating that for the given conditions, 
    the investment period is approximately 4 years. -/
theorem investment_period_is_four : 
  let principal : ℝ := 810
  let rate : ℝ := 4.783950617283951
  let interest : ℝ := 155
  ⌊investment_years principal rate interest⌋ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_period_is_four_l1270_127074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_walking_time_l1270_127027

/-- The time it takes for Maxwell and Brad to meet, given their speeds and the distance between their homes -/
noncomputable def meeting_time (distance : ℝ) (maxwell_speed : ℝ) (brad_speed : ℝ) (head_start : ℝ) : ℝ :=
  let t := (distance - maxwell_speed * head_start) / (maxwell_speed + brad_speed)
  t + head_start

/-- Theorem stating that under the given conditions, Maxwell walks for 8 hours before meeting Brad -/
theorem maxwell_walking_time :
  let distance : ℝ := 74
  let maxwell_speed : ℝ := 4
  let brad_speed : ℝ := 6
  let head_start : ℝ := 1
  meeting_time distance maxwell_speed brad_speed head_start = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_walking_time_l1270_127027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_l1270_127067

theorem book_pages : ∃ (x : ℕ),
  (x / 6 + 10 +
   (x - (x / 6 + 10)) / 5 + 20 +
   (x - (x / 6 + 10) - ((x - (x / 6 + 10)) / 5 + 20)) / 4 + 25) +
  (x - (x / 6 + 10) - ((x - (x / 6 + 10)) / 5 + 20) - ((x - (x / 6 + 10) - ((x - (x / 6 + 10)) / 5 + 20)) / 4 + 25)) = x ∧
  (x - (x / 6 + 10) - ((x - (x / 6 + 10)) / 5 + 20) - ((x - (x / 6 + 10) - ((x - (x / 6 + 10)) / 5 + 20)) / 4 + 25)) = 85 ∧
  x = 262 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_l1270_127067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l1270_127069

/-- The curve C in the x-y plane -/
def curve_C (x y : ℝ) : Prop := x^2 + y^2 / 3 = 1

/-- The line l in the x-y plane -/
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

/-- Distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (Real.sqrt 3 * x - y - Real.sqrt 3) / 2

/-- Maximum distance from curve C to line l -/
theorem max_distance_curve_to_line :
  ∃ (θ : ℝ), curve_C (Real.cos θ) (Real.sqrt 3 * Real.sin θ) ∧
    ∀ (φ : ℝ), curve_C (Real.cos φ) (Real.sqrt 3 * Real.sin φ) →
      distance_to_line (Real.cos φ) (Real.sqrt 3 * Real.sin φ) ≤
        distance_to_line (Real.cos θ) (Real.sqrt 3 * Real.sin θ) ∧
    distance_to_line (Real.cos θ) (Real.sqrt 3 * Real.sin θ) = (Real.sqrt 6 + Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l1270_127069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sequence_l1270_127079

noncomputable def a (n k : ℕ) : ℝ := ((n - k : ℝ) / k) ^ (k - n / 2 : ℝ)

theorem max_value_of_sequence (n : ℕ) (h : n > 1) :
  (∃ k : ℕ, 0 < k ∧ k ≤ n ∧
    ∀ j : ℕ, 0 < j → j ≤ n → a n k ≥ a n j) ∧
  (n % 2 = 1 →
    ∃ k : ℕ, 0 < k ∧ k ≤ n ∧ a n k = Real.sqrt ((n - 1 : ℝ) / (n + 1))) ∧
  (n % 2 = 0 →
    ∃ k : ℕ, 0 < k ∧ k ≤ n ∧ a n k = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sequence_l1270_127079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l1270_127037

theorem arrangement_count (n : ℕ) (h : n = 4) : 
  (Finset.univ.filter (λ s : Finset (Fin n) => s.card = 2)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l1270_127037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_a_imag_part_b_l1270_127034

-- Part a
noncomputable def complex_number_a : ℂ := (2 - Complex.I)^3 / (3 + 4*Complex.I)

theorem real_part_a : complex_number_a.re = -1.52 := by sorry

-- Part b
noncomputable def complex_number_b : ℂ := ((1 + Complex.I) / (1 - Complex.I))^11

theorem imag_part_b : complex_number_b.im = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_a_imag_part_b_l1270_127034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1270_127006

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2^((1 - x) / (1 + x))

-- State the theorem
theorem range_of_f :
  Set.range f = {y : ℝ | y ∈ Set.Ioo 0 (1/2) ∨ y ∈ Set.Ioi (1/2)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1270_127006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_ride_time_l1270_127050

/-- Represents the escalator problem with given times for walking and riding --/
structure EscalatorProblem where
  stationary_time : ℚ  -- Time to walk down stationary escalator
  moving_time : ℚ      -- Time to walk down moving escalator
  walking_speed : ℚ    -- Clea's walking speed
  escalator_length : ℚ -- Length of the escalator

/-- Calculates the time to ride the escalator without walking --/
def ride_time (problem : EscalatorProblem) : ℚ :=
  problem.stationary_time * problem.moving_time / (problem.stationary_time - problem.moving_time)

/-- Theorem stating that given the conditions, the ride time is 45 seconds --/
theorem escalator_ride_time (problem : EscalatorProblem) 
  (h1 : problem.stationary_time = 90)
  (h2 : problem.moving_time = 30) :
  ride_time problem = 45 := by
  sorry

def main : IO Unit := do
  let problem : EscalatorProblem := {
    stationary_time := 90,
    moving_time := 30,
    walking_speed := 1,
    escalator_length := 90
  }
  IO.println s!"Ride time: {ride_time problem}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_ride_time_l1270_127050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_rise_proof_l1270_127055

/-- Represents a cube-shaped container -/
structure Cube where
  edge : ℝ

/-- Calculates the volume of a cube -/
def volume (c : Cube) : ℝ := c.edge ^ 3

/-- Calculates the base area of a cube -/
def baseArea (c : Cube) : ℝ := c.edge ^ 2

/-- Represents the fishbowl and iron cube scenario -/
structure FishbowlScenario where
  fishbowl : Cube
  ironCube : Cube
  initialWaterHeight : ℝ

/-- Calculates the rise in water level when the iron cube is submerged -/
noncomputable def waterLevelRise (scenario : FishbowlScenario) : ℝ :=
  volume scenario.ironCube / baseArea scenario.fishbowl

theorem water_level_rise_proof (scenario : FishbowlScenario) 
  (h1 : scenario.fishbowl.edge = 20)
  (h2 : scenario.ironCube.edge = 10)
  (h3 : scenario.initialWaterHeight = 15) :
  waterLevelRise scenario = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_rise_proof_l1270_127055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleAllWhite4x4_impossibleAllWhite5x5_l1270_127085

/-- Represents a square on the chessboard -/
inductive Square
| White
| Black

/-- Represents a chessboard -/
def Chessboard (n : Nat) := Fin n → Fin n → Square

/-- Checks if all squares on the chessboard are white -/
def allWhite (board : Chessboard n) : Prop :=
  ∀ i j, board i j = Square.White

/-- Represents a move on the chessboard (changing a row or column) -/
inductive Move (n : Nat)
| Row (i : Fin n)
| Col (j : Fin n)

/-- Applies a move to the chessboard -/
def applyMove (board : Chessboard n) (move : Move n) : Chessboard n :=
  match move with
  | Move.Row i => λ j k => if j = i then
      match board j k with
      | Square.White => Square.Black
      | Square.Black => Square.White
    else board j k
  | Move.Col j => λ k i => if i = j then
      match board k i with
      | Square.White => Square.Black
      | Square.Black => Square.White
    else board k i

/-- Represents a sequence of moves -/
def MoveSequence (n : Nat) := List (Move n)

/-- Applies a sequence of moves to the chessboard -/
def applyMoveSequence (board : Chessboard n) : MoveSequence n → Chessboard n
| [] => board
| (move :: moves) => applyMoveSequence (applyMove board move) moves

/-- The initial board state with one black square -/
def initialBoard (n : Nat) : Chessboard n :=
  λ i j => if i.val = 0 ∧ j.val = 0 then Square.Black else Square.White

/-- Theorem: It's impossible to make all squares white on a 4x4 chessboard -/
theorem impossibleAllWhite4x4 :
  ¬∃ (moves : MoveSequence 4), allWhite (applyMoveSequence (initialBoard 4) moves) := by
  sorry

/-- Theorem: It's impossible to make all squares white on a 5x5 chessboard -/
theorem impossibleAllWhite5x5 :
  ¬∃ (moves : MoveSequence 5), allWhite (applyMoveSequence (initialBoard 5) moves) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleAllWhite4x4_impossibleAllWhite5x5_l1270_127085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_inequality_solution_l1270_127098

noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

-- Theorem for the increasing property of f
theorem f_increasing : 
  ∀ x₁ x₂ : ℝ, -1 < x₁ → x₁ < x₂ → x₂ < 1 → f x₁ < f x₂ :=
by
  sorry

-- Theorem for the solution of the inequality
theorem inequality_solution : 
  {x : ℝ | 0 < x ∧ x < 1/2} = {x : ℝ | f (x-1) + f x < 0 ∧ -1 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_inequality_solution_l1270_127098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_l1270_127010

/-- Triathlon race with swimming segment twice as long as biking and running segments -/
structure Triathlon where
  swim_speed : ℝ
  bike_speed : ℝ
  run_speed : ℝ
  segment_length : ℝ
  swim_twice_long : swim_speed > 0 ∧ bike_speed > 0 ∧ run_speed > 0 ∧ segment_length > 0

/-- Calculate the average speed of the triathlete for the entire race -/
noncomputable def average_speed (t : Triathlon) : ℝ :=
  (4 * t.segment_length) / (2 * t.segment_length / t.swim_speed + t.segment_length / t.bike_speed + t.segment_length / t.run_speed)

/-- Theorem stating that the average speed is approximately 3.56 km/h -/
theorem triathlon_average_speed :
  ∀ t : Triathlon, t.swim_speed = 2 ∧ t.bike_speed = 25 ∧ t.run_speed = 12 →
  ∃ ε > 0, |average_speed t - 3.56| < ε := by
  sorry

#eval "Triathlon average speed theorem defined"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_l1270_127010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l1270_127051

open Set Real

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, Real.log x ≥ 0 → 1/x ≤ 1) ∧ 
  (∃ x : ℝ, 1/x ≤ 1 ∧ Real.log x < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l1270_127051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1270_127049

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance between two parallel lines is 3/5 -/
theorem distance_between_given_lines :
  let l₁ : Line := ⟨4, -3, 2⟩
  let l₂ : Line := ⟨4, -3, -1⟩
  distance_between_parallel_lines l₁.a l₁.b l₁.c l₂.c = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1270_127049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_students_count_l1270_127057

theorem school_students_count : ∃ S : ℕ,
  -- Define the number of students in each class
  ∃ class_A class_B class_C : ℕ,
  -- Condition 1: 40% of students are in class A
  class_A = (S * 2) / 5 ∧
  -- Condition 2: Class B has 21 students fewer than class A
  class_B = class_A - 21 ∧
  -- Condition 3: There are 37 students in class C
  class_C = 37 ∧
  -- The total number of students is the sum of all classes
  S = class_A + class_B + class_C ∧
  -- Theorem: The total number of students is 80
  S = 80 := by
  -- Proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_students_count_l1270_127057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nanometer_scientific_notation_l1270_127009

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  norm : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Theorem stating that 0.000000001 in scientific notation is 1.0 × 10^(-9) -/
theorem nanometer_scientific_notation :
  toScientificNotation 0.000000001 = ScientificNotation.mk 1.0 (-9) (by sorry) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nanometer_scientific_notation_l1270_127009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_bisect_implies_parallelogram_l1270_127042

/-- A quadrilateral is represented by its four vertices -/
structure Quadrilateral :=
  (A B C D : EuclideanSpace ℝ (Fin 2))

/-- Definition of a parallelogram -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  (dist q.A q.B = dist q.C q.D) ∧ (dist q.A q.D = dist q.B q.C)

/-- Definition of diagonals bisecting each other -/
def diagonals_bisect (q : Quadrilateral) : Prop :=
  let M := midpoint ℝ q.A q.C
  let N := midpoint ℝ q.B q.D
  M = N

/-- Theorem: If the diagonals of a quadrilateral bisect each other, then it is a parallelogram -/
theorem diagonals_bisect_implies_parallelogram (q : Quadrilateral) :
  diagonals_bisect q → is_parallelogram q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_bisect_implies_parallelogram_l1270_127042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_line_covering_sum_at_least_two_l1270_127014

/-- A "2-line" is represented by its width (distance between parallel lines) -/
def TwoLine := ℝ

/-- A covering of a unit circle by "2-lines" -/
def UnitCircleCovering := List TwoLine

/-- The sum of lengths (widths) of "2-lines" in a covering -/
def sumLengths (covering : UnitCircleCovering) : ℝ :=
  covering.foldl (·+·) 0

/-- Predicate to check if a set of "2-lines" covers the unit circle -/
def coversUnitCircle (covering : UnitCircleCovering) : Prop :=
  sorry  -- Definition of coverage would go here

theorem two_line_covering_sum_at_least_two (covering : UnitCircleCovering) :
  coversUnitCircle covering → sumLengths covering ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_line_covering_sum_at_least_two_l1270_127014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1270_127061

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 
  (1 + Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x)) / 
  (1 + Real.sin x + Real.sqrt 3 * Real.cos x)

/-- The range of f(x) is [-3, -2) ∪ (-2, 1] -/
theorem range_of_f : Set.range f = Set.Icc (-3) (-2) ∪ Set.Ioc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1270_127061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coord_product_theorem_l1270_127068

/-- The product of the coordinates of a point --/
def coordProduct (p : ℝ × ℝ) : ℝ :=
  p.1 * p.2

theorem midpoint_coord_product_theorem :
  let p1 : ℝ × ℝ := (10, 7)
  let p2 : ℝ × ℝ := (-6, 3)
  coordProduct ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coord_product_theorem_l1270_127068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sets_count_l1270_127076

/-- A correct set of weights is a multiset of positive integers that sum to 500 and can uniquely represent any integer from 1 to 500. -/
def CorrectSet : Type := Multiset ℕ

/-- A function that checks if a given set of weights is correct -/
def isCorrect (s : CorrectSet) : Prop :=
  (s.sum = 500) ∧
  (∀ w : ℕ, w > 0 → w ≤ 500 → ∃! subset : Multiset ℕ, subset ⊆ s ∧ subset.sum = w)

/-- The number of different correct sets of weights -/
def numCorrectSets : ℕ := 3

theorem correct_sets_count :
  numCorrectSets = 3 :=
by
  -- The proof would go here, but we'll use sorry for now
  sorry

#check correct_sets_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sets_count_l1270_127076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_section_volume_is_seven_twentysevenths_main_theorem_l1270_127025

/-- A pyramid with a lateral edge divided into three equal parts by two planes parallel to the base. -/
structure DividedPyramid where
  /-- The volume of the entire pyramid -/
  total_volume : ℝ
  /-- Assumption that the total volume is 1 -/
  total_volume_is_one : total_volume = 1

/-- The volume of the section between two planes that divide the lateral edge of a pyramid into three equal parts -/
noncomputable def section_volume (p : DividedPyramid) : ℝ := 7 / 27 * p.total_volume

/-- Theorem stating that the volume of the section between the planes is 7/27 of the total volume -/
theorem section_volume_is_seven_twentysevenths (p : DividedPyramid) :
  section_volume p = 7 / 27 := by
  -- Unfold the definition of section_volume
  unfold section_volume
  -- Use the fact that total_volume is 1
  rw [p.total_volume_is_one]
  -- Simplify the arithmetic
  norm_num

/-- The main theorem: The volume of the section is 7/27 -/
theorem main_theorem (p : DividedPyramid) :
  section_volume p = 7 / 27 := by
  exact section_volume_is_seven_twentysevenths p

end NUMINAMATH_CALUDE_ERRORFEEDBACK_section_volume_is_seven_twentysevenths_main_theorem_l1270_127025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1270_127092

theorem simplify_expression : (-2) - (-10) + (-6) - (5) = -2 + 10 - 6 - 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1270_127092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_third_region_l1270_127019

/-- A regular dodecagon inscribed in a circle -/
structure RegularDodecagon where
  /-- The circle in which the dodecagon is inscribed -/
  circle : Real
  /-- The area of the circle is 4π -/
  circle_area : circle = 4 * Real.pi
  /-- A point Q inside the circle -/
  Q : Real × Real
  /-- The area of the region bounded by QB₁, QB₂, and the minor arc B₁B₂ -/
  area_region_1 : Real
  /-- The area of the region bounded by QB₁, QB₂, and the minor arc B₁B₂ is 1/12 -/
  area_region_1_value : area_region_1 = 1 / 12
  /-- The area of the region bounded by QB₄, QB₅, and the minor arc B₄B₅ -/
  area_region_2 : Real
  /-- The area of the region bounded by QB₄, QB₅, and the minor arc B₄B₅ is 1/10 -/
  area_region_2_value : area_region_2 = 1 / 10

/-- The theorem to be proved -/
theorem area_of_third_region (d : RegularDodecagon) :
  ∃ (area_region_3 : Real),
    area_region_3 = 1 / 10 ∧
    area_region_3 = (d.circle / 12 - (d.circle / 12 - d.area_region_2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_third_region_l1270_127019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_triangle_ratio_l1270_127023

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x ^ 2 + cos (π / 4 - x) ^ 2 - (1 + sqrt 3) / 2

-- Part 1: Maximum value of f in [0, π/2]
theorem f_max_value : 
  ∃ (x : ℝ), x ∈ Set.Icc 0 (π / 2) ∧ f x = 1 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 (π / 2) → f y ≤ f x :=
sorry

-- Part 2: BC/AB ratio in triangle ABC
theorem triangle_ratio (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hSum : A + B + C = π) (hLess : A < B) (hfA : f A = 1/2) (hfB : f B = 1/2) :
  sin A / sin C = sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_triangle_ratio_l1270_127023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_a_range_l1270_127012

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x > a}

theorem range_of_a (a : ℝ) : (A ∩ B a).Nonempty → a < 2 :=
by sorry

theorem a_range : {a : ℝ | (A ∩ B a).Nonempty} = Set.Iio 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_a_range_l1270_127012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_coloring_l1270_127048

/-- A type representing the three colors --/
inductive Color where
  | Red
  | Blue
  | Green

/-- A type representing the vertices of the pentagon and pentagram --/
inductive Vertex where
  | A
  | B
  | C
  | D
  | E
  | APrime
  | BPrime
  | CPrime
  | DPrime
  | EPrime

/-- A type representing a segment between two vertices --/
structure Segment where
  start : Vertex
  finish : Vertex

/-- A function type representing a coloring of segments --/
def Coloring := Segment → Color

/-- The list of all 15 segments in the figure --/
def allSegments : List Segment := sorry

/-- Predicate to check if a coloring is valid --/
def isValidColoring (c : Coloring) : Prop :=
  ∀ s1 s2 : Segment, s1 ∈ allSegments → s2 ∈ allSegments →
    s1 ≠ s2 → (s1.start = s2.start ∨ s1.start = s2.finish ∨ s1.finish = s2.start ∨ s1.finish = s2.finish) →
    c s1 ≠ c s2

/-- The main theorem stating that no valid coloring exists --/
theorem no_valid_coloring : ¬∃ c : Coloring, isValidColoring c := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_coloring_l1270_127048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fran_required_speed_l1270_127020

/-- Calculates the required average speed for Fran to cover the same distance as Joann -/
noncomputable def required_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) : ℝ :=
  (joann_speed * joann_time) / fran_time

/-- Proves that Fran needs to ride at 24 mph to cover the same distance as Joann -/
theorem fran_required_speed :
  required_speed 15 4 2.5 = 24 := by
  -- Unfold the definition of required_speed
  unfold required_speed
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- Check that the result is equal to 24
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fran_required_speed_l1270_127020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounded_by_power_function_l1270_127075

noncomputable def a_sequence : ℕ → ℝ
| 0 => 1
| n + 1 => Real.sqrt ((a_sequence n) ^ 2 + 1 / (a_sequence n))

theorem sequence_bounded_by_power_function :
  ∃ (α : ℝ), α > 0 ∧ ∀ (n : ℕ), n > 0 → 1/2 ≤ a_sequence n / (n : ℝ) ^ α ∧ a_sequence n / (n : ℝ) ^ α ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounded_by_power_function_l1270_127075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1270_127036

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (m : ℝ) (h_m : m > 0) : 
  let ellipse := fun (x y : ℝ) => x^2 / 16 + y^2 / m^2 = 1
  let line := fun (x : ℝ) => (Real.sqrt 2 / 2) * x
  ∃ (x y : ℝ), 
    ellipse x y ∧ 
    y = line x ∧
    x = Real.sqrt (16 - m^2) →
  Real.sqrt (16 - m^2) / 4 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1270_127036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_generatrix_length_l1270_127013

/-- Represents a cone with specific properties -/
structure Cone where
  r : ℝ  -- radius of the base
  l : ℝ  -- length of the generatrix
  h : ℝ  -- height of the cone

/-- The area of the sector formed by the unfolding of the cone's lateral surface -/
noncomputable def sector_area (c : Cone) : ℝ := Real.pi * c.r * c.l

/-- The area of the base circle of the cone -/
noncomputable def base_area (c : Cone) : ℝ := Real.pi * c.r^2

/-- The volume of the cone -/
noncomputable def volume (c : Cone) : ℝ := (1/3) * Real.pi * c.r^2 * c.h

/-- The Pythagorean theorem relating the cone's dimensions -/
def pythagorean (c : Cone) : Prop := c.l^2 = c.r^2 + c.h^2

/-- Theorem stating the properties of the specific cone and its generatrix length -/
theorem cone_generatrix_length :
  ∀ c : Cone,
  sector_area c = 2 * base_area c →
  volume c = 9 * Real.sqrt 3 * Real.pi →
  pythagorean c →
  c.l = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_generatrix_length_l1270_127013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_BE_ED_is_sqrt_3_l1270_127099

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let dist := λ p₁ p₂ : ℝ × ℝ => Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)
  dist q.A q.B = 5 ∧
  dist q.B q.C = 6 ∧
  dist q.C q.D = 5 ∧
  dist q.D q.A = 4 ∧
  (q.B.1 - q.A.1) * (q.C.1 - q.B.1) + (q.B.2 - q.A.2) * (q.C.2 - q.B.2) = 0

-- Define the intersection point E of diagonals AC and BD
noncomputable def intersection_point (q : Quadrilateral) : ℝ × ℝ :=
  sorry

-- Define the ratio BE/ED
noncomputable def ratio_BE_ED (q : Quadrilateral) : ℝ :=
  let E := intersection_point q
  let dist := λ p₁ p₂ : ℝ × ℝ => Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)
  dist q.B E / dist E q.D

-- Theorem statement
theorem ratio_BE_ED_is_sqrt_3 (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  ratio_BE_ED q = Real.sqrt 3 := by
  sorry

#check ratio_BE_ED_is_sqrt_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_BE_ED_is_sqrt_3_l1270_127099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1270_127040

-- Define a function to represent a quadratic radical
noncomputable def quadratic_radical (x : ℝ) : ℝ := Real.sqrt x

-- Define the simplicity of a quadratic radical
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, quadratic_radical y = quadratic_radical x → x ≤ y

-- Theorem statement
theorem simplest_quadratic_radical (a : ℝ) :
  let options : List ℝ := [a^2 + 4, 1/2, 3*(a^2), 0.3]
  is_simplest_quadratic_radical (a^2 + 4) ∧ 
  (a^2 + 4) ∈ options ∧
  ∀ x ∈ options, x ≠ (a^2 + 4) → ¬(is_simplest_quadratic_radical x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1270_127040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_yellow_marbles_count_l1270_127088

theorem smallest_yellow_marbles_count : ∃ (n : ℕ),
  (n % 12 = 0) ∧ 
  (n / 4 + n / 6 + 7 ≤ n) ∧
  (n / 4 : ℚ).den = 1 ∧ 
  (n / 6 : ℚ).den = 1 ∧ 
  ((n : ℤ) - (n / 4 + n / 6 + 7 : ℤ)).natAbs = 0 ∧
  (∀ m : ℕ, m < n → ¬(
    (m % 12 = 0) ∧ 
    (m / 4 + m / 6 + 7 ≤ m) ∧
    (m / 4 : ℚ).den = 1 ∧ 
    (m / 6 : ℚ).den = 1 ∧ 
    ((m : ℤ) - (m / 4 + m / 6 + 7 : ℤ)).natAbs = 0
  )) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_yellow_marbles_count_l1270_127088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_proof_l1270_127064

/-- The distance between two adjacent intersection points of y = 2016 and y = tan(3x) -/
noncomputable def intersection_distance : ℝ := Real.pi / 3

/-- The line y = 2016 -/
def line : ℝ → ℝ := λ _ ↦ 2016

/-- The curve y = tan(3x) -/
noncomputable def curve : ℝ → ℝ := λ x ↦ Real.tan (3 * x)

theorem intersection_distance_proof :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
  line x₁ = curve x₁ ∧ 
  line x₂ = curve x₂ ∧ 
  (∀ x ∈ Set.Ioo x₁ x₂, line x ≠ curve x) →
  x₂ - x₁ = intersection_distance :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_proof_l1270_127064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_over_plane_l1270_127004

/-- A convex polygon in the plane. -/
structure ConvexPolygon where
  perimeter : ℝ
  area : ℝ

/-- The distance function from a point to the nearest point of the polygon or its interior. -/
noncomputable def distance (P : ConvexPolygon) (x y : ℝ) : ℝ := sorry

/-- The main theorem stating the integral result. -/
theorem integral_over_plane (P : ConvexPolygon) :
  (∫ (x : ℝ), ∫ (y : ℝ), Real.exp (-(distance P x y))) = 2 * Real.pi + P.perimeter + P.area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_over_plane_l1270_127004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_equilateral_heights_different_l1270_127056

/-- Represents a triangle with three sides -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Represents a vertex of a triangle -/
inductive Vertex
  | A
  | B
  | C

/-- A triangle is equilateral if all its sides are equal -/
def IsEquilateral (t : Triangle) : Prop := 
  t.side1 = t.side2 ∧ t.side2 = t.side3

/-- The height of a triangle from a vertex to the opposite side -/
noncomputable def Height (t : Triangle) (v : Vertex) : ℝ := 
  sorry

/-- For any non-equilateral triangle, all three heights are different -/
theorem non_equilateral_heights_different (t : Triangle) : 
  ¬IsEquilateral t → 
  Height t Vertex.A ≠ Height t Vertex.B ∧ 
  Height t Vertex.B ≠ Height t Vertex.C ∧ 
  Height t Vertex.C ≠ Height t Vertex.A :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_equilateral_heights_different_l1270_127056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unspent_portion_is_two_thirds_l1270_127005

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℚ
  balance : ℚ

/-- Represents a person's credit cards -/
structure PersonCards where
  gold : CreditCard
  platinum : CreditCard

/-- Calculates the unspent portion of a credit card's limit after a balance transfer -/
def unspentPortionAfterTransfer (cards : PersonCards) : ℚ :=
  let newPlatinumBalance := cards.platinum.balance + cards.gold.balance
  let unspentAmount := cards.platinum.limit - newPlatinumBalance
  unspentAmount / cards.platinum.limit

/-- Theorem: The unspent portion of the platinum card's limit after transfer is 2/3 -/
theorem unspent_portion_is_two_thirds (cards : PersonCards)
    (h1 : cards.platinum.limit = 2 * cards.gold.limit)
    (h2 : cards.gold.balance = cards.gold.limit / 3)
    (h3 : cards.platinum.balance = cards.platinum.limit / 6) :
    unspentPortionAfterTransfer cards = 2 / 3 := by
  sorry

#check unspent_portion_is_two_thirds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unspent_portion_is_two_thirds_l1270_127005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_walk_l1270_127060

/-- Calculates the distance from the starting point after a series of movements -/
noncomputable def distance_from_start (north south east : ℝ) : ℝ :=
  Real.sqrt ((south - north)^2 + east^2)

/-- Proves that the distance from the starting point is 30√2 meters -/
theorem distance_after_walk : distance_from_start 10 40 30 = 30 * Real.sqrt 2 := by
  unfold distance_from_start
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_walk_l1270_127060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l1270_127066

noncomputable def f (x : ℝ) := Real.sin (2 * x) + Real.cos (2 * x)

noncomputable def g (x : ℝ) := f (x - Real.pi / 4)

theorem symmetry_center_of_g :
  ∃ (k : ℤ), g (5 * Real.pi / 8 + k * Real.pi) = 0 ∧ 
    g (5 * Real.pi / 8 - k * Real.pi) = g (5 * Real.pi / 8 + k * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l1270_127066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_differential_equation_l1270_127026

-- Define the function y as noncomputable
noncomputable def y (x : ℝ) : ℝ := x * Real.exp (-x^2 / 2)

-- State the theorem
theorem y_satisfies_differential_equation :
  ∀ x : ℝ, x * (deriv y x) = (1 - x^2) * y x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_differential_equation_l1270_127026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1270_127033

open Real

theorem trigonometric_identities :
  (∀ α : ℝ, (cos (180 * π / 180 + α) * sin (α + 360 * π / 180)) / 
    (sin (-α - 180 * π / 180) * cos (-180 * π / 180 - α)) = 1) ∧
  (∀ α : ℝ, tan α = -3/4 → 
    (cos (π/2 + α) * sin (-π - α)) / 
    (cos (11*π/2 - α) * sin (11*π/2 + α)) = 3/4) :=
by
  constructor
  · intro α
    sorry
  · intro α h
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1270_127033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_theorem_l1270_127044

theorem tan_plus_cot_theorem (x : ℝ) (a b c : ℕ+) 
  (h1 : 0 < x ∧ x < Real.pi/2)
  (h2 : Real.sin x - Real.cos x = Real.pi/4)
  (h3 : Real.tan x + 1 / Real.tan x = (a : ℝ) / ((b : ℝ) - Real.pi^(c : ℝ))) :
  a + b + c = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_theorem_l1270_127044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_solution_correct_l1270_127091

def is_valid_number (n : ℕ) : Prop :=
  n > 0 ∧
  n % 15 = 0 ∧
  n ≥ 10000 ∧ n < 100000 ∧
  ∀ d, d ∈ n.digits 10 → d = 8 ∨ d = 0

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 88080 :=
by sorry

theorem solution_correct :
  (88080 : ℕ) / 15 = 5872 :=
by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_solution_correct_l1270_127091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_germination_probability_estimate_l1270_127039

/-- Represents a single germination experiment result -/
structure GerminationResult where
  totalSeeds : ℕ
  germinatedSeeds : ℕ

/-- Calculates the germination ratio for a single experiment -/
def germinationRatio (result : GerminationResult) : ℚ :=
  result.germinatedSeeds / result.totalSeeds

/-- The experimental data for rapeseed germination -/
def experimentalData : List GerminationResult := [
  ⟨50, 45⟩,
  ⟨100, 96⟩,
  ⟨300, 283⟩,
  ⟨400, 380⟩,
  ⟨600, 571⟩,
  ⟨1000, 948⟩
]

/-- Calculates the average germination ratio from the experimental data -/
def averageGerminationRatio (data : List GerminationResult) : ℚ :=
  (data.map germinationRatio).sum / data.length

/-- The estimated germination probability based on the experimental data -/
def estimatedGerminationProbability : ℚ := averageGerminationRatio experimentalData

theorem germination_probability_estimate :
  |estimatedGerminationProbability - 95/100| ≤ 1/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_germination_probability_estimate_l1270_127039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1270_127024

/-- The eccentricity of a hyperbola C₂ given specific conditions -/
theorem hyperbola_eccentricity : 
  ∀ (F₁ F₂ A B : ℝ × ℝ) (C₁ C₂ : Set (ℝ × ℝ)),
  -- C₁ is an ellipse with equation x²/4 + y² = 1
  (∀ (x y : ℝ), (x, y) ∈ C₁ ↔ x^2/4 + y^2 = 1) →
  -- F₁ and F₂ are foci of both C₁ and C₂
  (F₁ ∈ C₁ ∧ F₁ ∈ C₂ ∧ F₂ ∈ C₁ ∧ F₂ ∈ C₂) →
  -- A and B are common points of C₁ and C₂
  (A ∈ C₁ ∧ A ∈ C₂ ∧ B ∈ C₁ ∧ B ∈ C₂) →
  -- A is in the second quadrant and B is in the fourth quadrant
  (A.1 < 0 ∧ A.2 > 0 ∧ B.1 > 0 ∧ B.2 < 0) →
  -- AF₁BF₂ is a rectangle
  (dist A F₁ = dist B F₂ ∧ dist A F₂ = dist B F₁ ∧ 
   ((A.1 - F₁.1) * (A.1 - F₂.1) + (A.2 - F₁.2) * (A.2 - F₂.2) = 0)) →
  -- The eccentricity of C₂ is √6/2
  ∃ (e : ℝ), e = Real.sqrt 6 / 2 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ C₂ → 
      e = (dist (x, y) F₁ - dist (x, y) F₂) / (2 * dist F₁ F₂)
:= by
  sorry

where
  dist (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1270_127024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parabolas_l1270_127038

/-- The area of the region bounded by the graphs of y = x^2 - 4x + 3 and y = -x^2 + 2x + 3 -/
theorem area_between_parabolas : ∃ area : ℝ, area = 9 := by
  let f : ℝ → ℝ := λ x => x^2 - 4*x + 3
  let g : ℝ → ℝ := λ x => -x^2 + 2*x + 3
  let intersection_points : Set ℝ := {x : ℝ | f x = g x}
  let lower_bound : ℝ := 0
  let upper_bound : ℝ := 3
  let area : ℝ := ∫ x in lower_bound..upper_bound, (g x - f x)

  have h : intersection_points = {0, 3} := by sorry
  have i : ∀ x ∈ (Set.Ioo lower_bound upper_bound), f x ≤ g x := by sorry
  have j : area = 9 := by sorry

  exact ⟨area, j⟩

#check area_between_parabolas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parabolas_l1270_127038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_icosahedron_ratio_l1270_127086

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- An octahedron with edge length m -/
structure Octahedron where
  m : ℝ
  m_pos : m > 0

/-- An icosahedron inscribed in an octahedron -/
structure InscribedIcosahedron (oct : Octahedron) where
  ratio : ℝ
  ratio_pos : ratio > 0
  ratio_lt_one : ratio < 1

/-- The theorem stating the ratio for inscribing an icosahedron in an octahedron -/
theorem inscribed_icosahedron_ratio (oct : Octahedron) :
  ∃ (ico : InscribedIcosahedron oct), ico.ratio = φ - 1 ∨ ico.ratio = 1 / φ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_icosahedron_ratio_l1270_127086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_sum_of_f_values_l1270_127046

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 3) + 1 / (x - 2)

-- Theorem for the domain of f
theorem domain_of_f :
  {x : ℝ | x ≥ -3 ∧ x ≠ 2} = {x : ℝ | f x ≠ 0} := by
  sorry

-- Theorem for the value of f(1) + f(-3)
theorem sum_of_f_values :
  f 1 + f (-3) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_sum_of_f_values_l1270_127046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_shaded_area_l1270_127018

/-- The side length of the carpet in feet -/
noncomputable def carpet_side : ℝ := 12

/-- The side length of the large shaded square in feet -/
noncomputable def large_square_side : ℝ := carpet_side / 4

/-- The side length of each small shaded square in feet -/
noncomputable def small_square_side : ℝ := large_square_side / 4

/-- The number of small shaded squares -/
def num_small_squares : ℕ := 12

/-- Calculates the total shaded area of the carpet -/
noncomputable def total_shaded_area : ℝ :=
  large_square_side ^ 2 + num_small_squares * small_square_side ^ 2

theorem carpet_shaded_area :
  total_shaded_area = 15.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_shaded_area_l1270_127018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_lower_bound_l1270_127065

/-- Represents a convex polygon --/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  convex : ∀ (a b : ℝ × ℝ), a ∈ vertices → b ∈ vertices → 
           ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 → 
           (t • a.1 + (1 - t) • b.1, t • a.2 + (1 - t) • b.2) ∈ vertices

/-- Represents the clipping operation on a convex polygon --/
noncomputable def clip (P : ConvexPolygon) : ConvexPolygon :=
  sorry

/-- The area of a convex polygon --/
noncomputable def area (P : ConvexPolygon) : ℝ :=
  sorry

/-- The sequence of polygons obtained by repeated clipping --/
noncomputable def polygon_sequence (n : ℕ) : ConvexPolygon :=
  sorry

/-- Theorem stating that the area of the polygon remains greater than 1/3 --/
theorem area_lower_bound (n : ℕ) (h : n ≥ 6) :
  area (polygon_sequence n) > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_lower_bound_l1270_127065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_types_count_l1270_127054

/-- The number of types of lettuce available -/
def lettuce_types : ℕ := 2

/-- The number of types of tomatoes available -/
def tomato_types : ℕ := 3

/-- The number of types of olives available -/
def olive_types : ℕ := 4

/-- The total number of lunch combo options -/
def total_combos : ℕ := 48

/-- The number of salad choices -/
def salad_choices : ℕ := lettuce_types * tomato_types * olive_types

/-- Theorem: The number of soup types is 2 -/
theorem soup_types_count : total_combos / salad_choices = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_types_count_l1270_127054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_monomial_degree_of_monomial_l1270_127089

-- Define the monomial structure
structure Monomial (α : Type*) [CommRing α] where
  coefficient : α
  x_exponent : ℕ
  y_exponent : ℕ

-- Define our specific monomial
noncomputable def our_monomial : Monomial ℝ :=
  { coefficient := -5 / Real.pi
  , x_exponent := 3
  , y_exponent := 2 }

-- Theorem for the coefficient
theorem coefficient_of_monomial :
  our_monomial.coefficient = -5 / Real.pi := by
  rfl

-- Theorem for the degree
theorem degree_of_monomial :
  our_monomial.x_exponent + our_monomial.y_exponent = 5 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_monomial_degree_of_monomial_l1270_127089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1270_127000

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (∀ x, f (-Real.pi / 6 + x) = f (-Real.pi / 6 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1270_127000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_application_l1270_127077

theorem remainder_theorem_application : ∃ q : Polynomial ℤ, 
  (X : Polynomial ℤ)^15 - X + 3 = (X - 2) * q + 32769 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_application_l1270_127077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_range_l1270_127028

theorem parabola_intersection_range (m : ℝ) : 
  let f (x : ℝ) := x^2 - (4*m + 1)*x + 2*m - 1
  (∃ x₁ x₂, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ < 2 ∧ x₂ > 2) →
  (f 0 < -1/2) →
  1/6 < m ∧ m < 1/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_range_l1270_127028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_circle_sector_l1270_127022

/-- Given a circle with radius 15 and a sector with central angle π/2,
    if a triangle is formed with vertices at the center and two points on the arc
    that divide the arc into three equal segments,
    then the area of this triangle is (1125 * √3) / 4. -/
theorem triangle_area_in_circle_sector (r : ℝ) (θ : ℝ) :
  r = 15 →
  θ = π / 2 →
  (1 / 2) * r^2 * Real.sin (θ / 3) = (1125 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_circle_sector_l1270_127022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_planes_formula_l1270_127062

/-- Represents a rectangular parallelepiped with edge lengths a, b, and c -/
structure RectangularParallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The cosine of the angle between planes BB₁D and ABC₁ in a rectangular parallelepiped -/
noncomputable def angle_between_planes (p : RectangularParallelepiped) : ℝ :=
  (p.a * p.c) / (Real.sqrt (p.a^2 + p.b^2) * Real.sqrt (p.a^2 + p.c^2))

/-- Theorem stating that the cosine of the angle between planes BB₁D and ABC₁
    in a rectangular parallelepiped is equal to (a · c) / (√(a² + b²) · √(a² + c²)) -/
theorem angle_between_planes_formula (p : RectangularParallelepiped) :
  angle_between_planes p = (p.a * p.c) / (Real.sqrt (p.a^2 + p.b^2) * Real.sqrt (p.a^2 + p.c^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_planes_formula_l1270_127062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_intersecting_diagonals_l1270_127052

/-- Represents a polygon with n sides -/
structure Polygon (n : ℕ) where
  sides : n > 2

/-- Represents a diagonal of a polygon -/
structure Diagonal (n : ℕ) where
  polygon : Polygon n
  start : Fin n
  finish : Fin n
  ne : start ≠ finish

/-- The number of intersecting diagonals in a 20-gon divided into a 14-gon and 8-gon -/
def intersecting_diagonals (p : Polygon 20) (d : Diagonal 20) 
  (h1 : d.start = 0)
  (h2 : d.finish = 13) : ℕ := 72

theorem count_intersecting_diagonals 
  (p : Polygon 20) 
  (d : Diagonal 20) 
  (h1 : d.start = 0) 
  (h2 : d.finish = 13) :
  intersecting_diagonals p d h1 h2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_intersecting_diagonals_l1270_127052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_segments_l1270_127059

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a segment between two points -/
structure Segment where
  p1 : Point
  p2 : Point

/-- A predicate to check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- A predicate to check if a segment is colored red -/
def is_red (s : Segment) : Prop := sorry

/-- A function to check if three points form a red triangle -/
def red_triangle (p1 p2 p3 : Point) : Prop :=
  is_red ⟨p1, p2⟩ ∧ is_red ⟨p2, p3⟩ ∧ is_red ⟨p3, p1⟩

/-- The main theorem -/
theorem min_red_segments (n : ℕ) (points : Fin n → Point) 
  (h_not_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k))
  (h_red_triangle : ∀ i j k l, i ≠ j → j ≠ k → k ≠ l → i ≠ k → i ≠ l → j ≠ l → 
    red_triangle (points i) (points j) (points k) ∨ 
    red_triangle (points i) (points j) (points l) ∨
    red_triangle (points i) (points k) (points l) ∨
    red_triangle (points j) (points k) (points l)) :
  ∃ (red_segments : Finset Segment), 
    (∀ s ∈ red_segments, is_red s) ∧
    (∀ s, is_red s → s ∈ red_segments) ∧
    (red_segments.card = (n - 1) * (n - 2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_segments_l1270_127059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_improvement_percentage_l1270_127083

/-- Bob's speed for the first half mile in mph -/
noncomputable def bob_speed1 : ℝ := 5.5

/-- Bob's speed for the second half mile in mph -/
noncomputable def bob_speed2 : ℝ := 6

/-- Bob's total time for a mile in minutes -/
noncomputable def bob_total_time : ℝ := 10 + 40 / 60

/-- Bob's sister's speed for the first three-quarters of a mile in mph -/
noncomputable def sister_speed1 : ℝ := 11

/-- Bob's sister's speed for the last quarter mile in mph -/
noncomputable def sister_speed2 : ℝ := 10

/-- Bob's sister's total time for a mile in minutes -/
noncomputable def sister_total_time : ℝ := 5 + 20 / 60

/-- The percentage improvement Bob needs to match his sister's time -/
noncomputable def improvement_percentage : ℝ := 
  (bob_total_time - sister_total_time) / bob_total_time * 100

theorem bob_improvement_percentage :
  abs (improvement_percentage - 48.98) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_improvement_percentage_l1270_127083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_45_l1270_127096

/-- 
Given a line ax + (2a-3)y = 0 with a slope angle of 45°, prove that a = 1.
-/
theorem line_slope_angle_45 (a : ℝ) : 
  (∃ x y : ℝ, a * x + (2 * a - 3) * y = 0) → -- Line equation
  (Real.tan (π / 4 : ℝ) = -a / (2 * a - 3)) →          -- Slope angle is 45°
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_45_l1270_127096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_is_pi_minus_arctan_two_l1270_127082

/-- The slope of a line given by parametric equations -/
noncomputable def slope_of_parametric_line (x y : ℝ → ℝ) : ℝ :=
  -((y 1 - y 0) / (x 1 - x 0))

/-- Parametric equations of the line -/
def x (t : ℝ) : ℝ := 1 + t
def y (t : ℝ) : ℝ := 1 - 2*t

theorem slope_of_line_is_pi_minus_arctan_two :
  slope_of_parametric_line x y = Real.pi - Real.arctan 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_is_pi_minus_arctan_two_l1270_127082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matching_socks_probability_l1270_127072

/-- The probability of selecting a matching pair of socks from a drawer -/
theorem matching_socks_probability
  (gray : ℕ) (white : ℕ) (black : ℕ)
  (h_gray : gray = 12)
  (h_white : white = 10)
  (h_black : black = 6) :
  (Nat.choose gray 2 + Nat.choose white 2 + Nat.choose black 2) / Nat.choose (gray + white + black) 2 = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matching_socks_probability_l1270_127072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1270_127011

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line
def line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the midpoint condition
def midpoint_condition (xA yA xB yB : ℝ) : Prop := (xA + xB) / 2 = 3 ∧ (yA + yB) / 2 = 2

-- Define the theorem
theorem parabola_line_intersection :
  ∃ (xA yA xB yB : ℝ),
    parabola xA yA ∧ parabola xB yB ∧  -- A and B are on the parabola
    line xA yA ∧ line xB yB ∧          -- A and B are on the line
    midpoint_condition xA yA xB yB →   -- Midpoint of AB is (3,2)
    (∀ (x y : ℝ), line x y ↔ x - y - 1 = 0) ∧  -- Equation of line l
    ∃ (len : ℝ), len = 8 ∧             -- Length of AB
      len = Real.sqrt ((xA - xB)^2 + (yA - yB)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1270_127011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_approx_six_percent_l1270_127084

/-- Calculates the simple interest rate given the principal, time, and total amount --/
noncomputable def simple_interest_rate (principal : ℝ) (time : ℝ) (total_amount : ℝ) : ℝ :=
  ((total_amount - principal) * 100) / (principal * time)

/-- Theorem: The simple interest rate for the given loan is approximately 6% --/
theorem loan_interest_rate_approx_six_percent :
  let principal := 5266.23
  let time := 9
  let total_amount := 8110
  let rate := simple_interest_rate principal time total_amount
  ∃ ε > 0, |rate - 6| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_approx_six_percent_l1270_127084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_implies_a_range_l1270_127078

theorem monotonic_increasing_f_implies_a_range (a : ℝ) :
  (∀ x : ℝ, Monotone (λ x => x - (1/3) * Real.sin (2 * x) + a * Real.sin x)) →
  a ∈ Set.Icc (-1/3) (1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_implies_a_range_l1270_127078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_example_l1270_127058

/-- Calculates the number of revolutions a wheel makes when rolled on a circular track -/
noncomputable def wheel_revolutions (wheel_radius : ℝ) (track_radius : ℝ) (laps : ℕ) : ℝ :=
  (2 * Real.pi * track_radius * (laps : ℝ)) / (2 * Real.pi * wheel_radius)

/-- Theorem: A wheel with radius 5 completes 21 revolutions when rolled for 15 laps on a track with radius 7 -/
theorem wheel_revolutions_example : 
  wheel_revolutions 5 7 15 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_example_l1270_127058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scramblian_word_count_l1270_127029

/-- The number of letters in the Scramblian alphabet -/
def alphabet_size : ℕ := 6

/-- The maximum word length in the Scramblian language -/
def max_word_length : ℕ := 4

/-- The number of possible words of length n in the Scramblian language -/
def words_of_length (n : ℕ) : ℕ := alphabet_size ^ n

/-- The total number of possible words in the Scramblian language -/
def total_words : ℕ := Finset.sum (Finset.range max_word_length) (fun n => words_of_length (n + 1))

theorem scramblian_word_count :
  total_words = 1554 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scramblian_word_count_l1270_127029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l1270_127093

theorem cos_theta_value (θ : ℝ) (h1 : θ ∈ Set.Ioo 0 π) (h2 : Real.tan θ = -3/2) :
  Real.cos θ = -2 / Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l1270_127093
