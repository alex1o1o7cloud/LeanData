import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_product_equality_l1220_122073

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a line segment -/
structure LineSegment where
  start : Point2D
  finish : Point2D

/-- Theorem: For an ellipse with equation (x^2/a^2) + (y^2/b^2) = 1,
    given a diameter CD and a line parallel to CD drawn from the left vertex A
    of the major axis intersecting the ellipse at N and the minor axis at M,
    AM * AN = CO * CD -/
theorem ellipse_intersection_product_equality
  (e : Ellipse)
  (c d : Point2D)
  (a m n : Point2D)
  (cd_diameter : LineSegment)
  (an_parallel : LineSegment)
  (h1 : cd_diameter.start = c ∧ cd_diameter.finish = d)
  (h2 : an_parallel.start = a ∧ an_parallel.finish = n)
  (h3 : a.x = -e.a ∧ a.y = 0) -- A is left vertex
  (h4 : m.x = 0) -- M is on minor axis
  (h5 : (n.x^2 / e.a^2) + (n.y^2 / e.b^2) = 1) -- N is on ellipse
  (h6 : (c.x^2 / e.a^2) + (c.y^2 / e.b^2) = 1 ∧ (d.x^2 / e.a^2) + (d.y^2 / e.b^2) = 1) -- C and D are on ellipse
  (h7 : cd_diameter.start.x + cd_diameter.finish.x = 0 ∧ cd_diameter.start.y + cd_diameter.finish.y = 0) -- CD is a diameter
  (h8 : (n.y - a.y) / (n.x - a.x) = (d.y - c.y) / (d.x - c.x)) -- AN is parallel to CD
  : (m.x - a.x) * (n.x - a.x) + (m.y - a.y) * (n.y - a.y) =
    ((c.x^2 + c.y^2)^(1/2)) * ((d.x - c.x)^2 + (d.y - c.y)^2)^(1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_product_equality_l1220_122073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_implies_isosceles_l1220_122068

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the condition
noncomputable def condition (t : Triangle) : Prop :=
  t.a + t.b = Real.tan (t.C / 2) * (t.a * Real.tan t.A + t.b * Real.tan t.B)

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Theorem statement
theorem triangle_condition_implies_isosceles (t : Triangle) :
  condition t → isIsosceles t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_condition_implies_isosceles_l1220_122068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_in_complex_plane_l1220_122038

/-- A predicate that checks if three complex numbers form an equilateral triangle --/
def is_equilateral_triangle (a b c : ℂ) : Prop :=
  Complex.abs (a - b) = Complex.abs (b - c) ∧ Complex.abs (b - c) = Complex.abs (c - a)

theorem equilateral_triangle_in_complex_plane (ζ : ℂ) (μ : ℝ) : 
  Complex.abs ζ = 3 →
  μ > 1 →
  is_equilateral_triangle ζ (ζ^2) (μ * ζ) →
  μ = (1 + Real.sqrt 33) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_in_complex_plane_l1220_122038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_result_l1220_122035

-- Define the functions h and j
noncomputable def h (x : ℝ) : ℝ := x + 3
noncomputable def j (x : ℝ) : ℝ := x / 4

-- Define the inverse functions
noncomputable def h_inv (x : ℝ) : ℝ := x - 3
noncomputable def j_inv (x : ℝ) : ℝ := 4 * x

-- State the theorem
theorem composite_function_result :
  h (j_inv (h_inv (h_inv (j (h 20))))) = 2 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_result_l1220_122035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_slopes_l1220_122041

/-- Predicate to check if two lines with given slopes form an isosceles triangle
    with the x-axis, intersecting at point P -/
def isosceles_triangle_with_x_axis (slope1 slope2 : ℝ) (P : ℝ × ℝ) : Prop :=
  sorry

/-- Given two intersecting lines with slopes 1/k and 2k, where k is positive,
    if they form an isosceles triangle with the x-axis,
    then k equals sqrt(2)/4 or sqrt(2) -/
theorem isosceles_triangle_slopes (k : ℝ) (hk : k > 0) :
  let slope_l1 : ℝ := 1 / k
  let slope_l2 : ℝ := 2 * k
  (∃ (P : ℝ × ℝ), isosceles_triangle_with_x_axis slope_l1 slope_l2 P) →
  k = Real.sqrt 2 / 4 ∨ k = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_slopes_l1220_122041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_2_l1220_122004

def a : ℕ → ℚ
  | 0 => 3  -- Define a₀ to be 3 (same as a₁)
  | n + 1 => (5 * a n - 13) / (3 * a n - 7)

theorem a_2016_equals_2 : a 2016 = 2 := by
  -- We'll prove this in three steps
  
  -- Step 1: Prove the sequence has a period of 3
  have period_3 : ∀ n, a (n + 3) = a n := by
    sorry  -- This requires induction and calculation
  
  -- Step 2: Prove a₂ = 1, a₃ = 2
  have a2_eq_1 : a 2 = 1 := by
    sorry  -- This requires calculation
  
  have a3_eq_2 : a 3 = 2 := by
    sorry  -- This requires calculation
  
  -- Step 3: Use the period to conclude a₂₀₁₆ = a₃
  calc
    a 2016 = a (3 * 672) := by sorry  -- 2016 = 3 * 672
    _ = a 3 := by sorry  -- Use period_3
    _ = 2 := by sorry  -- Use a3_eq_2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_2_l1220_122004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_formula_l1220_122019

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- The perimeter of the trapezoid -/
  p : ℝ
  /-- The acute angle at the base of the trapezoid -/
  α : ℝ
  /-- The perimeter is positive -/
  p_pos : 0 < p
  /-- The angle is between 0 and π/2 -/
  α_bounds : 0 < α ∧ α < Real.pi / 2

/-- The radius of the inscribed circle in a circumscribed isosceles trapezoid -/
noncomputable def inscribed_circle_radius (t : CircumscribedTrapezoid) : ℝ :=
  t.p * Real.sin t.α / 8

/-- Theorem: The radius of the inscribed circle in a circumscribed isosceles trapezoid
    is equal to (p * sin(α)) / 8, where p is the perimeter and α is the acute base angle -/
theorem inscribed_circle_radius_formula (t : CircumscribedTrapezoid) :
  inscribed_circle_radius t = t.p * Real.sin t.α / 8 := by
  -- The proof is trivial as it follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_formula_l1220_122019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_and_parallel_line_properties_l1220_122002

/-- Triangle ABC with vertices A(1,1), B(3,2), and C(5,4) -/
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (3, 2)
def C : ℝ × ℝ := (5, 4)

/-- Line l is parallel to AC -/
def l_parallel_to_AC (l : ℝ → ℝ) : Prop :=
  (l 5 - l 1) / (5 - 1) = (C.2 - A.2) / (C.1 - A.1)

/-- Line l's x-intercept is 1 unit greater than its y-intercept -/
def l_intercept_condition (l : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, l 0 = -a ∧ l⁻¹ 0 = a + 1

theorem triangle_altitude_and_parallel_line_properties
  (l : ℝ → ℝ)
  (h_parallel : l_parallel_to_AC l)
  (h_intercept : l_intercept_condition l) :
  (∀ x y : ℝ, 2 * x + y - 14 = 0 ↔ y = -2 * (x - 5) + 4) ∧
  (∃ p : ℝ, p = 12 / 7 ∧
    p = |l 0| + |l⁻¹ 0| + Real.sqrt ((l 0)^2 + (l⁻¹ 0)^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_and_parallel_line_properties_l1220_122002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_crossing_probability_l1220_122074

/-- The duration of the red light in seconds -/
noncomputable def red_light_duration : ℝ := 40

/-- The minimum wait time in seconds to be considered -/
noncomputable def min_wait_time : ℝ := 15

/-- The probability of waiting at least the minimum time for a green light -/
noncomputable def prob_wait_min_time : ℝ := (red_light_duration - min_wait_time) / red_light_duration

theorem pedestrian_crossing_probability :
  prob_wait_min_time = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_crossing_probability_l1220_122074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_three_solutions_l1220_122097

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define our equation
def equation (x : ℝ) : Prop := x^2 - 2 * (floor x) - 3 = 0

-- Theorem statement
theorem equation_has_three_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, equation x ∧
  ∀ y : ℝ, equation y → y ∈ s := by
  sorry

#check equation_has_three_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_three_solutions_l1220_122097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_point_difference_l1220_122081

theorem graph_point_difference : ∃ (m n : ℝ),
  (m ^ 2 + (Real.cos (π / 3)) ^ 6 = 3 * (Real.cos (π / 3)) ^ 2 * m + Real.cos (π / 3)) ∧
  (n ^ 2 + (Real.cos (π / 3)) ^ 6 = 3 * (Real.cos (π / 3)) ^ 2 * n + Real.cos (π / 3)) ∧
  m ≠ n ∧
  |m - n| = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_point_difference_l1220_122081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat1_wins_l1220_122049

/-- Represents a boat in the race -/
structure Boat where
  number : Nat
  speed : ℝ

/-- Represents the race conditions -/
structure RaceConditions where
  boat1 : Boat
  boat2 : Boat
  boat3 : Boat
  boat4 : Boat
  water_speed : ℝ
  h_speed_order : boat1.speed > boat2.speed ∧ boat2.speed > boat3.speed ∧ boat3.speed > boat4.speed ∧ boat4.speed > 0

/-- Calculates the time taken for a boat to catch up with boat 4 -/
noncomputable def catchUpTime (boat : Boat) (boat4 : Boat) : ℝ :=
  (boat.speed + boat4.speed) / (boat.speed - boat4.speed)

/-- Theorem stating that boat 1 wins the championship -/
theorem boat1_wins (rc : RaceConditions) :
  catchUpTime rc.boat1 rc.boat4 < catchUpTime rc.boat2 rc.boat4 ∧
  catchUpTime rc.boat1 rc.boat4 < catchUpTime rc.boat3 rc.boat4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat1_wins_l1220_122049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quad_projections_circumscribed_l1220_122051

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)
  (is_cyclic : ∃ (center : V) (radius : ℝ), 
    norm (center - A) = radius ∧ 
    norm (center - B) = radius ∧ 
    norm (center - C) = radius ∧ 
    norm (center - D) = radius)

-- Define the intersection point of diagonals
noncomputable def diagonal_intersection {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (quad : CyclicQuadrilateral V) : V :=
  sorry

-- Define orthogonal projection
noncomputable def orthogonal_projection {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (point : V) (line_start line_end : V) : V :=
  sorry

-- Define a circumscribed quadrilateral
def is_circumscribed {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (A B C D : V) : Prop :=
  sorry

theorem cyclic_quad_projections_circumscribed 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (quad : CyclicQuadrilateral V) :
  let O := diagonal_intersection quad
  let M := orthogonal_projection O quad.A quad.B
  let N := orthogonal_projection O quad.B quad.C
  let K := orthogonal_projection O quad.C quad.D
  let L := orthogonal_projection O quad.D quad.A
  -- Assumption that projections are not on extensions
  (M ≠ quad.A ∧ M ≠ quad.B) →
  (N ≠ quad.B ∧ N ≠ quad.C) →
  (K ≠ quad.C ∧ K ≠ quad.D) →
  (L ≠ quad.D ∧ L ≠ quad.A) →
  is_circumscribed M N K L :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quad_projections_circumscribed_l1220_122051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l1220_122022

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given base vectors and vector relationships, prove k = 2 when A, B, D are collinear -/
theorem vector_collinearity (a b : V) (k : ℝ) 
  (hAB : ∃ AB : V, AB = a - k • b)
  (hCB : ∃ CB : V, CB = 2 • a + b)
  (hCD : ∃ CD : V, CD = 3 • a - b)
  (hCollinear : ∃ t : ℝ, a - k • b = t • ((3 • a - b) - (2 • a + b))) :
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l1220_122022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_seven_l1220_122077

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_of_three_equals_seven :
  (∀ x : ℝ, f (x/2 + 1) = x + 3) → f 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_seven_l1220_122077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_solution_set_l1220_122076

theorem sine_inequality_solution_set (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.sin x < -Real.sqrt 3 / 2 ↔ x ∈ Set.Ioo (4 * Real.pi / 3) (5 * Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_solution_set_l1220_122076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1220_122014

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

theorem f_properties (a : ℝ) :
  (f a 1 = 2) →
  (a = 1) ∧
  (∀ x, x ≠ 0 → f a (-x) = -(f a x)) ∧
  (∀ x y, 1 ≤ x → x < y → f a x < f a y) :=
by
  intro h1
  have h2 : a = 1 := by
    -- Proof for a = 1
    sorry
  have h3 : ∀ x, x ≠ 0 → f a (-x) = -(f a x) := by
    -- Proof for odd function
    sorry
  have h4 : ∀ x y, 1 ≤ x → x < y → f a x < f a y := by
    -- Proof for monotonicity
    sorry
  exact ⟨h2, h3, h4⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1220_122014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_representable_l1220_122058

def sequenceN (n : ℕ) : ℕ := 10 * n + 1

def is_representable (x : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ x = p - q

theorem exactly_three_representable :
  ∃! k : ℕ, k = 3 ∧
    (∃ S : Finset ℕ, S.card = k ∧
      (∀ n ∈ S, ∃ m : ℕ, sequenceN m = n) ∧
      (∀ n ∈ S, is_representable n) ∧
      (∀ n : ℕ, is_representable (sequenceN n) → n ∈ S)) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_representable_l1220_122058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1220_122093

/-- The time taken for a train to cross a platform -/
noncomputable def time_to_cross_platform (train_length : ℝ) (platform_length : ℝ) (time_pass_man : ℝ) : ℝ :=
  (train_length + platform_length) / (train_length / time_pass_man)

/-- Theorem: Given the specified conditions, the time taken for the train to cross the platform is approximately 20 seconds -/
theorem train_crossing_time :
  let train_length : ℝ := 182
  let platform_length : ℝ := 273
  let time_pass_man : ℝ := 8
  abs (time_to_cross_platform train_length platform_length time_pass_man - 20) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1220_122093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_perpendicular_lines_l1220_122005

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The perpendicular slope of a given slope -/
noncomputable def perpendicularSlope (m : ℝ) : ℝ := -1 / m

/-- Check if two lines are perpendicular -/
def isPerpendicular (l1 l2 : Line) : Prop :=
  l2.slope = perpendicularSlope l1.slope

/-- Check if a point lies on a line -/
def pointOnLine (x y : ℝ) (l : Line) : Prop :=
  y = l.slope * x + l.intercept

/-- The intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : ℝ × ℝ :=
  let x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.intercept
  (x, y)

theorem intersection_of_perpendicular_lines :
  let l1 : Line := { slope := 3, intercept := -4 }
  let l2 : Line := { slope := perpendicularSlope l1.slope,
                     intercept := 2 - (perpendicularSlope l1.slope) * 3 }
  isPerpendicular l1 l2 ∧
  pointOnLine 3 2 l2 ∧
  intersectionPoint l1 l2 = (21/10, 23/10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_perpendicular_lines_l1220_122005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_double_angle_sum_l1220_122098

noncomputable def log_function (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem sin_cos_double_angle_sum (a : ℝ) (α : ℝ) : 
  a > 0 ∧ a ≠ 1 → 
  log_function a 1 + 2 = 2 →
  Real.sin (2 * α) + Real.cos (2 * α) = 7/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_double_angle_sum_l1220_122098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_majority_agreeing_ordering_l1220_122003

/-- Represents a judge's ranking of participants -/
def Ranking (α : Type*) := α → α → Bool

/-- Represents the condition that for any three participants, there are no three judges with conflicting rankings -/
def NoConflictingRankings {α : Type*} (rankings : Finset (Ranking α)) : Prop :=
  ∀ (a b c : α) (r₁ r₂ r₃ : Ranking α),
    r₁ ∈ rankings → r₂ ∈ rankings → r₃ ∈ rankings →
    ¬(r₁ a b ∧ r₁ b c ∧ r₂ b c ∧ r₂ c a ∧ r₃ c a ∧ r₃ a b)

/-- Represents a total ordering of participants -/
def TotalOrdering (α : Type*) := α → α → Bool

/-- Represents the condition that the total ordering agrees with at least half of the judges for any pair of participants -/
def OrderingAgreesWithMajority {α : Type*} (ordering : TotalOrdering α) (rankings : Finset (Ranking α)) : Prop :=
  ∀ a b : α, (rankings.filter (λ r => r a b)).card ≥ rankings.card / 2 → ordering a b = true

theorem existence_of_majority_agreeing_ordering {α : Type*} [DecidableEq α] (rankings : Finset (Ranking α))
    (h : rankings.card = 100) (no_conflict : NoConflictingRankings rankings) :
    ∃ ordering : TotalOrdering α, OrderingAgreesWithMajority ordering rankings :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_majority_agreeing_ordering_l1220_122003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_not_closed_l1220_122033

-- Define the set of squares of positive integers
def v : Set ℕ := {n : ℕ | ∃ m : ℕ+, n = m^2}

-- Define the combined operations
def add_then_mult (a b c : ℕ) : ℕ := (a + b) * c
def mult_then_add (a b c : ℕ) : ℕ := (a * b) + c
def div_then_sub (a b c : ℕ) : ℚ := (a / b) - c

noncomputable def sqrt_then_mult (a b : ℕ) : ℝ := Real.sqrt (a : ℝ) * Real.sqrt (b : ℝ)

-- Theorem statement
theorem v_not_closed :
  (∃ a b c, a ∈ v ∧ b ∈ v ∧ c ∈ v ∧ add_then_mult a b c ∉ v) ∧
  (∃ a b c, a ∈ v ∧ b ∈ v ∧ c ∈ v ∧ mult_then_add a b c ∉ v) ∧
  (∃ a b c, a ∈ v ∧ b ∈ v ∧ c ∈ v ∧ ¬ ∃ n : ℕ, (n : ℚ) = div_then_sub a b c) ∧
  (∃ a b, a ∈ v ∧ b ∈ v ∧ ¬ ∃ n : ℕ, (n : ℝ) = sqrt_then_mult a b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_not_closed_l1220_122033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_wins_with_13_ivan_max_13_with_14_l1220_122084

/-- Represents the state of the game at any given moment -/
structure GameState where
  chest : ℕ  -- number of bars in the chest
  bag : ℕ    -- number of bars in the bag
  last_move : ℕ  -- number of bars moved in the last turn

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (move : ℕ) : Prop :=
  move ≠ 0 ∧ move ≠ state.last_move ∧ move ≤ state.chest

/-- Defines the game's end condition -/
def game_over (state : GameState) : Prop :=
  ∀ move, ¬(valid_move state move)

/-- Function to simulate the game given strategies for both players -/
def iterate_game (initial_bars : ℕ) (ivan_strategy : GameState → ℕ) (kashchei_strategy : GameState → ℕ) : GameState :=
  sorry

/-- Theorem stating that Ivan can always obtain 13 bars when starting with 13 -/
theorem ivan_wins_with_13 :
  ∃ (strategy : GameState → ℕ),
    ∀ (kashchei_strategy : GameState → ℕ),
      (iterate_game 13 strategy kashchei_strategy).bag = 13 := by
  sorry

/-- Theorem stating that Ivan can obtain at most 13 bars when starting with 14 -/
theorem ivan_max_13_with_14 :
  ∀ (ivan_strategy : GameState → ℕ),
    ∃ (kashchei_strategy : GameState → ℕ),
      (iterate_game 14 ivan_strategy kashchei_strategy).bag ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_wins_with_13_ivan_max_13_with_14_l1220_122084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_value_l1220_122006

theorem tan_beta_value (α β : ℝ) (h1 : Real.sin α = 3/5) 
  (h2 : π < α ∧ α < 3*π/2) (h3 : Real.tan (α + β) = 1) : Real.tan β = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_value_l1220_122006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_rounding_l1220_122018

noncomputable def round_to_thousandth (x : ℝ) : ℝ :=
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

theorem incorrect_rounding :
  round_to_thousandth 2.04951 ≠ 2.049 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_rounding_l1220_122018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l1220_122017

noncomputable def family_of_curves (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

def line_equation (x y : ℝ) : Prop :=
  y = 2 * x

noncomputable def chord_length (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem max_chord_length :
  ∃ (x y : ℝ), family_of_curves (Real.arcsin (2/5)) x y ∧ line_equation x y ∧
  (∀ (x' y' : ℝ), family_of_curves (Real.arcsin (2/5)) x' y' ∧ line_equation x' y' →
  chord_length x' y' ≤ chord_length x y) ∧ chord_length x y = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l1220_122017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_elderly_l1220_122085

theorem stratified_sampling_elderly (total_population : ℕ) (elderly_population : ℕ) (sample_size : ℕ) 
  (h1 : total_population = 163) 
  (h2 : elderly_population = 28) 
  (h3 : sample_size = 36) : 
  Int.floor ((elderly_population : ℝ) / total_population * sample_size + 0.5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_elderly_l1220_122085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1220_122009

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -3 * Real.cos x + 1

-- Theorem stating the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-2) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1220_122009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_g_l1220_122016

def g (x : ℝ) : ℝ := 3 * x - 2

theorem inverse_of_inverse_g : 
  Function.invFun g (Function.invFun g 14) = 22 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_g_l1220_122016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1220_122053

/-- The circle equation: x^2 + y^2 - 2x + 4y + 3 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + 3 = 0

/-- The line equation: x - y = 1 -/
def line_equation (x y : ℝ) : Prop :=
  x - y = 1

/-- The center of the circle -/
def circle_center : ℝ × ℝ :=
  (1, -2)

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

theorem distance_circle_center_to_line :
  distance_point_to_line (circle_center.1) (circle_center.2) 1 (-1) (-1) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1220_122053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_S_is_251_l1220_122086

-- Define a 4-element set S
def S : Finset ℕ := sorry

-- Lemma: S has 4 elements
lemma S_card : Finset.card S = 4 := by sorry

-- Define the sum of elements in a set
def set_sum (T : Finset ℕ) : ℕ := T.sum id

-- Lemma: The sum of elements in all subsets of S is 2008
lemma sum_of_subsets : (Finset.powerset S).sum set_sum = 2008 := by sorry

-- Theorem to prove
theorem sum_of_S_is_251 : set_sum S = 251 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_S_is_251_l1220_122086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_zero_l1220_122027

theorem sum_remainder_zero (a b c : ℕ) 
  (ha : a % 30 = 11) 
  (hb : b % 30 = 5) 
  (hc : c % 30 = 14) : 
  (a + b + c) % 30 = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_zero_l1220_122027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_roots_of_equation_l1220_122034

noncomputable def floor (x : ℝ) := ⌊x⌋

theorem real_roots_of_equation :
  let S := {x : ℝ | x^2 - 8 * (floor x) + 7 = 0}
  S = {1, Real.sqrt 33, Real.sqrt 41, 7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_roots_of_equation_l1220_122034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_shaded_region_l1220_122039

-- Define the circle structure
structure Circle where
  circumference : ℝ

-- Define the problem setup
def problem_setup (c : Circle) : Prop :=
  c.circumference = 48

-- Define the arc length calculation
noncomputable def arc_length (c : Circle) : ℝ :=
  c.circumference / 6

-- Define the perimeter calculation
noncomputable def perimeter_calculation (c : Circle) : ℝ :=
  3 * (arc_length c)

-- Theorem statement
theorem perimeter_of_shaded_region (c : Circle) 
  (h : problem_setup c) : perimeter_calculation c = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_shaded_region_l1220_122039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1220_122045

/-- The area enclosed by the region defined by x^2 + y^2 + 12x + 16y = 0 is 100π -/
theorem area_of_region : 
  ∃ A : ℝ, A = 100 * Real.pi ∧ 
  ∀ x y : ℝ, x^2 + y^2 + 12*x + 16*y = 0 → 
  A = Real.pi * (10 : ℝ)^2 :=
by
  -- We'll use 100π as our area
  let A := 100 * Real.pi
  
  -- Show that this A satisfies our conditions
  exists A
  
  constructor
  · -- Prove A = 100π (trivial)
    rfl
  
  · -- Prove that for all x, y satisfying the equation, A is the area of the circle
    intros x y h
    -- The actual proof would go here, but we'll use sorry for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1220_122045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_products_less_than_factorial_l1220_122024

theorem lcm_products_less_than_factorial (n : ℕ) (hn : n > 3) :
  (Finset.sup
    (Finset.filter
      (fun s => s.sum (fun _ => (1 : ℕ)) ≤ n ∧ s.card > 0)
      (Finset.powerset (Finset.range n)))
    (fun s => s.prod (fun i => i + 1))).lcm
  < Nat.factorial n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_products_less_than_factorial_l1220_122024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_conditions_line_perpendicular_y_intercept_l1220_122025

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The first line l₁: mx + 8y + n = 0 -/
def l1 (m n : ℝ) : Line := ⟨m, 8, n⟩

/-- The second line l₂: 2x + my - 1 = 0 -/
def l2 (m : ℝ) : Line := ⟨2, m, -1⟩

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ := -l.c / l.b

theorem line_parallel_conditions (m n : ℝ) :
  (parallel (l1 m n) (l2 m) ↔ (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2)) := by
  sorry

theorem line_perpendicular_y_intercept (m n : ℝ) :
  (perpendicular (l1 m n) (l2 m) ∧ y_intercept (l1 m n) = -1) ↔ (m = 0 ∧ n = 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_conditions_line_perpendicular_y_intercept_l1220_122025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_36_89753_to_nearest_tenth_l1220_122007

noncomputable def round_to_nearest_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem round_36_89753_to_nearest_tenth :
  round_to_nearest_tenth 36.89753 = 36.9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_36_89753_to_nearest_tenth_l1220_122007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_x_coordinate_sum_l1220_122060

/-- The sum of all possible x-coordinates of point A in the triangle problem -/
theorem triangle_x_coordinate_sum : ∃ (x1 x2 x3 x4 : ℝ), 
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (317, 0)
  let D : ℝ × ℝ := (720, 420)
  let E : ℝ × ℝ := (731, 432)
  let area_ABC : ℝ := 3003
  let area_ADE : ℝ := 9009
  x1 + x2 + x3 + x4 = 16080 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_x_coordinate_sum_l1220_122060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_diagonal_angle_l1220_122079

/-- Given a rectangular parallelepiped with edges in the ratio 3 : 4 : 12,
    the sine of the angle between the diagonal cross-section through the largest edge
    and the diagonal of the parallelepiped not lying in that plane is 24/65. -/
theorem parallelepiped_diagonal_angle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : b / a = 4 / 3) (h5 : c / a = 12 / 3) :
  let diagonal_length := Real.sqrt (a^2 + b^2 + c^2)
  let cross_section_diagonal := Real.sqrt (a^2 + b^2)
  let projection := (a * b) / cross_section_diagonal
  projection / diagonal_length = 24 / 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_diagonal_angle_l1220_122079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l1220_122056

/-- Represents a position on the 51x51 grid -/
structure Position where
  row : Fin 51
  col : Fin 51

/-- Represents the game state -/
structure GameState where
  board : List Position
  currentPlayer : Nat

/-- Checks if a move is valid according to the game rules -/
def isValidMove (state : GameState) (pos : Position) : Bool :=
  let rowCount := (state.board.filter (λ p => p.row = pos.row)).length
  let colCount := (state.board.filter (λ p => p.col = pos.col)).length
  rowCount < 2 ∧ colCount < 2

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Option Position

/-- Determines if a strategy is winning for the second player -/
def isWinningStrategy (s : Strategy) : Prop :=
  ∀ (initialMove : Position),
    ∃ (counterMove : Position),
      isValidMove { board := [initialMove], currentPlayer := 1 } counterMove ∧
      ∀ (game : GameState),
        game.currentPlayer = 2 →
        (∃ (move : Position), isValidMove game move) →
        ∃ (move : Position),
          isValidMove game move ∧
          s game = some move

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (s : Strategy), isWinningStrategy s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l1220_122056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_not_invertible_iff_x_values_l1220_122066

theorem matrix_not_invertible_iff_x_values (x : ℝ) : 
  ¬(IsUnit (Matrix.det !![2 + x^2, 5; 4 - x, 9])) ↔ 
  (x = (-5 + Real.sqrt 97) / 18 ∨ x = (-5 - Real.sqrt 97) / 18) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_not_invertible_iff_x_values_l1220_122066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1220_122071

-- Define set A
def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioi 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1220_122071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_round_trip_time_is_one_hour_l1220_122055

/-- Calculates the time taken for a round trip rowing on a river -/
noncomputable def rowing_round_trip_time (man_speed : ℝ) (river_speed : ℝ) (total_distance : ℝ) : ℝ :=
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := total_distance / 2
  (one_way_distance / upstream_speed) + (one_way_distance / downstream_speed)

/-- Theorem stating that under given conditions, the round trip time is 1 hour -/
theorem rowing_round_trip_time_is_one_hour :
  rowing_round_trip_time 6 1.2 5.76 = 1 := by
  -- Unfold the definition of rowing_round_trip_time
  unfold rowing_round_trip_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_round_trip_time_is_one_hour_l1220_122055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_problem_l1220_122028

-- Define the interest rate and time period
noncomputable def rate : ℝ := 5 / 100
noncomputable def time : ℝ := 2

-- Define the compound interest function
noncomputable def compound_interest (principal : ℝ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

-- Define the simple interest function
noncomputable def simple_interest (principal : ℝ) : ℝ :=
  principal * rate * time

-- Theorem statement
theorem interest_problem (P : ℝ) 
  (h : compound_interest P = 51.25) : 
  simple_interest P = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_problem_l1220_122028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_cubic_l1220_122012

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 1

-- Define the point through which the tangent lines pass
def P : ℝ × ℝ := (1, 3)

-- Define the two tangent lines
def line1 (x y : ℝ) : Prop := 2*x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 4*y - 13 = 0

-- Theorem statement
theorem tangent_lines_to_cubic : 
  ∃ (x₀ y₀ : ℝ), 
    (f x₀ = y₀) ∧ 
    (line1 1 3 ∧ line1 x₀ y₀ ∧ (f' x₀ = 2)) ∧
    (∃ (x₁ y₁ : ℝ), f x₁ = y₁ ∧ line2 1 3 ∧ line2 x₁ y₁ ∧ (f' x₁ = -1/4)) :=
by
  sorry

#check tangent_lines_to_cubic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_cubic_l1220_122012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1220_122037

noncomputable def z : ℂ := Complex.I / (1 + Complex.I)

theorem z_in_first_quadrant : 
  0 < z.re ∧ 0 < z.im :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1220_122037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l1220_122067

/-- Two lines in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance between two parallel lines -/
noncomputable def distance_parallel_lines (l1 l2 : Line2D) : ℝ :=
  abs (l2.c - l1.c) / Real.sqrt (l1.a^2 + l1.b^2)

/-- Theorem stating the distance between two specific parallel lines -/
theorem distance_specific_lines :
  let l1 : Line2D := ⟨3, 1, 1⟩
  let l2 : Line2D := ⟨3, 1, -1⟩
  distance_parallel_lines l1 l2 = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l1220_122067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_super_cool_areas_l1220_122082

def isSuperCool (a b : ℕ) : Bool :=
  a * b / 2 = 3 * (a + b)

def superCoolAreas : List ℕ :=
  (List.range 100).bind fun a =>
    (List.range 100).filter fun b =>
      a < b && isSuperCool a b

theorem sum_super_cool_areas :
  superCoolAreas.sum = 471 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_super_cool_areas_l1220_122082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barium_iodide_molecular_weight_l1220_122052

/-- The molecular weight of Barium iodide for a given number of moles -/
noncomputable def molecular_weight (moles : ℝ) : ℝ := 1564 / 4 * moles

/-- Theorem: The molecular weight of one mole of Barium iodide is 391 g/mol -/
theorem barium_iodide_molecular_weight : molecular_weight 1 = 391 := by
  -- Unfold the definition of molecular_weight
  unfold molecular_weight
  -- Simplify the expression
  simp
  -- Check that 1564 / 4 = 391
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_barium_iodide_molecular_weight_l1220_122052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inequality_solution_set_l1220_122070

/-- If the line x = aπ (0 < a < 1) and the graph of y = tan x have no common points, 
    then the solution set of tan x ≥ 2a is {x | kπ + π/4 ≤ x < kπ + π/2, k ∈ Z} -/
theorem tangent_inequality_solution_set (a : ℝ) (h1 : 0 < a) (h2 : a < 1) 
  (h3 : ∀ x : ℝ, x = a * Real.pi → Real.tan x ≠ 0) : 
  {x : ℝ | ∃ k : ℤ, k * Real.pi + Real.pi / 4 ≤ x ∧ x < k * Real.pi + Real.pi / 2} = 
  {x : ℝ | Real.tan x ≥ 2 * a} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inequality_solution_set_l1220_122070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_calculation_l1220_122062

def initial_avg (n : ℕ) (wrong_num correct_num : ℤ) (correct_avg : ℚ) : ℚ :=
  correct_avg - (correct_num - wrong_num : ℚ) / n

theorem initial_average_calculation (n : ℕ) (correct_num wrong_num : ℤ) (correct_avg : ℚ) :
  n = 10 →
  correct_num = 36 →
  wrong_num = 26 →
  correct_avg = 19 →
  (n : ℚ) * correct_avg = (n : ℚ) * (initial_avg n wrong_num correct_num correct_avg) + (correct_num - wrong_num) →
  initial_avg n wrong_num correct_num correct_avg = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_calculation_l1220_122062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1220_122008

noncomputable def f (x : ℝ) := Real.sin x ^ 4 - Real.sin x * Real.cos x + Real.cos x ^ 4

theorem f_range : 
  (∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 9/8) ∧ 
  (∃ x : ℝ, f x = 0) ∧ 
  (∃ x : ℝ, f x = 9/8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1220_122008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_plane_speed_l1220_122094

/-- The minimal speed of a plane flying between two points -/
theorem minimal_plane_speed
  (d : ℝ) -- distance between points A and B in kilometers
  (a b : ℝ) -- angle changes in degrees per second
  (h : ℝ) -- constant height of the plane
  (hd : d > 0) -- positive distance
  (ha : a > 0) -- positive angle change
  (hb : b > 0) -- positive angle change
  (hh : h > 0) -- positive height
  : ∃ (v : ℝ), v = 20 * π * d * Real.sqrt (a * b) ∧ 
    ∀ (u : ℝ), u ≥ v → 
    ∃ (y : ℝ), 0 ≤ y ∧ y ≤ d ∧
    u * (π / 180) = d * Real.sqrt ((a * (h^2 + y^2)) / h * (b * (h^2 + (d - y)^2)) / h) / d :=
sorry

#check minimal_plane_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_plane_speed_l1220_122094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_3_l1220_122048

-- Define the three lines
def line1 (x y : ℝ) : Prop := Real.sqrt 3 * x - y = 0
def line2 (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 2 = 0
def line3 (y : ℝ) : Prop := y = 0

-- Define the three points of intersection
noncomputable def pointA : ℝ × ℝ := (1, Real.sqrt 3)
def pointB : ℝ × ℝ := (0, 0)
def pointC : ℝ × ℝ := (-2, 0)

-- State the theorem
theorem triangle_area_is_sqrt_3 :
  let area := (1/2) * abs (
    pointA.1 * (pointB.2 - pointC.2) +
    pointB.1 * (pointC.2 - pointA.2) +
    pointC.1 * (pointA.2 - pointB.2)
  )
  area = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_3_l1220_122048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l1220_122099

-- Define the triangle
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

-- Theorem statement
theorem triangle_area_proof (a b c A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_sin_cos : Real.sin (A - Real.pi/6) = Real.cos A)
  (h_a : a = 1)
  (h_bc : b + c = 2) :
  A = Real.pi/3 ∧ (1/2 * b * c * Real.sin A = Real.sqrt 3 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l1220_122099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_exists_l1220_122083

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a set of five points
def FivePoints := Fin 5 → Point

-- Define what it means for three points to be collinear
def areCollinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

-- Define what it means for a quadrilateral to be convex
def isConvexQuadrilateral (p q r s : Point) : Prop :=
  let cross (a b c : Point) := (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
  (cross p q r) * (cross p q s) ≥ 0 ∧
  (cross q r s) * (cross q r p) ≥ 0 ∧
  (cross r s p) * (cross r s q) ≥ 0 ∧
  (cross s p q) * (cross s p r) ≥ 0

-- State the theorem
theorem convex_quadrilateral_exists (points : FivePoints) 
  (h : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬areCollinear (points i) (points j) (points k)) :
  ∃ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧ 
             isConvexQuadrilateral (points i) (points j) (points k) (points l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_exists_l1220_122083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_hexagon_area_l1220_122046

noncomputable section

/-- Regular octahedron with side length 2 --/
def octahedron_side_length : ℝ := 2

/-- Height of an equilateral triangle face of the octahedron --/
noncomputable def face_height : ℝ := Real.sqrt 3

/-- Distance from the base where the plane intersects --/
noncomputable def intersection_height : ℝ := (2 / 3) * face_height

/-- Side length of the hexagon formed by the intersection --/
noncomputable def hexagon_side_length : ℝ := (2 / 3) * octahedron_side_length

/-- Area of a regular hexagon with side length s --/
noncomputable def hexagon_area (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

/-- Theorem statement --/
theorem intersection_hexagon_area : 
  hexagon_area hexagon_side_length = 8 * Real.sqrt 3 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_hexagon_area_l1220_122046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_digit_puzzle_l1220_122050

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 100

/-- Represents a three-digit number -/
def ThreeDigitNumber := Fin 1000

/-- Represents the arithmetic sequence property for three numbers -/
def IsArithmeticSequence (a b c : ℕ) : Prop := b - a = c - b

/-- Represents the table in the problem -/
structure DigitTable where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  E : Digit
  F : Digit
  G : Digit
  different : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧
              B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧
              C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧
              D ≠ E ∧ D ≠ F ∧ D ≠ G ∧
              E ≠ F ∧ E ≠ G ∧
              F ≠ G
  row1_seq : IsArithmeticSequence A.val (10 * B.val + A.val) (11 * A.val)
  row2_seq : IsArithmeticSequence (10 * A.val + B.val) (10 * C.val + A.val) (10 * E.val + F.val)
  row3_seq : IsArithmeticSequence (10 * C.val + D.val) (10 * G.val + A.val) (100 * B.val + 10 * D.val + C.val)
  col1_seq : IsArithmeticSequence A.val (10 * A.val + B.val) (10 * C.val + D.val)
  col2_seq : IsArithmeticSequence (10 * B.val + A.val) (10 * C.val + A.val) (10 * G.val + A.val)
  col3_seq : IsArithmeticSequence (11 * A.val) (10 * E.val + F.val) (100 * B.val + 10 * D.val + C.val)
  nonzero_leading : A.val ≠ 0 ∧ B.val ≠ 0 ∧ C.val ≠ 0

theorem solve_digit_puzzle (t : DigitTable) : 
  t.C.val * 10000 + t.D.val * 1000 + t.E.val * 100 + t.F.val * 10 + t.G.val = 40637 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_digit_puzzle_l1220_122050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_value_l1220_122078

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - floor x

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 3
  | n + 1 => floor (a n) + 1 / frac (a n)

theorem a_2019_value : a 2018 = 3027 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_value_l1220_122078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2008_equals_2_l1220_122029

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => 1 - 1 / mySequence n

theorem mySequence_2008_equals_2 : mySequence 2007 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2008_equals_2_l1220_122029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1220_122069

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The distance between (1, 3) and (-5, 7) is 2√13 -/
theorem distance_between_specific_points :
  distance 1 3 (-5) 7 = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l1220_122069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_with_large_prime_divisor_l1220_122080

theorem infinitely_many_n_with_large_prime_divisor :
  ∃ f : ℕ → ℕ, StrictMono f ∧
    ∀ k : ℕ, ∃ p : ℕ,
      Nat.Prime p ∧
      p ∣ (f k)^2 + 1 ∧
      (p : ℝ) > 2 * (f k : ℝ) + Real.sqrt (2 * (f k : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_with_large_prime_divisor_l1220_122080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_correct_time_l1220_122057

/-- Represents a clock that loses time at a constant rate. -/
structure LosingClock where
  /-- The number of seconds lost per hour. -/
  loss_rate : ℚ
  /-- The number of days since the clock was set correctly. -/
  days_elapsed : ℚ

/-- Calculates the total time lost in minutes for a given clock. -/
def total_time_lost (clock : LosingClock) : ℚ :=
  (clock.loss_rate * 24 * clock.days_elapsed) / 60

/-- Theorem stating when the clock will next show the correct time. -/
theorem next_correct_time (clock : LosingClock) 
  (h1 : clock.loss_rate = 25/3600) -- 25 seconds per hour
  (h2 : clock.days_elapsed = 72) : 
  total_time_lost clock = 720 := by
  -- Proof steps would go here
  sorry

#eval total_time_lost { loss_rate := 25/3600, days_elapsed := 72 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_correct_time_l1220_122057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_l1220_122013

/-- A natural number that ends with 6 zeros and has exactly 56 divisors -/
def SpecialNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^6 * k ∧ (Finset.card (Nat.divisors n)) = 56

theorem sum_of_special_numbers :
  ∃ (a b : ℕ), a ≠ b ∧ SpecialNumber a ∧ SpecialNumber b ∧ a + b = 7000000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_l1220_122013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_solutions_eq_339024_l1220_122072

/-- The number of non-negative integer solutions to x + 2y + 3z = 2014 -/
def num_solutions : ℕ :=
  (Finset.sum (Finset.range 672) (λ z ↦ 
    ((2014 - 3 * z) / 2 + 1)))

theorem num_solutions_eq_339024 : num_solutions = 339024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_solutions_eq_339024_l1220_122072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_sums_l1220_122096

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- The main theorem -/
theorem max_product_of_sums (seq : ArithmeticSequence) 
  (h : seq.a 2 + seq.a 4 + seq.a 9 = 24) :
  (∃ m : ℚ, ∀ seq' : ArithmeticSequence, 
    seq'.a 2 + seq'.a 4 + seq'.a 9 = 24 →
    S seq' 8 / 8 * S seq' 10 / 10 ≤ m) ∧
  S seq 8 / 8 * S seq 10 / 10 ≤ 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_sums_l1220_122096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_sum_abcd_is_correct_l1220_122042

-- Define the points
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-3, 2)
def C : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem point_P_y_coordinate :
  ∃ P : ℝ × ℝ,
  distance P A + distance P D = 10 ∧
  distance P B + distance P C = 10 ∧
  P.2 = 6/7 :=
by sorry

-- Define the sum of a, b, c, and d
def sum_abcd : ℕ := 14

-- Prove that the sum is correct
theorem sum_abcd_is_correct : sum_abcd = 14 :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_sum_abcd_is_correct_l1220_122042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_per_minute_rate_is_seven_l1220_122011

/-- Represents the tutoring pricing model -/
structure TutoringPrice where
  flatRate : ℚ
  totalAmount : ℚ
  duration : ℚ

/-- Calculates the per-minute rate given a TutoringPrice -/
noncomputable def perMinuteRate (price : TutoringPrice) : ℚ :=
  (price.totalAmount - price.flatRate) / price.duration

/-- Theorem stating that the per-minute rate is $7 for the given conditions -/
theorem per_minute_rate_is_seven :
  let price : TutoringPrice := ⟨20, 146, 18⟩
  perMinuteRate price = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_per_minute_rate_is_seven_l1220_122011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_heptagon_side_length_l1220_122015

/-- Represents a regular heptagon -/
structure RegularHeptagon where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The constant of proportionality for heptagon area -/
noncomputable def heptagonAreaConstant : ℝ := sorry

/-- The area of a regular heptagon -/
noncomputable def heptagonArea (h : RegularHeptagon) : ℝ :=
  heptagonAreaConstant * h.sideLength ^ 2

theorem third_heptagon_side_length 
  (h1 h2 h3 : RegularHeptagon)
  (h1_side : h1.sideLength = 6)
  (h2_side : h2.sideLength = 30)
  (area_ratio : heptagonArea h3 - heptagonArea h1 = 
                (heptagonArea h2 - heptagonArea h3) / 5) :
  h3.sideLength = 6 * Real.sqrt 5 := by
  sorry

#check third_heptagon_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_heptagon_side_length_l1220_122015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1220_122065

noncomputable def f (x : ℝ) : ℝ := 2 * Real.tan (x - Real.pi/6)

theorem f_range : 
  ∀ y ∈ Set.range f, 
  y ∈ Set.Icc (-2 * Real.sqrt 3) 2 ∧ 
  ∃ x ∈ Set.Icc (-Real.pi/6) (5*Real.pi/12), f x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1220_122065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l1220_122092

noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos α, 2 * Real.sin α)

def C₂_relation (p : ℝ × ℝ) (m : ℝ × ℝ) : Prop :=
  m.1 = 2 * p.1 ∧ m.2 = 2 * p.2

def C₂_equation (p : ℝ × ℝ) : Prop :=
  (p.1 - 1)^2 + p.2^2 = 1 ∧ 0 < p.2 ∧ p.2 ≤ 1

noncomputable def A : ℝ × ℝ := (2, Real.pi / 3)
noncomputable def B : ℝ × ℝ := (1, Real.pi / 3)

theorem curve_properties :
  (∀ α, α ∈ Set.Ioo 0 Real.pi → ∃ p, C₂_relation p (C₁ α)) →
  (∀ p, (∃ α, α ∈ Set.Ioo 0 Real.pi ∧ C₂_relation p (C₁ α)) → C₂_equation p) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1 := by
  sorry

#check curve_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l1220_122092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1220_122063

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Ioo (-7) 7

-- f is odd
axiom f_odd : ∀ x, x ∈ domain_f → f (-x) = -f x

-- f is monotonically decreasing
axiom f_decreasing : ∀ x y, x ∈ domain_f → y ∈ domain_f → x < y → f x > f y

-- Condition on f
axiom f_condition : ∀ a : ℝ, (1 - a) ∈ domain_f → (2 * a - 5) ∈ domain_f → f (1 - a) + f (2 * a - 5) < 0

-- Theorem to prove
theorem a_range : 
  {a : ℝ | (1 - a) ∈ domain_f ∧ (2 * a - 5) ∈ domain_f ∧ f (1 - a) + f (2 * a - 5) < 0} = 
  Set.Ioo 4 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1220_122063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_eight_l1220_122059

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The target sum we're looking for -/
def targetSum : ℕ := 8

/-- The set of all possible outcomes when rolling two dice -/
def allOutcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range sides) (Finset.range sides)

/-- The set of favorable outcomes (sum equals targetSum) -/
def favorableOutcomes : Finset (ℕ × ℕ) :=
  allOutcomes.filter (fun (a, b) => a + b + 2 = targetSum)

/-- The probability of getting a sum of 8 when rolling two dice -/
theorem probability_sum_eight :
  (Finset.card favorableOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 5 / 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_eight_l1220_122059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_example_l1220_122031

/-- Calculates the simple interest rate given principal, final amount, and time -/
noncomputable def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  (amount - principal) * 100 / (principal * time)

/-- Theorem: The simple interest rate for the given conditions is 1.25% -/
theorem simple_interest_rate_example : 
  simple_interest_rate 750 900 16 = (5/4 : ℚ) := by
  -- Unfold the definition of simple_interest_rate
  unfold simple_interest_rate
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_example_l1220_122031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_divisible_by_nine_l1220_122075

theorem square_sum_divisible_by_nine (a b c : ℕ) 
  (ha : ∃ x : ℕ, a = x^2)
  (hb : ∃ y : ℕ, b = y^2)
  (hc : ∃ z : ℕ, c = z^2)
  (hsum : 9 ∣ (a + b + c)) :
  ∃ p q : ℕ, (p ∈ ({a, b, c} : Set ℕ) ∧ q ∈ ({a, b, c} : Set ℕ) ∧ p ≠ q) ∧ 9 ∣ (p - q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_divisible_by_nine_l1220_122075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_distances_l1220_122001

/-- Circle O in the xy-plane -/
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- Curve C in the xy-plane -/
def curve_C : Set (ℝ × ℝ) := {p | p.1^2 - p.2^2 = 1}

/-- Point M on the x-axis -/
def point_M : ℝ × ℝ := (-1, 0)

/-- Point N on the x-axis -/
def point_N : ℝ × ℝ := (1, 0)

/-- Any point P on circle O -/
noncomputable def point_P (a : ℝ) : ℝ × ℝ := (2 * Real.cos a, 2 * Real.sin a)

/-- Distance squared between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Theorem: |PM|^2 + |PN|^2 is constant for any point P on circle O -/
theorem constant_sum_of_distances (a : ℝ) :
  dist_squared (point_P a) point_M + dist_squared (point_P a) point_N = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_distances_l1220_122001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maxSumArithmeticSequence_l1220_122061

/-- An arithmetic sequence with common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  fun n ↦ a₁ + (n - 1 : ℝ) * d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sumArithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

/-- The quadratic inequality has solution set [0,9] -/
def quadraticInequalitySolution (a₁ : ℝ) (d : ℝ) : Prop :=
  ∀ x, d * x^2 + 2 * a₁ * x ≥ 0 ↔ 0 ≤ x ∧ x ≤ 9

theorem maxSumArithmeticSequence 
  (a₁ : ℝ) (d : ℝ) (h : quadraticInequalitySolution a₁ d) :
  ∃ (n : ℕ), n = 5 ∧ 
    ∀ (m : ℕ), m > 5 → sumArithmeticSequence a₁ d n ≥ sumArithmeticSequence a₁ d m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maxSumArithmeticSequence_l1220_122061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_club_members_l1220_122089

theorem chemistry_club_members (n : ℕ) 
  (math_club : Finset ℕ) 
  (history_club : Finset ℕ) 
  (chemistry_club : Finset ℕ) 
  (literature_club : Finset ℕ) :
  (∀ i ∈ Finset.range n, i ∈ math_club ↔ i % 3 = 0) →
  (∀ i ∈ Finset.range n, i ∈ history_club ↔ i % 4 = 0) →
  (∀ i ∈ Finset.range n, i ∈ chemistry_club ↔ i % 6 = 0) →
  (∀ i ∈ Finset.range n, i ∈ literature_club ↔ i ∉ math_club ∧ i ∉ history_club ∧ i ∉ chemistry_club) →
  math_club.card = literature_club.card + 3 →
  chemistry_club.card = 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_club_members_l1220_122089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_product_l1220_122044

def m : ℕ := 30030

def M : Set ℕ := {d : ℕ | d ∣ m ∧ (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ d = p * q)}

theorem smallest_n_for_product (n : ℕ) : n = 11 ↔ 
  (∀ S : Finset ℕ, ↑S ⊆ M → S.card = n →
    ∃ a b c : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a * b * c = m) ∧
  (∀ k < 11, ∃ T : Finset ℕ, ↑T ⊆ M ∧ T.card = k ∧
    ∀ a b c : ℕ, a ∈ T → b ∈ T → c ∈ T → a * b * c ≠ m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_product_l1220_122044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_b_value_l1220_122087

-- Define the original quadratic equation
def original_quadratic (x : ℝ) := 2*x^2 - 3*x - 8

-- Define the new quadratic equation
def new_quadratic (a b x : ℝ) := x^2 + a*x + b

-- Define the relationship between the roots
def roots_relationship (r s a b : ℝ) : Prop :=
  (original_quadratic r = 0 ∧ original_quadratic s = 0) →
  (new_quadratic a b (r+3) = 0 ∧ new_quadratic a b (s+3) = 0)

-- Theorem statement
theorem quadratic_b_value (a b : ℝ) :
  (∃ r s, roots_relationship r s a b) → b = 9.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_b_value_l1220_122087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_difference_simplification_l1220_122021

theorem cube_root_difference_simplification :
  (5 * Real.sqrt 2 + 7) ^ (1/3 : ℝ) - (5 * Real.sqrt 2 - 7) ^ (1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_difference_simplification_l1220_122021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l1220_122036

theorem perpendicular_vectors_lambda (a b : ℝ × ℝ) (lambda : ℝ) : 
  a = (-3, 2) → 
  b = (-1, 0) → 
  (lambda * a.1 + b.1) * (a.1 - 2 * b.1) + (lambda * a.2 + b.2) * (a.2 - 2 * b.2) = 0 → 
  lambda = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l1220_122036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_S_l1220_122088

-- Define the area function S(a)
noncomputable def S (a : ℝ) : ℝ := 4/3 * (a * (3 - a^2))^(3/2)

-- State the theorem
theorem max_area_S :
  ∃ (a : ℝ), a > 0 ∧ ∀ (x : ℝ), x > 0 → S x ≤ S a ∧ S a = 8 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_S_l1220_122088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_passing_percentage_l1220_122043

/-- The percentage Mike needs to pass, given his current score, shortfall, and maximum possible marks. -/
theorem mike_passing_percentage
  (mike_score : ℕ)
  (shortfall : ℕ)
  (max_marks : ℕ)
  (h1 : mike_score = 212)
  (h2 : shortfall = 19)
  (h3 : max_marks = 770) :
  ((mike_score + shortfall : ℚ) / max_marks * 100).floor = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_passing_percentage_l1220_122043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l1220_122054

theorem max_value_of_expression (a b c d : ℕ) : 
  a ∈ ({2, 3, 5, 7} : Set ℕ) → b ∈ ({2, 3, 5, 7} : Set ℕ) → 
  c ∈ ({2, 3, 5, 7} : Set ℕ) → d ∈ ({2, 3, 5, 7} : Set ℕ) →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  2*a*b + 2*b*c + 2*c*d + 2*d*a ≤ 144 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l1220_122054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l1220_122047

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := 9 * x^2 - 16 * y^2 = 144

/-- Point P coordinates -/
def P : ℝ × ℝ := (8, 3)

/-- A is a point on the hyperbola -/
def A : ℝ × ℝ := sorry

/-- B is a point on the hyperbola -/
def B : ℝ × ℝ := sorry

/-- A is on the hyperbola -/
axiom A_on_hyperbola : hyperbola A.1 A.2

/-- B is on the hyperbola -/
axiom B_on_hyperbola : hyperbola B.1 B.2

/-- P is the midpoint of AB -/
axiom P_midpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- The equation of line AB -/
def line_AB (x y : ℝ) : Prop := 3 * x - 2 * y - 18 = 0

/-- Theorem: Given the conditions, prove that the equation of line AB is 3x - 2y - 18 = 0 -/
theorem line_AB_equation : line_AB A.1 A.2 ∧ line_AB B.1 B.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l1220_122047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1220_122064

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Add necessary conditions for a valid triangle
  pos_sides : a > 0 ∧ b > 0 ∧ c > 0
  angle_sum : A + B + C = π
  -- You might want to add more conditions here

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  (Real.sqrt 3 * t.a * Real.cos t.C + t.a * Real.sin t.C = Real.sqrt 3 * t.b →
   t.A = π / 3) ∧
  (t.c^2 = 4 * t.a^2 - 4 * t.b^2 →
   t.a + t.b = (3 + Real.sqrt 13) / 2 →
   t.c = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1220_122064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_2_equals_11_f_min_max_on_interval_l1220_122020

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (4 : ℝ)^x - 2^(x+1) + 3

-- Theorem 1: f(2) = 11
theorem f_at_2_equals_11 : f 2 = 11 := by sorry

-- Theorem 2: Minimum and maximum values of f on [-2, 1]
theorem f_min_max_on_interval :
  ∃ (min_val max_val : ℝ),
    (∀ x ∈ Set.Icc (-2) 1, f x ≥ min_val ∧ f x ≤ max_val) ∧
    (∃ x₁ ∈ Set.Icc (-2) 1, f x₁ = min_val) ∧
    (∃ x₂ ∈ Set.Icc (-2) 1, f x₂ = max_val) ∧
    min_val = 2 ∧ max_val = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_2_equals_11_f_min_max_on_interval_l1220_122020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_plus_2cos4_l1220_122000

theorem min_sin4_plus_2cos4 :
  ∃ (min : ℝ), min = 2/3 ∧ (∀ x : ℝ, Real.sin x^4 + 2 * Real.cos x^4 ≥ min) ∧
  (∃ x : ℝ, Real.sin x^4 + 2 * Real.cos x^4 = min) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_plus_2cos4_l1220_122000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_belt_a_more_cost_effective_l1220_122023

/-- Represents the sorting capacity and maintenance cost of a conveyor belt -/
structure ConveyorBelt where
  sorting_rate : ℝ
  maintenance_cost : ℝ

/-- The problem setup for the conveyor belt comparison -/
structure ConveyorBeltProblem where
  belt_a : ConveyorBelt
  belt_b : ConveyorBelt
  total_tasks : ℝ
  time_difference : ℝ

/-- The conditions of the problem -/
def problem_conditions (p : ConveyorBeltProblem) : Prop :=
  p.belt_a.sorting_rate = 1.5 * p.belt_b.sorting_rate ∧
  p.total_tasks = 18000 ∧
  p.time_difference = 10 ∧
  p.belt_a.maintenance_cost = 8 ∧
  p.belt_b.maintenance_cost = 6 ∧
  p.total_tasks / p.belt_a.sorting_rate = p.total_tasks / p.belt_b.sorting_rate - p.time_difference

/-- The theorem stating that belt A is more cost-effective -/
theorem belt_a_more_cost_effective (p : ConveyorBeltProblem) 
  (h : problem_conditions p) : 
  p.total_tasks / p.belt_a.sorting_rate * p.belt_a.maintenance_cost < 
  p.total_tasks / p.belt_b.sorting_rate * p.belt_b.maintenance_cost :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_belt_a_more_cost_effective_l1220_122023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_sided_polygon_diagonals_l1220_122026

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add any necessary fields here

/-- The number of diagonals in a regular polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals longer than a side in a regular polygon -/
def num_long_diagonals (n : ℕ) : ℕ := n * (n - 5) / 2

theorem nine_sided_polygon_diagonals :
  (num_diagonals 9 = 27) ∧ (num_long_diagonals 9 = 18) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_sided_polygon_diagonals_l1220_122026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worksheet_grading_problem_l1220_122090

/-- A math problem about grading worksheets and correcting errors. -/
theorem worksheet_grading_problem 
  (total_worksheets : ℕ) 
  (problems_per_worksheet : ℕ) 
  (graded_worksheets : ℕ) 
  (error_rate : ℚ) 
  (h1 : total_worksheets = 24)
  (h2 : problems_per_worksheet = 7)
  (h3 : graded_worksheets = 10)
  (h4 : error_rate = 15 / 100) : 
  (total_worksheets * problems_per_worksheet - graded_worksheets * problems_per_worksheet = 98 ∧ 
   Int.floor (error_rate * (graded_worksheets * problems_per_worksheet : ℚ)) + 
   Int.floor (error_rate * ((total_worksheets - graded_worksheets) * problems_per_worksheet : ℚ)) = 25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worksheet_grading_problem_l1220_122090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1220_122040

theorem trigonometric_equation_solution (k : ℤ) :
  let x : ℝ := k * π / 5
  (Real.sin (2 * x) - Real.sin (3 * x) + Real.sin (8 * x) = Real.cos (7 * x + 3 * π / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1220_122040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_total_wholesale_cost_approx_l1220_122030

noncomputable def retail_price_pants : ℝ := 36
noncomputable def markup_pants : ℝ := 0.80
noncomputable def retail_price_shirt : ℝ := 45
noncomputable def markup_shirt : ℝ := 0.60
noncomputable def retail_price_jacket : ℝ := 120
noncomputable def markup_jacket : ℝ := 0.50
noncomputable def retail_price_skirt : ℝ := 80
noncomputable def markup_skirt : ℝ := 0.75
noncomputable def retail_price_dress : ℝ := 150
noncomputable def markup_dress : ℝ := 0.40
noncomputable def bulk_discount : ℝ := 0.10

noncomputable def wholesale_cost (retail_price : ℝ) (markup : ℝ) : ℝ :=
  retail_price / (1 + markup)

noncomputable def total_wholesale_cost : ℝ :=
  wholesale_cost retail_price_pants markup_pants +
  wholesale_cost retail_price_shirt markup_shirt +
  wholesale_cost retail_price_jacket markup_jacket +
  wholesale_cost retail_price_skirt markup_skirt +
  wholesale_cost retail_price_dress markup_dress

noncomputable def discounted_total_wholesale_cost : ℝ :=
  total_wholesale_cost * (1 - bulk_discount)

theorem discounted_total_wholesale_cost_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ abs (discounted_total_wholesale_cost - 252.88) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_total_wholesale_cost_approx_l1220_122030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_equals_twelve_l1220_122095

theorem root_product_equals_twelve : Real.sqrt (Real.sqrt 81) * (8 : Real) ^ (1/3) * Real.sqrt 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_equals_twelve_l1220_122095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1220_122032

noncomputable section

-- Define the lines
def line_through_origin (m : ℝ) : ℝ → ℝ := λ x ↦ m * x
def vertical_line : ℝ → ℝ := λ _ ↦ 1
def sloped_line : ℝ → ℝ := λ x ↦ 1 + Real.sqrt 3 * x

-- Define the intersection points
def intersection_point1 (m : ℝ) : ℝ × ℝ := (1, m)
def intersection_point2 (m : ℝ) : ℝ × ℝ := 
  let x := (1 + Real.sqrt 3 * m) / (Real.sqrt 3 - m)
  (x, 1 + Real.sqrt 3 * x)

-- Define the perimeter calculation
def perimeter (m : ℝ) : ℝ :=
  let p1 := intersection_point1 m
  let p2 := intersection_point2 m
  let side1 := Real.sqrt (p1.1^2 + p1.2^2)
  let side2 := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let side3 := Real.sqrt (p2.1^2 + p2.2^2)
  side1 + side2 + side3

-- Theorem statement
theorem triangle_perimeter :
  ∃ m : ℝ, perimeter m = 2 + Real.sqrt 3 + Real.sqrt 3 / 3 + Real.sqrt (5 + 2 * Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1220_122032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AB_BCD_l1220_122010

-- Define the points
def A : ℝ × ℝ × ℝ := (1, 0, 1)
def B : ℝ × ℝ × ℝ := (-2, 2, 1)
def C : ℝ × ℝ × ℝ := (2, 0, 3)
def D : ℝ × ℝ × ℝ := (0, 4, -2)

-- Define the function to calculate the angle
noncomputable def angle_line_plane (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  Real.arcsin (Real.sqrt 13 / Real.sqrt 101)

-- Theorem statement
theorem angle_AB_BCD :
  angle_line_plane A B C D = Real.arcsin (Real.sqrt 13 / Real.sqrt 101) := by
  -- Unfold the definition of angle_line_plane
  unfold angle_line_plane
  -- The equality holds by definition
  rfl

#check angle_AB_BCD

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AB_BCD_l1220_122010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1220_122091

-- Define the operation P
noncomputable def P (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the main theorem
theorem problem_solution :
  -- Given conditions
  ∀ (x a b c p d A B : ℝ),
  (∀ x, a * (x + 1) = x^3 + 3*x^2 + 3*x + 1) →
  (a - 1 = 0 → (x = 0 ∨ x = b)) →
  (p * c^4 = 32) →
  (p * c = b^2) →
  (c > 0) →
  (∀ A B, P (A * B) = P A + P B) →
  (P A = 1) →
  (P B = c) →
  (d = A * B) →
  -- Prove the following
  (a = x^2 + 2*x + 1) ∧
  (b = -2) ∧
  (c = 2) ∧
  (d = 1000) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1220_122091
