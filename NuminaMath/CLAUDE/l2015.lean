import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2015_201532

def M : Set ℝ := {x | x^2 - x - 2 = 0}
def N : Set ℝ := {-1, 0}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2015_201532


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l2015_201524

theorem rectangle_width_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.4 * L
  let W' := W * L / L'
  (W - W') / W = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l2015_201524


namespace NUMINAMATH_CALUDE_female_managers_count_l2015_201587

theorem female_managers_count (total_employees : ℕ) (female_employees : ℕ) (total_managers : ℕ) (male_associates : ℕ)
  (h1 : total_employees = 250)
  (h2 : female_employees = 90)
  (h3 : total_managers = 40)
  (h4 : male_associates = 160) :
  total_managers = total_employees - female_employees - male_associates :=
by
  sorry

end NUMINAMATH_CALUDE_female_managers_count_l2015_201587


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2015_201530

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  m < -6 ∨ m > 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2015_201530


namespace NUMINAMATH_CALUDE_stating_count_quadrilaterals_correct_l2015_201522

/-- 
For a convex n-gon, count_quadrilaterals n returns the number of ways to choose 
four vertices that form a quadrilateral with sides that are diagonals of the n-gon.
-/
def count_quadrilaterals (n : ℕ) : ℕ := 
  n / 4 * Nat.choose (n - 5) 3

/-- 
Theorem stating that count_quadrilaterals correctly counts the number of ways 
to choose four vertices forming a quadrilateral with diagonal sides in an n-gon.
-/
theorem count_quadrilaterals_correct (n : ℕ) : 
  count_quadrilaterals n = n / 4 * Nat.choose (n - 5) 3 := by
  sorry

#eval count_quadrilaterals 10  -- Example evaluation

end NUMINAMATH_CALUDE_stating_count_quadrilaterals_correct_l2015_201522


namespace NUMINAMATH_CALUDE_special_pentagon_area_l2015_201504

/-- Represents a pentagon with specific properties -/
structure SpecialPentagon where
  -- 10 line segments of 3 meters each
  segment_length : ℝ
  segment_count : ℕ
  -- FGH and FIJ are identical equilateral triangles
  equilateral_area : ℝ
  -- GHI is a right-angled triangle
  right_triangle_area : ℝ
  -- Conditions
  h_segment_length : segment_length = 3
  h_segment_count : segment_count = 10
  h_equilateral_area : equilateral_area = (9 * Real.sqrt 3) / 4
  h_right_triangle_area : right_triangle_area = 9 / 4

/-- The theorem to be proved -/
theorem special_pentagon_area (p : SpecialPentagon) :
  p.equilateral_area * 2 + p.right_triangle_area = Real.sqrt 81 + Real.sqrt 48 := by
  sorry

#check special_pentagon_area

end NUMINAMATH_CALUDE_special_pentagon_area_l2015_201504


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l2015_201520

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 10

-- Define the line y = -x + m
def line (x y m : ℝ) : Prop := y = -x + m

-- Define the points A and B
def point_A : ℝ × ℝ := (-1, 1)
def point_B : ℝ × ℝ := (1, 3)

-- Define the intersection points M and N
def intersection_points (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ line x₁ y₁ m ∧
    circle_C x₂ y₂ ∧ line x₂ y₂ m ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the condition for MN to pass through the origin
def passes_through_origin (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ line x₁ y₁ m ∧
    circle_C x₂ y₂ ∧ line x₂ y₂ m ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (x₁ + x₂)^2 + (y₁ + y₂)^2 = (x₁ - x₂)^2 + (y₁ - y₂)^2

theorem circle_intersection_theorem :
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 ∧
  (∀ m : ℝ, intersection_points m) →
  passes_through_origin (1 + Real.sqrt 7) ∧
  passes_through_origin (1 - Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l2015_201520


namespace NUMINAMATH_CALUDE_set_relationship_l2015_201574

def P : Set ℝ := {y | ∃ x, y = -x^2 + 1}
def Q : Set ℝ := {y | ∃ x, y = 2^x}

theorem set_relationship : ∀ y : ℝ, y > 1 → y ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_set_relationship_l2015_201574


namespace NUMINAMATH_CALUDE_equilateral_triangle_paths_l2015_201566

/-- Represents the number of paths in an equilateral triangle of side length n --/
def f (n : ℕ) : ℕ := n.factorial

/-- 
Theorem: The number of paths from the top triangle to the middle triangle 
in the bottom row of an equilateral triangle with side length n, 
where paths can only move downward and never revisit a triangle, is equal to n!.
-/
theorem equilateral_triangle_paths (n : ℕ) : f n = n.factorial := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_paths_l2015_201566


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_measure_l2015_201567

/-- In a right triangle where the ratio of the measures of the acute angles is 7:2,
    the measure of the smaller angle is 20°. -/
theorem right_triangle_acute_angle_measure (α β : ℝ) : 
  α > 0 ∧ β > 0 ∧  -- Angles are positive
  α + β + 90 = 180 ∧  -- Sum of angles in a triangle is 180°
  α / β = 7 / 2 ∧  -- Ratio of acute angles
  α > β  -- α is the larger acute angle
  → β = 20 := by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_measure_l2015_201567


namespace NUMINAMATH_CALUDE_same_grade_percentage_l2015_201586

theorem same_grade_percentage :
  let total_students : ℕ := 40
  let same_grade_A : ℕ := 3
  let same_grade_B : ℕ := 6
  let same_grade_C : ℕ := 4
  let same_grade_D : ℕ := 2
  let total_same_grade := same_grade_A + same_grade_B + same_grade_C + same_grade_D
  (total_same_grade : ℚ) / total_students * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l2015_201586


namespace NUMINAMATH_CALUDE_boat_current_speed_l2015_201519

/-- Proves that given a boat with a speed of 20 km/hr in still water, 
    traveling 9.2 km downstream in 24 minutes, the rate of the current is 3 km/hr. -/
theorem boat_current_speed 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : downstream_distance = 9.2)
  (h3 : time_minutes = 24) :
  let time_hours : ℝ := time_minutes / 60
  let current_speed : ℝ := downstream_distance / time_hours - boat_speed
  current_speed = 3 := by sorry

end NUMINAMATH_CALUDE_boat_current_speed_l2015_201519


namespace NUMINAMATH_CALUDE_lena_video_game_time_l2015_201585

/-- Proves that Lena played video games for 3.5 hours given the conditions of the problem -/
theorem lena_video_game_time (lena_time brother_time : ℕ) : 
  brother_time = lena_time + 17 →
  lena_time + brother_time = 437 →
  (lena_time : ℚ) / 60 = 3.5 := by
    sorry

end NUMINAMATH_CALUDE_lena_video_game_time_l2015_201585


namespace NUMINAMATH_CALUDE_prism_volume_given_tangent_sphere_l2015_201525

-- Define the sphere
structure Sphere where
  volume : ℝ

-- Define the right triangular prism
structure RightTriangularPrism where
  baseEdgeLength : ℝ
  height : ℝ

-- Define the property of the sphere being tangent to the prism
def isTangentTo (s : Sphere) (p : RightTriangularPrism) : Prop :=
  ∃ (r : ℝ), s.volume = (4/3) * Real.pi * r^3 ∧
             p.baseEdgeLength = 2 * Real.sqrt 3 * r ∧
             p.height = 2 * r

-- Theorem statement
theorem prism_volume_given_tangent_sphere (s : Sphere) (p : RightTriangularPrism) :
  s.volume = 9 * Real.pi / 2 →
  isTangentTo s p →
  (Real.sqrt 3 / 4) * p.baseEdgeLength^2 * p.height = 81 * Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_prism_volume_given_tangent_sphere_l2015_201525


namespace NUMINAMATH_CALUDE_perpendicular_line_parallel_planes_l2015_201580

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_parallel_planes 
  (l m n : Line) (α β : Plane) 
  (h1 : l ≠ m) (h2 : l ≠ n) (h3 : m ≠ n) (h4 : α ≠ β)
  (h5 : perpendicularToPlane l α) 
  (h6 : parallel α β) 
  (h7 : containedIn m β) : 
  perpendicular l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_parallel_planes_l2015_201580


namespace NUMINAMATH_CALUDE_triangle_theorem_l2015_201553

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle with geometric sequence sides -/
theorem triangle_theorem (t : Triangle) 
  (geom_seq : t.b^2 = t.a * t.c)
  (cos_B : Real.cos t.B = 3/5)
  (area : 1/2 * t.a * t.c * Real.sin t.B = 2) :
  (t.a + t.b + t.c = Real.sqrt 5 + Real.sqrt 21) ∧
  ((Real.sqrt 5 - 1)/2 < (Real.sin t.A + Real.cos t.A * Real.tan t.C) / 
                         (Real.sin t.B + Real.cos t.B * Real.tan t.C) ∧
   (Real.sin t.A + Real.cos t.A * Real.tan t.C) / 
   (Real.sin t.B + Real.cos t.B * Real.tan t.C) < (Real.sqrt 5 + 1)/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2015_201553


namespace NUMINAMATH_CALUDE_inclined_line_and_triangle_l2015_201529

/-- A line passing through a point with a given angle of inclination -/
structure InclinedLine where
  point : ℝ × ℝ
  angle : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about an inclined line and the triangle it forms with the axes -/
theorem inclined_line_and_triangle (l : InclinedLine) 
    (h1 : l.point = (0, -2))
    (h2 : l.angle = Real.pi / 3) : 
  ∃ (eq : LineEquation) (area : ℝ),
    eq.a = Real.sqrt 3 ∧ 
    eq.b = -1 ∧ 
    eq.c = -2 ∧
    area = (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inclined_line_and_triangle_l2015_201529


namespace NUMINAMATH_CALUDE_truck_speed_difference_l2015_201500

/-- Represents the speed difference between paved and dirt roads for a semi truck journey --/
theorem truck_speed_difference 
  (total_distance : ℝ) 
  (paved_time dirt_time : ℝ) 
  (dirt_speed : ℝ) :
  total_distance = 200 →
  paved_time = 2 →
  dirt_time = 3 →
  dirt_speed = 32 →
  (total_distance - dirt_speed * dirt_time) / paved_time - dirt_speed = 20 := by
  sorry

#check truck_speed_difference

end NUMINAMATH_CALUDE_truck_speed_difference_l2015_201500


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l2015_201516

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  (∃ n : ℕ, n < 1000 ∧ 5 ∣ n ∧ 6 ∣ n) →
  (∀ m : ℕ, m < 1000 ∧ 5 ∣ m ∧ 6 ∣ m → m ≤ 990) ∧
  990 < 1000 ∧ 5 ∣ 990 ∧ 6 ∣ 990 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l2015_201516


namespace NUMINAMATH_CALUDE_theoretical_yield_NaNO3_l2015_201565

/-- Theoretical yield of NaNO3 given initial conditions and overall yield -/
theorem theoretical_yield_NaNO3 (initial_NH4NO3 : ℝ) (initial_NaOH : ℝ) (percent_yield : ℝ) :
  initial_NH4NO3 = 2 →
  initial_NaOH = 2 →
  percent_yield = 0.85 →
  ∃ (theoretical_yield : ℝ),
    theoretical_yield = 289 ∧
    theoretical_yield = initial_NH4NO3 * 2 * 85 * percent_yield :=
by sorry

/-- Molar mass of NaNO3 in g/mol -/
def molar_mass_NaNO3 : ℝ := 85

/-- Theoretical yield in moles of NaNO3 -/
def theoretical_yield_moles (initial_NH4NO3 : ℝ) : ℝ := initial_NH4NO3 * 2

/-- Theoretical yield in grams of NaNO3 -/
def theoretical_yield_grams (theoretical_yield_moles : ℝ) : ℝ :=
  theoretical_yield_moles * molar_mass_NaNO3

/-- Actual yield in grams of NaNO3 considering percent yield -/
def actual_yield_grams (theoretical_yield_grams : ℝ) (percent_yield : ℝ) : ℝ :=
  theoretical_yield_grams * percent_yield

end NUMINAMATH_CALUDE_theoretical_yield_NaNO3_l2015_201565


namespace NUMINAMATH_CALUDE_parabola_properties_l2015_201509

/-- Represents a parabola with focus at distance 2 from directrix -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0
  h_focus_dist : p = 2

/-- Point on the parabola -/
structure ParabolaPoint (c : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * c.p * x

/-- Point Q satisfying PQ = 9QF -/
structure PointQ (c : Parabola) (p : ParabolaPoint c) where
  x : ℝ
  y : ℝ
  h_relation : (x - p.x)^2 + (y - p.y)^2 = 81 * ((x - c.p)^2 + y^2)

/-- The theorem to be proved -/
theorem parabola_properties (c : Parabola) :
  (∀ (x y : ℝ), y^2 = 2 * c.p * x ↔ y^2 = 4 * x) ∧
  (∀ (p : ParabolaPoint c) (q : PointQ c p),
    ∀ (slope : ℝ), slope = q.y / q.x → slope ≤ 1/3) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l2015_201509


namespace NUMINAMATH_CALUDE_perpendicular_line_parallel_lines_l2015_201536

-- Define the original lines
def line1 (x y : ℝ) : Prop := x + 3 * y - 5 = 0
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 0)

-- Define the distance between parallel lines
def distance : ℝ := 7

-- Theorem for the perpendicular line
theorem perpendicular_line :
  ∃ (a b c : ℝ), (∀ x y, a * x + b * y + c = 0 ↔ 3 * x - y + 3 = 0) ∧
  (∀ x y, line1 x y → (a * x + b * y + c = 0 → a * 3 + b = 0)) ∧
  (a * point_P.1 + b * point_P.2 + c = 0) :=
sorry

-- Theorem for the parallel lines
theorem parallel_lines :
  ∃ (c1 c2 : ℝ), 
  (∀ x y, 3 * x + 4 * y + c1 = 0 ∨ 3 * x + 4 * y + c2 = 0 ↔ 
    (3 * x + 4 * y + 23 = 0 ∨ 3 * x + 4 * y - 47 = 0)) ∧
  (∀ x y, line2 x y → 
    (|c1 + 12| / Real.sqrt 25 = distance ∧ |c2 + 12| / Real.sqrt 25 = distance)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_parallel_lines_l2015_201536


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2015_201555

open Real
open InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem necessary_but_not_sufficient (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b : E, a - 2 • b = 0 → ‖a - b‖ = ‖b‖) ∧
  (∃ a b : E, ‖a - b‖ = ‖b‖ ∧ a - 2 • b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2015_201555


namespace NUMINAMATH_CALUDE_number_subtraction_problem_l2015_201523

theorem number_subtraction_problem (x y : ℝ) 
  (h1 : (x - 5) / 7 = 7)
  (h2 : (x - y) / 10 = 5) : 
  y = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_problem_l2015_201523


namespace NUMINAMATH_CALUDE_vote_change_theorem_l2015_201564

/-- Represents the voting results of an assembly --/
structure VotingResults where
  total_members : ℕ
  initial_for : ℕ
  initial_against : ℕ
  revote_for : ℕ
  revote_against : ℕ

/-- Theorem about the change in votes for a resolution --/
theorem vote_change_theorem (v : VotingResults) : 
  v.total_members = 500 →
  v.initial_for + v.initial_against = v.total_members →
  v.revote_for + v.revote_against = v.total_members →
  v.initial_against > v.initial_for →
  v.revote_for > v.revote_against →
  (v.revote_for - v.revote_against) = 3 * (v.initial_against - v.initial_for) →
  v.revote_for = (7 * v.initial_against) / 6 →
  v.revote_for - v.initial_for = 90 := by
  sorry


end NUMINAMATH_CALUDE_vote_change_theorem_l2015_201564


namespace NUMINAMATH_CALUDE_ellipse_param_sum_l2015_201562

/-- An ellipse with given properties -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  constant_sum : ℝ
  tangent_slope : ℝ

/-- The standard form parameters of an ellipse -/
structure EllipseParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Theorem stating the sum of h, k, a, and b for the given ellipse -/
theorem ellipse_param_sum (e : Ellipse) (p : EllipseParams) : 
  e.F₁ = (-1, 1) → 
  e.F₂ = (5, 1) → 
  e.constant_sum = 10 → 
  e.tangent_slope = 1 → 
  p.h + p.k + p.a + p.b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_param_sum_l2015_201562


namespace NUMINAMATH_CALUDE_multiply_31_15_by_4_l2015_201550

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define multiplication of an angle by a natural number
def multiplyAngle (a : Angle) (n : ℕ) : Angle :=
  let totalMinutes := (a.degrees * 60 + a.minutes) * n
  { degrees := totalMinutes / 60,
    minutes := totalMinutes % 60 }

-- Theorem statement
theorem multiply_31_15_by_4 :
  let initial_angle : Angle := { degrees := 31, minutes := 15 }
  let result := multiplyAngle initial_angle 4
  result.degrees = 125 ∧ result.minutes = 0 :=
by sorry

end NUMINAMATH_CALUDE_multiply_31_15_by_4_l2015_201550


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2015_201583

theorem expand_and_simplify (x : ℝ) : 3 * (x - 3) * (x + 10) + 2 * x = 3 * x^2 + 23 * x - 90 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2015_201583


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2015_201545

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℚ) :
  is_geometric_sequence a →
  a 2 * a 5 = -3/4 →
  a 2 + a 3 + a 4 + a 5 = 5/4 →
  1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 = -5/3 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2015_201545


namespace NUMINAMATH_CALUDE_seven_mult_five_equals_34_l2015_201569

/-- Custom multiplication operation -/
def custom_mult (A B : ℝ) : ℝ := (A + 2*B) * (A - B)

/-- Theorem stating that 7 * 5 = 34 under the custom multiplication -/
theorem seven_mult_five_equals_34 : custom_mult 7 5 = 34 := by
  sorry

end NUMINAMATH_CALUDE_seven_mult_five_equals_34_l2015_201569


namespace NUMINAMATH_CALUDE_intersection_and_coefficients_l2015_201597

def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def B : Set ℝ := {x | -3 < x ∧ x < 1}

theorem intersection_and_coefficients :
  (A ∩ B = {x | -1 < x ∧ x < 1}) ∧
  (∃ a b : ℝ, (∀ x : ℝ, x ∈ B ↔ 2*x^2 + a*x + b < 0) ∧ a = 3 ∧ b = 4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_coefficients_l2015_201597


namespace NUMINAMATH_CALUDE_pies_from_36_apples_l2015_201515

/-- Given that 3 pies can be made from 12 apples, this function calculates
    the number of pies that can be made from a given number of apples. -/
def pies_from_apples (apples : ℕ) : ℕ :=
  (apples * 3) / 12

theorem pies_from_36_apples :
  pies_from_apples 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pies_from_36_apples_l2015_201515


namespace NUMINAMATH_CALUDE_log_equation_implies_sum_l2015_201578

theorem log_equation_implies_sum (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^4 + (Real.log y / Real.log 3)^4 + 8 = 
       8 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^Real.sqrt 2 + y^Real.sqrt 2 = 13 := by
sorry

end NUMINAMATH_CALUDE_log_equation_implies_sum_l2015_201578


namespace NUMINAMATH_CALUDE_hundredth_bracket_numbers_l2015_201531

def bracket_sequence (n : ℕ) : ℕ := 
  if n % 4 = 1 then 1
  else if n % 4 = 2 then 2
  else if n % 4 = 3 then 3
  else 1

def first_number_in_group (group : ℕ) : ℕ := 2 * group - 1

theorem hundredth_bracket_numbers :
  let group := (100 - 1) / 3 + 1
  let first_num := first_number_in_group group - 2
  bracket_sequence 100 = 2 ∧ first_num = 65 ∧ first_num + 2 = 67 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_bracket_numbers_l2015_201531


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l2015_201556

/-- Given Jessie's initial and current weights, calculate her weight loss. -/
theorem jessie_weight_loss (initial_weight current_weight : ℝ) 
  (h1 : initial_weight = 74)
  (h2 : current_weight = 67) :
  initial_weight - current_weight = 7 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l2015_201556


namespace NUMINAMATH_CALUDE_sarah_trip_distance_l2015_201573

/-- Represents Sarah's trip to the airport -/
structure AirportTrip where
  initial_speed : ℝ
  initial_time : ℝ
  final_speed : ℝ
  early_arrival : ℝ
  total_distance : ℝ

/-- The theorem stating the total distance of Sarah's trip -/
theorem sarah_trip_distance (trip : AirportTrip) : 
  trip.initial_speed = 15 ∧ 
  trip.initial_time = 1 ∧ 
  trip.final_speed = 60 ∧ 
  trip.early_arrival = 0.5 →
  trip.total_distance = 45 := by
  sorry

#check sarah_trip_distance

end NUMINAMATH_CALUDE_sarah_trip_distance_l2015_201573


namespace NUMINAMATH_CALUDE_apartment_buildings_count_l2015_201581

/-- The number of floors in each apartment building -/
def floors_per_building : ℕ := 12

/-- The number of apartments on each floor -/
def apartments_per_floor : ℕ := 6

/-- The number of doors needed for each apartment -/
def doors_per_apartment : ℕ := 7

/-- The total number of doors needed to be bought -/
def total_doors : ℕ := 1008

/-- The number of apartment buildings being constructed -/
def num_buildings : ℕ := total_doors / (floors_per_building * apartments_per_floor * doors_per_apartment)

theorem apartment_buildings_count : num_buildings = 2 := by
  sorry

end NUMINAMATH_CALUDE_apartment_buildings_count_l2015_201581


namespace NUMINAMATH_CALUDE_marble_difference_l2015_201594

theorem marble_difference (total : ℕ) (bag_a : ℕ) (bag_b : ℕ) : 
  total = 72 → bag_a = 42 → bag_b = total - bag_a → bag_a - bag_b = 12 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l2015_201594


namespace NUMINAMATH_CALUDE_jelly_bean_ratio_l2015_201588

theorem jelly_bean_ratio : 
  let napoleon_beans : ℕ := 17
  let sedrich_beans : ℕ := napoleon_beans + 4
  let mikey_beans : ℕ := 19
  let total_beans : ℕ := napoleon_beans + sedrich_beans
  2 * total_beans = 4 * mikey_beans := by sorry

end NUMINAMATH_CALUDE_jelly_bean_ratio_l2015_201588


namespace NUMINAMATH_CALUDE_smallest_n_value_l2015_201544

-- Define the cost of purple candy
def purple_cost : ℕ := 20

-- Define the quantities of other candies
def red_quantity : ℕ := 12
def green_quantity : ℕ := 14
def blue_quantity : ℕ := 15

-- Define the theorem
theorem smallest_n_value :
  ∃ (n : ℕ), n > 0 ∧ 
  (purple_cost * n) % red_quantity = 0 ∧
  (purple_cost * n) % green_quantity = 0 ∧
  (purple_cost * n) % blue_quantity = 0 ∧
  (∀ (m : ℕ), m > 0 → 
    (purple_cost * m) % red_quantity = 0 →
    (purple_cost * m) % green_quantity = 0 →
    (purple_cost * m) % blue_quantity = 0 →
    m ≥ n) ∧
  n = 21 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l2015_201544


namespace NUMINAMATH_CALUDE_arthur_hamburgers_l2015_201535

/-- The number of hamburgers Arthur bought on the first day -/
def hamburgers_day1 : ℕ := 1

/-- The price of a hamburger in dollars -/
def hamburger_price : ℚ := 6

/-- The price of a hot dog in dollars -/
def hotdog_price : ℚ := 1

/-- Total cost of Arthur's purchase on day 1 in dollars -/
def total_cost_day1 : ℚ := 10

/-- Total cost of Arthur's purchase on day 2 in dollars -/
def total_cost_day2 : ℚ := 7

/-- Number of hot dogs bought on day 1 -/
def hotdogs_day1 : ℕ := 4

/-- Number of hamburgers bought on day 2 -/
def hamburgers_day2 : ℕ := 2

/-- Number of hot dogs bought on day 2 -/
def hotdogs_day2 : ℕ := 3

theorem arthur_hamburgers :
  (hamburgers_day1 : ℚ) * hamburger_price + (hotdogs_day1 : ℚ) * hotdog_price = total_cost_day1 ∧
  (hamburgers_day2 : ℚ) * hamburger_price + (hotdogs_day2 : ℚ) * hotdog_price = total_cost_day2 :=
sorry

end NUMINAMATH_CALUDE_arthur_hamburgers_l2015_201535


namespace NUMINAMATH_CALUDE_deposit_withdrawal_ratio_l2015_201593

/-- Prove that the ratio of the deposited amount to the withdrawn amount is 2:1 --/
theorem deposit_withdrawal_ratio (initial_savings withdrawal final_balance : ℚ) 
  (h1 : initial_savings = 230)
  (h2 : withdrawal = 60)
  (h3 : final_balance = 290) : 
  (final_balance - (initial_savings - withdrawal)) / withdrawal = 2 := by
  sorry

end NUMINAMATH_CALUDE_deposit_withdrawal_ratio_l2015_201593


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l2015_201563

theorem polynomial_root_sum (a b c d e : ℤ) : 
  let g : ℝ → ℝ := λ x => x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e
  (∀ r : ℝ, g r = 0 → ∃ k : ℤ, r = -k ∧ k > 0) →
  a + b + c + d + e = 3403 →
  e = 9240 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l2015_201563


namespace NUMINAMATH_CALUDE_cheese_cookie_price_l2015_201533

/-- Proves that the price of a pack of cheese cookies is $1 -/
theorem cheese_cookie_price (boxes_per_carton : ℕ) (packs_per_box : ℕ) (cost_dozen_cartons : ℕ) 
  (h1 : boxes_per_carton = 12)
  (h2 : packs_per_box = 10)
  (h3 : cost_dozen_cartons = 1440) :
  (cost_dozen_cartons : ℚ) / ((12 * boxes_per_carton * packs_per_box) : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cheese_cookie_price_l2015_201533


namespace NUMINAMATH_CALUDE_min_fraction_sum_l2015_201512

theorem min_fraction_sum (A B C D : ℕ) : 
  A ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  B ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  C ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  D ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  B ≠ 0 → D ≠ 0 →
  (A : ℚ) / B + (C : ℚ) / D ≥ 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l2015_201512


namespace NUMINAMATH_CALUDE_herd_size_l2015_201538

/-- Given a herd of cows divided among four sons, prove that the total number of cows is 224 --/
theorem herd_size (herd : ℕ) : herd = 224 :=
  by
  have h1 : (3 : ℚ) / 7 + 1 / 3 + 1 / 6 + (herd - 16 : ℚ) / herd = 1 := by sorry
  have h2 : (herd - 16 : ℚ) / herd = 1 - (3 / 7 + 1 / 3 + 1 / 6) := by sorry
  have h3 : (herd - 16 : ℚ) / herd = 1 / 14 := by sorry
  have h4 : (16 : ℚ) / herd = 1 / 14 := by sorry
  sorry

end NUMINAMATH_CALUDE_herd_size_l2015_201538


namespace NUMINAMATH_CALUDE_sin_675_degrees_l2015_201568

theorem sin_675_degrees :
  Real.sin (675 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_675_degrees_l2015_201568


namespace NUMINAMATH_CALUDE_triangle_angle_measurement_l2015_201510

theorem triangle_angle_measurement (A B C : ℝ) : 
  -- Triangle ABC exists
  A + B + C = 180 →
  -- Measure of angle C is three times the measure of angle B
  C = 3 * B →
  -- Angle B measures 30°
  B = 30 →
  -- Then the measure of angle A is 60°
  A = 60 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measurement_l2015_201510


namespace NUMINAMATH_CALUDE_num_cows_bought_l2015_201528

/-- The number of sheep Zara bought -/
def num_sheep : ℕ := 7

/-- The number of goats Zara bought -/
def num_goats : ℕ := 113

/-- The number of groups for transporting animals -/
def num_groups : ℕ := 3

/-- The number of animals per group -/
def animals_per_group : ℕ := 48

/-- The total number of animals Zara bought -/
def total_animals : ℕ := num_groups * animals_per_group

theorem num_cows_bought : 
  total_animals - (num_sheep + num_goats) = 24 :=
by sorry

end NUMINAMATH_CALUDE_num_cows_bought_l2015_201528


namespace NUMINAMATH_CALUDE_function_inequality_l2015_201513

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 1

-- State the theorem
theorem function_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x + 1| < b → |f x + 4| < a) ↔ b ≤ a / 4 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l2015_201513


namespace NUMINAMATH_CALUDE_qin_jiushao_v3_value_main_theorem_l2015_201501

def f (x : ℝ) : ℝ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def qin_jiushao_v3 (a b c d : ℝ) (x : ℝ) : ℝ :=
  ((a * x + b) * x + c) * x + d

theorem qin_jiushao_v3_value :
  qin_jiushao_v3 2 5 6 23 (-4) = -49 :=
by sorry

-- The main theorem
theorem main_theorem :
  ∃ (v3 : ℝ), qin_jiushao_v3 2 5 6 23 (-4) = v3 ∧ v3 = -49 :=
by sorry

end NUMINAMATH_CALUDE_qin_jiushao_v3_value_main_theorem_l2015_201501


namespace NUMINAMATH_CALUDE_simplify_expression_l2015_201502

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2015_201502


namespace NUMINAMATH_CALUDE_sam_distance_l2015_201582

/-- Proves that Sam drove 160 miles given the conditions of the problem -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ)
  (h1 : marguerite_distance = 120)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) :
  (marguerite_distance / marguerite_time) * sam_time = 160 :=
by sorry

end NUMINAMATH_CALUDE_sam_distance_l2015_201582


namespace NUMINAMATH_CALUDE_heartsuit_properties_l2015_201527

def heartsuit (x y : ℝ) : ℝ := |x - y| + 1

theorem heartsuit_properties :
  (∀ x y : ℝ, heartsuit x y = heartsuit y x) ∧
  (∃ x y : ℝ, 2 * (heartsuit x y) ≠ heartsuit (2*x) (2*y)) ∧
  (∃ x : ℝ, heartsuit x 0 ≠ x + 1) ∧
  (∀ x : ℝ, heartsuit x x = 1) ∧
  (∀ x y : ℝ, x ≠ y → heartsuit x y > 1) :=
by sorry

end NUMINAMATH_CALUDE_heartsuit_properties_l2015_201527


namespace NUMINAMATH_CALUDE_tangency_and_tangent_line_l2015_201591

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x = Real.sqrt (2 * y^2 + 25/2)
def C₂ (a x y : ℝ) : Prop := y = a * x^2

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x y : ℝ, C₁ x y ∧ C₂ a x y ∧
  (∀ x' y' : ℝ, C₁ x' y' → C₂ a x' y' → (x = x' ∧ y = y'))

-- State the theorem
theorem tangency_and_tangent_line :
  ∃ a : ℝ, a > 0 ∧ is_tangent a ∧
  (∀ x y : ℝ, C₁ x y ∧ C₂ a x y → x = 5 ∧ y = 5/2) ∧
  (∀ x y : ℝ, 2*x - 2*y - 5 = 0 ↔ (C₁ x y ∧ C₂ a x y ∨ (x = 5 ∧ y = 5/2))) :=
sorry

end NUMINAMATH_CALUDE_tangency_and_tangent_line_l2015_201591


namespace NUMINAMATH_CALUDE_french_fries_cooking_time_l2015_201557

/-- Calculates the remaining cooking time in seconds -/
def remaining_cooking_time (recommended_minutes : ℕ) (actual_seconds : ℕ) : ℕ :=
  recommended_minutes * 60 - actual_seconds

/-- Theorem: Given the recommended cooking time of 5 minutes and an actual cooking time of 45 seconds, the remaining cooking time is 255 seconds -/
theorem french_fries_cooking_time : remaining_cooking_time 5 45 = 255 := by
  sorry

end NUMINAMATH_CALUDE_french_fries_cooking_time_l2015_201557


namespace NUMINAMATH_CALUDE_range_of_a_for_unique_integer_solution_l2015_201506

/-- Given a system of inequalities, prove the range of a for which there is exactly one integer solution. -/
theorem range_of_a_for_unique_integer_solution (a : ℝ) : 
  (∃! (x : ℤ), (x^3 + 3*x^2 - x - 3 > 0) ∧ 
                (x^2 - 2*a*x - 1 ≤ 0) ∧ 
                (a > 0)) ↔ 
  (3/4 ≤ a ∧ a < 4/3) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_for_unique_integer_solution_l2015_201506


namespace NUMINAMATH_CALUDE_problem_solution_l2015_201551

theorem problem_solution (x y : ℝ) (h1 : y > 2*x) (h2 : x > 0) (h3 : x/y + y/x = 8) :
  (x + y) / (x - y) = -Real.sqrt (5/3) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2015_201551


namespace NUMINAMATH_CALUDE_root_equation_implies_sum_l2015_201503

theorem root_equation_implies_sum (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → m^2 - 2*m + 2023 = 2026 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_sum_l2015_201503


namespace NUMINAMATH_CALUDE_inequality_proof_l2015_201589

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x^4 + y^2)) + (y / (x^2 + y^4)) ≤ 1 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2015_201589


namespace NUMINAMATH_CALUDE_franklin_valentines_l2015_201558

/-- The number of Valentines Mrs. Franklin gave to her students -/
def valentines_given (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that Mrs. Franklin gave 42 Valentines to her students -/
theorem franklin_valentines : valentines_given 58 16 = 42 := by
  sorry

end NUMINAMATH_CALUDE_franklin_valentines_l2015_201558


namespace NUMINAMATH_CALUDE_chemical_solution_replacement_exists_l2015_201505

theorem chemical_solution_replacement_exists : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ 
  (1 - x)^5 * 0.5 + x * (0.6 + 0.65 + 0.7 + 0.75 + 0.8) = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_chemical_solution_replacement_exists_l2015_201505


namespace NUMINAMATH_CALUDE_max_cake_pieces_l2015_201592

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 15

/-- The size of the small cake piece in inches -/
def small_piece_size : ℕ := 5

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := (large_cake_size * large_cake_size) / (small_piece_size * small_piece_size)

theorem max_cake_pieces : max_pieces = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_cake_pieces_l2015_201592


namespace NUMINAMATH_CALUDE_power_of_power_l2015_201552

theorem power_of_power : (3^2)^4 = 6561 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l2015_201552


namespace NUMINAMATH_CALUDE_jungkook_balls_count_l2015_201549

/-- The number of boxes Jungkook has -/
def num_boxes : ℕ := 3

/-- The number of balls in each box -/
def balls_per_box : ℕ := 2

/-- The total number of balls Jungkook has -/
def total_balls : ℕ := num_boxes * balls_per_box

theorem jungkook_balls_count : total_balls = 6 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_balls_count_l2015_201549


namespace NUMINAMATH_CALUDE_quinary_444_equals_octal_174_l2015_201559

/-- Converts a quinary (base 5) number to decimal (base 10) -/
def quinary_to_decimal (q : ℕ) : ℕ :=
  (q / 100) * 5^2 + ((q / 10) % 10) * 5^1 + (q % 10) * 5^0

/-- Converts a decimal (base 10) number to octal (base 8) -/
def decimal_to_octal (d : ℕ) : ℕ :=
  (d / 64) * 100 + ((d / 8) % 8) * 10 + (d % 8)

/-- Theorem stating that 444₅ is equal to 174₈ -/
theorem quinary_444_equals_octal_174 :
  decimal_to_octal (quinary_to_decimal 444) = 174 := by
  sorry

end NUMINAMATH_CALUDE_quinary_444_equals_octal_174_l2015_201559


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l2015_201584

theorem coefficient_x_squared (p q : Polynomial ℤ) (hp : p = 3 * X^2 + 4 * X + 5) (hq : q = 6 * X^2 + 7 * X + 8) :
  (p * q).coeff 2 = 82 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_l2015_201584


namespace NUMINAMATH_CALUDE_probability_in_specific_club_l2015_201514

/-- A club with members of different genders and seniority levels -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ
  senior_boys : ℕ
  junior_boys : ℕ
  senior_girls : ℕ
  junior_girls : ℕ

/-- The probability of selecting two girls, one senior and one junior, from the club -/
def probability_two_girls_diff_seniority (c : Club) : ℚ :=
  (c.senior_girls.choose 1 * c.junior_girls.choose 1 : ℚ) / c.total_members.choose 2

/-- Theorem stating the probability for the given club configuration -/
theorem probability_in_specific_club : 
  ∃ c : Club, 
    c.total_members = 12 ∧ 
    c.boys = 6 ∧ 
    c.girls = 6 ∧ 
    c.senior_boys = 3 ∧ 
    c.junior_boys = 3 ∧ 
    c.senior_girls = 3 ∧ 
    c.junior_girls = 3 ∧ 
    probability_two_girls_diff_seniority c = 9 / 66 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_specific_club_l2015_201514


namespace NUMINAMATH_CALUDE_sphere_surface_volume_relation_l2015_201548

theorem sphere_surface_volume_relation :
  ∀ (r R : ℝ),
  r > 0 →
  R > 0 →
  (4 * Real.pi * R^2) = (4 * (4 * Real.pi * r^2)) →
  ((4/3) * Real.pi * R^3) = (8 * ((4/3) * Real.pi * r^3)) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_volume_relation_l2015_201548


namespace NUMINAMATH_CALUDE_elder_sister_age_when_sum_was_twenty_l2015_201518

/-- 
Given:
- The younger sister is currently 18 years old
- The elder sister is currently 26 years old
- At some point in the past, the sum of their ages was 20 years

Prove that when the sum of their ages was 20 years, the elder sister was 14 years old.
-/
theorem elder_sister_age_when_sum_was_twenty 
  (younger_current : ℕ) 
  (elder_current : ℕ) 
  (years_ago : ℕ) 
  (h1 : younger_current = 18) 
  (h2 : elder_current = 26) 
  (h3 : younger_current - years_ago + elder_current - years_ago = 20) : 
  elder_current - years_ago = 14 :=
sorry

end NUMINAMATH_CALUDE_elder_sister_age_when_sum_was_twenty_l2015_201518


namespace NUMINAMATH_CALUDE_fair_die_probability_l2015_201575

/-- Probability of rolling at least a four on a fair die -/
def p : ℚ := 1/2

/-- Number of rolls -/
def n : ℕ := 8

/-- Minimum number of successful rolls required -/
def k : ℕ := 6

/-- The probability of rolling at least a four, at least six times in eight rolls of a fair die -/
theorem fair_die_probability : (Finset.range (n + 1 - k)).sum (λ i => n.choose (n - i) * p^(n - i) * (1 - p)^i) = 129/256 := by
  sorry

end NUMINAMATH_CALUDE_fair_die_probability_l2015_201575


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2015_201511

theorem right_triangle_hypotenuse (base height hypotenuse : ℝ) :
  base = 3 →
  height * base / 2 = 6 →
  base^2 + height^2 = hypotenuse^2 →
  hypotenuse = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2015_201511


namespace NUMINAMATH_CALUDE_club_membership_theorem_l2015_201537

/-- Represents the number of students in various club combinations -/
structure ClubMembership where
  total : ℕ
  music : ℕ
  science : ℕ
  sports : ℕ
  none : ℕ
  onlyMusic : ℕ
  onlyScience : ℕ
  onlySports : ℕ
  musicScience : ℕ
  scienceSports : ℕ
  musicSports : ℕ
  allThree : ℕ

/-- Theorem stating that given the conditions, the number of students in all three clubs is 1 -/
theorem club_membership_theorem (c : ClubMembership) : 
  c.total = 40 ∧ 
  c.music = c.total / 4 ∧ 
  c.science = c.total / 5 ∧ 
  c.sports = 8 ∧ 
  c.none = 7 ∧ 
  c.onlyMusic = 6 ∧ 
  c.onlyScience = 5 ∧ 
  c.onlySports = 2 ∧ 
  c.music = c.onlyMusic + c.musicScience + c.musicSports + c.allThree ∧ 
  c.science = c.onlyScience + c.musicScience + c.scienceSports + c.allThree ∧ 
  c.sports = c.onlySports + c.scienceSports + c.musicSports + c.allThree ∧ 
  c.total = c.none + c.onlyMusic + c.onlyScience + c.onlySports + c.musicScience + c.scienceSports + c.musicSports + c.allThree →
  c.allThree = 1 := by
sorry


end NUMINAMATH_CALUDE_club_membership_theorem_l2015_201537


namespace NUMINAMATH_CALUDE_mode_of_scores_l2015_201570

def scores : List ℕ := [35, 37, 39, 37, 38, 38, 37]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_scores :
  mode scores = 37 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_scores_l2015_201570


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l2015_201507

/-- The volume of a cone formed by rolling up a half-sector of a circle --/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) : 
  let base_radius : ℝ := r / 2
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  (1/3 : ℝ) * Real.pi * base_radius^2 * cone_height = 9 * Real.pi * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l2015_201507


namespace NUMINAMATH_CALUDE_distinct_terms_in_expansion_l2015_201577

/-- The number of distinct terms in the expansion of (a+b+c+d)(e+f+g+h+i),
    given that terms involving the product of a and e, and b and f are identical
    and combine into a single term. -/
theorem distinct_terms_in_expansion : ℕ := by
  sorry

end NUMINAMATH_CALUDE_distinct_terms_in_expansion_l2015_201577


namespace NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l2015_201539

-- Define the property of lines being skew
def are_skew_lines (a b : Line3D) : Prop := sorry

-- Define the property of lines having no common points
def have_no_common_points (a b : Line3D) : Prop := sorry

-- Theorem stating that "are_skew_lines" is a sufficient but not necessary condition for "have_no_common_points"
theorem skew_lines_sufficient_not_necessary (a b : Line3D) :
  (are_skew_lines a b → have_no_common_points a b) ∧
  ¬(have_no_common_points a b → are_skew_lines a b) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l2015_201539


namespace NUMINAMATH_CALUDE_quadrilateral_centers_collinearity_l2015_201572

-- Define the points
variable (A B C D E U H V K : Euclidean_plane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Euclidean_plane) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect_at (A B C D E : Euclidean_plane) : Prop := sorry

-- Define the circumcenter
def is_circumcenter (U A B E : Euclidean_plane) : Prop := sorry

-- Define the orthocenter
def is_orthocenter (H A B E : Euclidean_plane) : Prop := sorry

-- Define collinearity
def collinear (P Q R : Euclidean_plane) : Prop := sorry

-- State the theorem
theorem quadrilateral_centers_collinearity 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : diagonals_intersect_at A B C D E)
  (h3 : is_circumcenter U A B E)
  (h4 : is_orthocenter H A B E)
  (h5 : is_circumcenter V C D E)
  (h6 : is_orthocenter K C D E) :
  collinear U E K ↔ collinear V E H := by sorry

end NUMINAMATH_CALUDE_quadrilateral_centers_collinearity_l2015_201572


namespace NUMINAMATH_CALUDE_marbles_exceed_500_on_day_5_l2015_201521

def marble_sequence (n : ℕ) : ℕ := 4^n

theorem marbles_exceed_500_on_day_5 :
  ∀ k : ℕ, k < 5 → marble_sequence k ≤ 500 ∧ marble_sequence 5 > 500 :=
by sorry

end NUMINAMATH_CALUDE_marbles_exceed_500_on_day_5_l2015_201521


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l2015_201571

theorem triangle_third_side_length 
  (a b : ℝ) 
  (angle : ℝ) 
  (ha : a = 9) 
  (hb : b = 10) 
  (hangle : angle = Real.pi * 3 / 4) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2 * a * b * Real.cos angle ∧ 
            c = Real.sqrt (181 + 90 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l2015_201571


namespace NUMINAMATH_CALUDE_surface_area_circumscribed_sphere_l2015_201517

/-- The surface area of a sphere circumscribing a cube with edge length 1 is 3π. -/
theorem surface_area_circumscribed_sphere (cube_edge : Real) (sphere_radius : Real) :
  cube_edge = 1 →
  sphere_radius = (Real.sqrt 3) / 2 →
  4 * Real.pi * sphere_radius ^ 2 = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_surface_area_circumscribed_sphere_l2015_201517


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l2015_201543

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l2015_201543


namespace NUMINAMATH_CALUDE_stream_speed_l2015_201598

/-- Given a canoe's upstream and downstream speeds, prove the speed of the stream -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 3)
  (h_downstream : downstream_speed = 12) :
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2015_201598


namespace NUMINAMATH_CALUDE_opposite_of_neg_abs_l2015_201541

theorem opposite_of_neg_abs : -(-(|5 - 6|)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_abs_l2015_201541


namespace NUMINAMATH_CALUDE_angle_CAD_measure_l2015_201579

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the properties of the triangle and square
def is_right_triangle (A B C : ℝ × ℝ) : Prop := sorry
def is_isosceles (A B C : ℝ × ℝ) : Prop := sorry
def is_square (B C D E : ℝ × ℝ) : Prop := sorry

-- Define angle measurement function
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_CAD_measure 
  (h_right : is_right_triangle A B C)
  (h_isosceles : is_isosceles A B C)
  (h_square : is_square B C D E) :
  angle_measure C A D = 22.5 := by sorry

end NUMINAMATH_CALUDE_angle_CAD_measure_l2015_201579


namespace NUMINAMATH_CALUDE_matrix_cube_sum_l2015_201561

/-- Definition of the matrix N -/
def N (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![a, c, b],
    ![c, b, a],
    ![b, a, c]]

/-- The theorem statement -/
theorem matrix_cube_sum (a b c : ℂ) :
  (N a b c)^2 = 1 → a * b * c = 1 → a^3 + b^3 + c^3 = 2 ∨ a^3 + b^3 + c^3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_cube_sum_l2015_201561


namespace NUMINAMATH_CALUDE_ball_count_proof_l2015_201590

/-- Proves that given 9 yellow balls in a box and a 30% probability of drawing a yellow ball,
    the total number of balls in the box is 30. -/
theorem ball_count_proof (yellow_balls : ℕ) (probability : ℚ) (total_balls : ℕ) : 
  yellow_balls = 9 → probability = 3/10 → (yellow_balls : ℚ) / total_balls = probability → total_balls = 30 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_proof_l2015_201590


namespace NUMINAMATH_CALUDE_women_to_total_ratio_in_salem_l2015_201595

/-- The population of Leesburg -/
def leesburg_population : ℕ := 58940

/-- The original population of Salem before people moved out -/
def salem_original_population : ℕ := 15 * leesburg_population

/-- The number of people who moved out of Salem -/
def people_moved_out : ℕ := 130000

/-- The new population of Salem after people moved out -/
def salem_new_population : ℕ := salem_original_population - people_moved_out

/-- The number of women living in Salem after the population change -/
def women_in_salem : ℕ := 377050

/-- The theorem stating the ratio of women to the total population in Salem -/
theorem women_to_total_ratio_in_salem :
  (women_in_salem : ℚ) / salem_new_population = 377050 / 754100 := by sorry

end NUMINAMATH_CALUDE_women_to_total_ratio_in_salem_l2015_201595


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l2015_201596

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion 
  (x y z : ℝ) 
  (h_x : x = 3) 
  (h_y : y = -3 * Real.sqrt 3) 
  (h_z : z = 4) :
  ∃ (r θ : ℝ), 
    r > 0 ∧ 
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r = 6 ∧ 
    θ = 5 * Real.pi / 3 ∧
    r * (Real.cos θ) = x ∧
    r * (Real.sin θ) = y ∧
    z = 4 :=
by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l2015_201596


namespace NUMINAMATH_CALUDE_paint_for_similar_statues_l2015_201554

/-- Amount of paint needed for similar statues -/
theorem paint_for_similar_statues
  (original_height : ℝ)
  (original_paint : ℝ)
  (new_height : ℝ)
  (num_new_statues : ℕ)
  (h1 : original_height = 8)
  (h2 : original_paint = 1)
  (h3 : new_height = 2)
  (h4 : num_new_statues = 400) :
  (num_new_statues : ℝ) * original_paint * (new_height / original_height)^2 = 25 := by
  sorry

#check paint_for_similar_statues

end NUMINAMATH_CALUDE_paint_for_similar_statues_l2015_201554


namespace NUMINAMATH_CALUDE_count_perfect_squares_l2015_201540

theorem count_perfect_squares (max_value : Nat) (divisor : Nat) : 
  (Finset.filter (fun n : Nat => 
    n^2 % divisor = 0 ∧ n^2 < max_value) 
    (Finset.range max_value)).card = 175 :=
by sorry

#check count_perfect_squares (4 * 10^7) 36

end NUMINAMATH_CALUDE_count_perfect_squares_l2015_201540


namespace NUMINAMATH_CALUDE_ellipse_min_sum_l2015_201534

theorem ellipse_min_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 4/n = 1) :
  m + n ≥ 9 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 1/m₀ + 4/n₀ = 1 ∧ m₀ + n₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_min_sum_l2015_201534


namespace NUMINAMATH_CALUDE_second_train_speed_prove_second_train_speed_l2015_201546

/-- Calculates the speed of the second train given the conditions of the problem -/
theorem second_train_speed 
  (distance : ℝ) 
  (speed_first : ℝ) 
  (extra_distance : ℝ) : ℝ :=
  let speed_second := (3 * distance - 2 * extra_distance) / (6 * distance / speed_first - 2 * extra_distance / speed_first)
  speed_second

/-- Proves that the speed of the second train is 125/3 kmph given the problem conditions -/
theorem prove_second_train_speed :
  second_train_speed 1100 50 100 = 125/3 := by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_prove_second_train_speed_l2015_201546


namespace NUMINAMATH_CALUDE_balance_equation_l2015_201542

/-- The balance on a fuel card after refueling -/
def balance (initial_balance : ℝ) (price_per_liter : ℝ) (liters_refueled : ℝ) : ℝ :=
  initial_balance - price_per_liter * liters_refueled

/-- Theorem stating the functional relationship between balance and liters refueled -/
theorem balance_equation (x y : ℝ) :
  let initial_balance : ℝ := 1000
  let price_per_liter : ℝ := 7.92
  y = balance initial_balance price_per_liter x →
  y = 1000 - 7.92 * x :=
by sorry

end NUMINAMATH_CALUDE_balance_equation_l2015_201542


namespace NUMINAMATH_CALUDE_court_cases_guilty_l2015_201560

theorem court_cases_guilty (total : ℕ) (dismissed : ℕ) (delayed : ℕ) : 
  total = 27 → dismissed = 3 → delayed = 2 → 
  ∃ (guilty : ℕ), guilty = total - dismissed - (3 * (total - dismissed) / 4) - delayed ∧ guilty = 4 := by
sorry

end NUMINAMATH_CALUDE_court_cases_guilty_l2015_201560


namespace NUMINAMATH_CALUDE_h_equals_three_l2015_201599

-- Define the quadratic coefficients
variable (a b c : ℝ)

-- Define the condition that ax^2 + bx + c = 3(x - 3)^2 + 9
def quadratic_condition (a b c : ℝ) : Prop :=
  ∀ x, a * x^2 + b * x + c = 3 * (x - 3)^2 + 9

-- Define the transformed quadratic
def transformed_quadratic (a b c : ℝ) (x : ℝ) : ℝ :=
  5 * a * x^2 + 5 * b * x + 5 * c

-- Theorem stating that h = 3 in the transformed quadratic
theorem h_equals_three (a b c : ℝ) 
  (h : quadratic_condition a b c) :
  ∃ (m k : ℝ), ∀ x, transformed_quadratic a b c x = m * (x - 3)^2 + k :=
sorry

end NUMINAMATH_CALUDE_h_equals_three_l2015_201599


namespace NUMINAMATH_CALUDE_expression_evaluation_l2015_201526

theorem expression_evaluation :
  let x : ℤ := -2
  2 * x * (x - 3) - (x - 2) * (x + 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2015_201526


namespace NUMINAMATH_CALUDE_discount_percentage_l2015_201508

/-- Proves that given a cost price of 66.5, a marked price of 87.5, and a profit of 25% on the cost price, the percentage deducted from the list price is 5%. -/
theorem discount_percentage (cost_price : ℝ) (marked_price : ℝ) (profit_percentage : ℝ) :
  cost_price = 66.5 →
  marked_price = 87.5 →
  profit_percentage = 25 →
  let selling_price := cost_price * (1 + profit_percentage / 100)
  let discount_percentage := (marked_price - selling_price) / marked_price * 100
  discount_percentage = 5 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l2015_201508


namespace NUMINAMATH_CALUDE_second_smallest_hot_dog_packs_l2015_201547

theorem second_smallest_hot_dog_packs : 
  (∃ n : ℕ, n > 0 ∧ 12 * n % 8 = 6 ∧ 
   (∀ m : ℕ, m > 0 ∧ 12 * m % 8 = 6 → m ≥ n) ∧
   (∃ k : ℕ, k > 0 ∧ 12 * k % 8 = 6 ∧ k < n)) → 
  (∃ n : ℕ, n = 4 ∧ 12 * n % 8 = 6 ∧ 
   (∀ m : ℕ, m > 0 ∧ 12 * m % 8 = 6 → m = n ∨ m > n) ∧
   (∃ k : ℕ, k > 0 ∧ 12 * k % 8 = 6 ∧ k < n)) :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_hot_dog_packs_l2015_201547


namespace NUMINAMATH_CALUDE_point_Q_coordinates_l2015_201576

/-- Given a point P in ℝ² and a length l, this function returns the two possible
    points Q such that PQ is parallel to the x-axis and has length l -/
def possible_Q (P : ℝ × ℝ) (l : ℝ) : Set (ℝ × ℝ) :=
  {(P.1 + l, P.2), (P.1 - l, P.2)}

theorem point_Q_coordinates :
  let P : ℝ × ℝ := (2, 1)
  let l : ℝ := 3
  possible_Q P l = {(5, 1), (-1, 1)} := by
  sorry

#check point_Q_coordinates

end NUMINAMATH_CALUDE_point_Q_coordinates_l2015_201576
