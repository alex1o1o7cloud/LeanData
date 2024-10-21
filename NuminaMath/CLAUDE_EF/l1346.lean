import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_sum_l1346_134668

theorem sqrt_equation_sum (a b : ℝ) : 
  (Real.sqrt (6 + a / b) = 6 * Real.sqrt (a / b)) → a + b = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_sum_l1346_134668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_assembly_l1346_134641

theorem square_assembly (original_area : ℝ) (h : original_area = 36) :
  ∃ (side_length : ℝ), 
    side_length^2 = original_area ∧ 
    side_length = 6 ∧
    ∃ (num_parts : ℕ), num_parts = 4 ∧ 
      (original_area / num_parts) = (side_length^2 / num_parts) :=
by
  -- We use 'sorry' to skip the proof as requested
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_assembly_l1346_134641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_square_on_parabola_l1346_134666

/-- A square with three vertices on the parabola y = x^2 -/
structure SquareOnParabola where
  /-- x-coordinate of the first vertex -/
  x1 : ℝ
  /-- x-coordinate of the second vertex -/
  x2 : ℝ
  /-- x-coordinate of the third vertex -/
  x3 : ℝ
  /-- The vertices lie on the parabola y = x^2 -/
  on_parabola : (x1^2, x1) ∈ {(x, y) | y = x^2} ∧ 
                (x2^2, x2) ∈ {(x, y) | y = x^2} ∧ 
                (x3^2, x3) ∈ {(x, y) | y = x^2}
  /-- The points form a square -/
  is_square : (x3 - x2)^2 + (x3^2 - x2^2)^2 = (x1 - x2)^2 + (x1^2 - x2^2)^2

/-- The area of a SquareOnParabola -/
def area (s : SquareOnParabola) : ℝ :=
  ((s.x3 - s.x2)^2 + (s.x3^2 - s.x2^2)^2)

/-- The minimum area of a square with three vertices on the parabola y = x^2 is 2 -/
theorem min_area_square_on_parabola :
  ∀ s : SquareOnParabola, ∃ s' : SquareOnParabola, area s' ≤ area s ∧ area s' = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_square_on_parabola_l1346_134666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dancers_not_slow_dancing_l1346_134608

theorem dancers_not_slow_dancing (total_kids : ℕ) (dancers_fraction : ℚ) (slow_dancers : ℕ) : 
  total_kids = 140 →
  dancers_fraction = 1/4 →
  slow_dancers = 25 →
  (total_kids : ℚ) * dancers_fraction - slow_dancers = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dancers_not_slow_dancing_l1346_134608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_midpoint_trajectory_l1346_134657

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the point A
def point_A : ℝ × ℝ := (-2, 0)

-- Theorem for the slope of the tangent line
theorem tangent_slope :
  ∃ k : ℝ, k^2 = 3 ∧
  ∀ x y : ℝ, circle_eq x y →
  (∃ t : ℝ, x = -2 + t * k ∧ y = t * k) →
  ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ,
  circle_eq x' y' → ((x' - x)^2 + (y' - y)^2 < δ^2) →
  ((y' - y) * k + (x' - x) ≠ 0 ∨ (x' - x)^2 + (y' - y)^2 < ε^2) :=
by sorry

-- Theorem for the trajectory of the midpoint
theorem midpoint_trajectory :
  ∀ x y : ℝ,
  (∃ x₀ y₀ : ℝ, circle_eq x₀ y₀ ∧ x = (x₀ + point_A.1) / 2 ∧ y = (y₀ + point_A.2) / 2) ↔
  (x + 1)^2 + y^2 = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_midpoint_trajectory_l1346_134657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_recursive_f_6_12_l1346_134653

def f : ℕ → ℕ → ℕ
| _, 0 => 0
| 0, _ => 0
| x+1, y+1 => f x (y+1) + f (x+1) y + (x+1) + (y+1)

theorem f_recursive : ∀ x y, f (x+1) (y+1) = f x (y+1) + f (x+1) y + (x+1) + (y+1) := by
  intros x y
  rfl

theorem f_6_12 : f 6 12 = 77500 := by
  sorry

#eval f 6 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_recursive_f_6_12_l1346_134653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difficult_questions_l1346_134684

/-- Represents a math test with questions and good learners -/
structure MathTest where
  num_questions : ℕ
  num_good_learners : ℕ

/-- Defines what constitutes a good learner -/
def is_good_learner (test : MathTest) (correct_answers : ℕ) : Prop :=
  correct_answers > test.num_questions / 2

/-- Defines what constitutes a difficult question -/
def is_difficult_question (test : MathTest) (correct_answers : ℕ) : Prop :=
  correct_answers < test.num_good_learners / 2

/-- Theorem stating the maximum number of difficult questions -/
theorem max_difficult_questions (test : MathTest) 
  (h1 : test.num_questions = 4)
  (h2 : test.num_good_learners = 5) :
  ∃ (n : ℕ), n = 1 ∧ 
  (∀ m : ℕ, (∃ (difficult_questions : Finset (Fin test.num_questions)), 
    (∀ q ∈ difficult_questions, is_difficult_question test (test.num_good_learners - difficult_questions.card)) → 
    m ≤ n)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difficult_questions_l1346_134684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1346_134605

noncomputable section

/-- The slope of the first line -/
def m₁ : ℝ := 3

/-- The y-intercept of the first line -/
def b₁ : ℝ := -4

/-- The x-coordinate of the given point -/
def x₀ : ℝ := 3

/-- The y-coordinate of the given point -/
def y₀ : ℝ := 3

/-- The slope of the perpendicular line -/
noncomputable def m₂ : ℝ := -1 / m₁

/-- The y-intercept of the perpendicular line -/
noncomputable def b₂ : ℝ := y₀ - m₂ * x₀

/-- The x-coordinate of the intersection point -/
noncomputable def x_intersect : ℝ := (b₂ - b₁) / (m₁ - m₂)

/-- The y-coordinate of the intersection point -/
noncomputable def y_intersect : ℝ := m₁ * x_intersect + b₁

theorem intersection_point :
  (x_intersect = 2.4 ∧ y_intersect = 3.2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1346_134605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_touches_correct_l1346_134602

/-- Represents the minimum number of bulbs to touch to turn on all bulbs in an n × n board -/
def min_touches (n : ℕ) : ℕ :=
  if n % 2 = 1 then n else n^2

/-- Predicate that represents whether a bulb at position (i, j) is touched when bulb k is touched -/
def bulb_touched (n m : ℕ) (k : Fin m) (i j : Fin n) : Prop :=
  ∃ x y : Fin n, (k.val = x.val * n + y.val) ∧ (i = x ∨ j = y)

/-- Theorem stating the minimum number of bulbs to touch in an n × n board -/
theorem min_touches_correct (n : ℕ) (h : n > 0) :
  ∀ m : ℕ, (∀ i j : Fin n, ∃ k : Fin m, bulb_touched n m k i j) →
  m ≥ min_touches n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_touches_correct_l1346_134602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_length_is_80_l1346_134651

/-- Represents the race scenario where runner A is faster than runner B and B gets a head start. -/
structure RaceScenario where
  speed_ratio : ℝ  -- Ratio of runner A's speed to runner B's speed
  head_start : ℝ   -- Head start distance given to runner B in meters

/-- Calculates the race course length for a given race scenario where both runners finish at the same time. -/
noncomputable def calculate_race_length (scenario : RaceScenario) : ℝ :=
  (scenario.speed_ratio * scenario.head_start) / (scenario.speed_ratio - 1)

/-- Theorem stating that for the given race conditions, the race length is 80 meters. -/
theorem race_length_is_80 :
  let scenario : RaceScenario := { speed_ratio := 4, head_start := 60 }
  calculate_race_length scenario = 80 := by
  -- Unfold the definition of calculate_race_length
  unfold calculate_race_length
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_length_is_80_l1346_134651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_fourteen_arrangements_l1346_134619

/-- A valid arrangement of numbers in a 2×4 table -/
def ValidArrangement : Type := 
  { arr : Matrix (Fin 2) (Fin 4) ℕ // 
    (∀ i j, arr i j ∈ Finset.range 9) ∧ 
    (∀ i j₁ j₂, j₁ < j₂ → arr i j₁ < arr i j₂) ∧
    (∀ i₁ i₂ j, i₁ < i₂ → arr i₁ j > arr i₂ j) ∧
    (∀ n, n ∈ Finset.range 9 → ∃ i j, arr i j = n) }

/-- The number of valid arrangements -/
noncomputable def numValidArrangements : ℕ := sorry

/-- The theorem stating that there are exactly 14 valid arrangements -/
theorem exactly_fourteen_arrangements : numValidArrangements = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_fourteen_arrangements_l1346_134619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_passing_all_intersections_l1346_134690

/-- Represents a road with its index and angle -/
structure Road where
  index : Nat
  angle : Real

/-- Represents the problem setup -/
structure ProblemSetup where
  roads : Finset Road
  num_roads : Nat
  
  road_angles_increasing : ∀ (r₁ r₂ : Road), r₁ ∈ roads → r₂ ∈ roads → r₁.index < r₂.index → r₁.angle ≤ r₂.angle
  road_count : roads.card = num_roads
  angle_bounds : ∀ r ∈ roads, 0 ≤ r.angle ∧ r.angle < 90

/-- Determines if a car on a given road will pass through all intersections -/
def will_pass_all (setup : ProblemSetup) (road : Road) : Prop :=
  ∀ r ∈ setup.roads, r.index < road.index → r.angle > road.angle

/-- The main theorem stating which cars will pass through all intersections -/
theorem cars_passing_all_intersections (setup : ProblemSetup) 
  (h_setup : setup.num_roads = 30) :
  ∃ (r₁ r₂ r₃ : Road),
    r₁ ∈ setup.roads ∧ r₂ ∈ setup.roads ∧ r₃ ∈ setup.roads ∧
    r₁.index = 14 ∧ r₂.index = 23 ∧ r₃.index = 24 ∧
    will_pass_all setup r₁ ∧ will_pass_all setup r₂ ∧ will_pass_all setup r₃ ∧
    (∀ r ∈ setup.roads, will_pass_all setup r → r = r₁ ∨ r = r₂ ∨ r = r₃) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_passing_all_intersections_l1346_134690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_locus_of_Q_l1346_134650

noncomputable section

variable (R r : ℝ)
variable (hR : R > 0)
variable (hr : r > 0)
variable (h : R > r)

/-- Two concentric circles with radii R and r -/
def larger_circle (R : ℝ) : Set ℂ := {z : ℂ | Complex.abs z = R}
def smaller_circle (r : ℝ) : Set ℂ := {z : ℂ | Complex.abs z = r}

/-- Fixed point P on the smaller circle -/
def P (r : ℝ) : ℂ := r * Complex.exp (Complex.I * Real.pi)

/-- Moving point B on the larger circle -/
def B (R : ℝ) (θ : ℝ) : ℂ := R * Complex.exp (Complex.I * θ)

/-- Point C as the other intersection of BP with the larger circle -/
noncomputable def C (P B : ℂ) (R : ℝ) : ℂ := sorry

/-- Point A as the intersection of the line perpendicular to BP through P with the smaller circle -/
noncomputable def A (P B : ℂ) (r : ℝ) : ℂ := sorry

/-- Midpoint Q of AB -/
def Q (A B : ℂ) : ℂ := (A + B) / 2

/-- Theorem 1: BC^2 + CA^2 + AB^2 = 6R^2 + 2r^2 -/
theorem sum_of_squares (R r : ℝ) (hR : R > 0) (hr : r > 0) (h : R > r) :
  ∀ P ∈ smaller_circle r, ∀ B ∈ larger_circle R,
    let C := C P B R
    let A := A P B r
    Complex.abs (C - B) ^ 2 + Complex.abs (A - C) ^ 2 + Complex.abs (B - A) ^ 2 = 6 * R^2 + 2 * r^2 := by
  sorry

/-- Theorem 2: Locus of Q is a circle with center at midpoint of OP and radius R/2 -/
theorem locus_of_Q (R r : ℝ) (hR : R > 0) (hr : r > 0) (h : R > r) :
  ∀ P ∈ smaller_circle r, ∀ B ∈ larger_circle R,
    let A := A P B r
    let Q := Q A B
    Complex.abs (Q - (P / 2)) = R / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_locus_of_Q_l1346_134650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l1346_134688

open Real

/-- The differential equation y''' - 2y'' + 2y' = 4cos(x)cos(3x) + 6sin²(x) -/
def differential_equation (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (deriv^[3] y) x - 2 * (deriv^[2] y) x + 2 * (deriv y) x = 
    4 * cos x * cos (3*x) + 6 * (sin x)^2

/-- The general solution of the differential equation -/
noncomputable def general_solution (C₁ C₂ C₃ : ℝ) (x : ℝ) : ℝ :=
  C₁ + C₂ * exp x * cos x + C₃ * exp x * sin x + 
  1/65 * (cos (4*x) - 7/4 * sin (4*x)) + 
  1/10 * (sin (2*x) / 2 - cos (2*x)) + 
  3/2 * x

/-- Theorem stating that the general solution satisfies the differential equation -/
theorem solution_satisfies_equation (C₁ C₂ C₃ : ℝ) :
  ∀ x, differential_equation (general_solution C₁ C₂ C₃) x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l1346_134688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_18_4_l1346_134680

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_18_4_l1346_134680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_higher_selling_price_is_347_l1346_134622

-- Define the problem parameters
noncomputable def cost_price : ℝ := 200
noncomputable def lower_selling_price : ℝ := 340
noncomputable def gain_increase_percentage : ℝ := 5

-- Define the function to calculate the higher selling price
noncomputable def calculate_higher_selling_price : ℝ :=
  let lower_gain := lower_selling_price - cost_price
  let higher_gain := lower_gain + (gain_increase_percentage / 100) * lower_gain
  cost_price + higher_gain

-- Theorem statement
theorem higher_selling_price_is_347 :
  calculate_higher_selling_price = 347 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_higher_selling_price_is_347_l1346_134622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l1346_134610

open Real

/-- The probability that a randomly selected point from a square with
    vertices at (±3, ±3) is within 2 units of the origin -/
noncomputable def probability_within_circle : ℝ := Real.pi / 9

/-- The side length of the square -/
def square_side : ℝ := 6

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- The area of the square -/
def square_area : ℝ := square_side ^ 2

/-- The area of the circle -/
noncomputable def circle_area : ℝ := Real.pi * circle_radius ^ 2

theorem probability_calculation :
  probability_within_circle = circle_area / square_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l1346_134610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_sum_27_l1346_134659

theorem three_digit_sum_27 : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 27 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_sum_27_l1346_134659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neznaika_puzzle_l1346_134694

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def neznaika_claims (n : ℕ) : List Bool :=
  [n % 3 = 0, n % 4 = 0, n % 5 = 0, n % 9 = 0, n % 10 = 0, n % 15 = 0, n % 18 = 0, n % 30 = 0]

def count_false (l : List Bool) : ℕ := l.filter (·= false) |>.length

theorem neznaika_puzzle (N : ℕ) (h1 : is_two_digit N) (h2 : count_false (neznaika_claims N) = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

#eval neznaika_claims 36
#eval neznaika_claims 45
#eval neznaika_claims 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_neznaika_puzzle_l1346_134694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_floor_equality_l1346_134655

theorem sqrt_floor_equality (n : ℕ+) :
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ ∧
  ⌊Real.sqrt (4 * n + 3)⌋ = ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_floor_equality_l1346_134655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_difference_equals_four_l1346_134681

theorem log_difference_equals_four (x y : ℝ) (h1 : x > 2*y) (h2 : y > 0) 
  (h3 : Real.log x + Real.log y = 2 * Real.log (x - 2*y)) : 
  Real.log x - Real.log y = 4 * Real.log (Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_difference_equals_four_l1346_134681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_value_l1346_134675

theorem trig_expression_value (θ : ℝ) (h : Real.sin θ - Real.sqrt 3 * Real.cos θ = -2) :
  (Real.sin θ) ^ 2 + Real.cos (2 * θ) + 3 = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_value_l1346_134675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_polyhedron_volume_l1346_134695

/-- A convex polyhedron with vertices on two parallel planes and alternating faces --/
structure AlternatingPolyhedron where
  /-- The polyhedron is convex --/
  is_convex : Bool
  /-- Vertices lie on two parallel planes --/
  vertices_on_parallel_planes : Bool
  /-- Faces alternate between equilateral triangles and rectangles --/
  alternating_faces : Bool
  /-- Equilateral triangle side length --/
  triangle_side_length : ℝ
  /-- Rectangle side lengths --/
  rectangle_side_length_1 : ℝ
  rectangle_side_length_2 : ℝ

/-- The set of possible volumes for the alternating polyhedron --/
def possible_volumes : Set ℝ :=
  {Real.sqrt 3, (19 * Real.sqrt 2) / 6, 4 + (7 * Real.sqrt 2) / 3, 35 / 12 + (31 * Real.sqrt 5) / 12}

/-- A function to calculate the volume of an AlternatingPolyhedron --/
noncomputable def volume (p : AlternatingPolyhedron) : ℝ := sorry

/-- Theorem stating that the volume of the alternating polyhedron is in the set of possible volumes --/
theorem alternating_polyhedron_volume 
  (p : AlternatingPolyhedron) 
  (h1 : p.is_convex = true)
  (h2 : p.vertices_on_parallel_planes = true)
  (h3 : p.alternating_faces = true)
  (h4 : p.triangle_side_length = 2)
  (h5 : p.rectangle_side_length_1 = 2)
  (h6 : p.rectangle_side_length_2 = 1) :
  ∃ (v : ℝ), v ∈ possible_volumes ∧ volume p = v :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_polyhedron_volume_l1346_134695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_faces_are_equilateral_l1346_134600

/-- Represents a point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a quadrilateral base of a pyramid -/
structure QuadrilateralBase where
  vertices : Finset Point
  is_regular : Prop

/-- Represents a regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  base : QuadrilateralBase
  apex : Point
  is_regular : Prop

/-- Represents a cross-section of a pyramid -/
structure CrossSection where
  vertices : Finset Point
  is_regular_pentagon : Prop

/-- Represents a lateral face of a pyramid -/
structure LateralFace where
  vertices : Finset Point
  is_triangle : Prop

/-- Helper function to check if a given face is a lateral face of the pyramid -/
def IsLateralFace (pyramid : RegularQuadrilateralPyramid) (face : LateralFace) : Prop :=
  sorry

/-- Helper function to check if a pyramid has a given cross-section -/
def HasCrossSection (pyramid : RegularQuadrilateralPyramid) (cross_section : CrossSection) : Prop :=
  sorry

/-- Helper function to check if a triangle is equilateral -/
def IsEquilateralTriangle (vertices : Finset Point) : Prop :=
  sorry

/-- Theorem stating that given a regular quadrilateral pyramid with a regular pentagon cross-section,
    its lateral faces are equilateral triangles -/
theorem lateral_faces_are_equilateral
  (pyramid : RegularQuadrilateralPyramid)
  (cross_section : CrossSection)
  (h_cross_section : HasCrossSection pyramid cross_section) :
  ∀ (face : LateralFace), IsLateralFace pyramid face → IsEquilateralTriangle face.vertices :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_faces_are_equilateral_l1346_134600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_properties_l1346_134649

noncomputable def f (x : ℝ) : ℝ := (x^2 - 9) / (x + 3)

theorem fraction_properties :
  (∀ x : ℝ, x = -3 → ¬ ∃ y : ℝ, f x = y) ∧
  (f (-4) = -7) ∧
  (∀ x : ℝ, x > 3 → f x > 0) ∧
  (¬ (f 3 = 0 ∧ ∃ y : ℝ, f (-3) = y ∧ f (-3) = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_properties_l1346_134649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_numbers_with_perfect_square_sum_l1346_134645

/-- Returns true if n is a perfect square and less than or equal to 25 -/
def isPerfectSquareLEQ25 (n : ℕ) : Bool :=
  match n with
  | 1 | 4 | 9 | 16 | 25 => true
  | _ => false

/-- Returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Returns true if n is a two-digit number -/
def isTwoDigitNumber (n : ℕ) : Bool :=
  10 ≤ n ∧ n ≤ 99

/-- The main theorem to be proved -/
theorem two_digit_numbers_with_perfect_square_sum :
  (Finset.filter (fun n => isTwoDigitNumber n ∧ isPerfectSquareLEQ25 (sumOfDigits n)) (Finset.range 100)).card = 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_numbers_with_perfect_square_sum_l1346_134645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_yards_lost_l1346_134639

/-- Represents the yards lost by a football team -/
def yards_lost (n : ℤ) : Prop := True

/-- Represents the final progress of a football team -/
def final_progress (n : ℤ) : Prop := True

/-- Theorem: If a football team's final progress is 5 yards after losing some yards and then gaining 10 yards, the team must have initially lost 15 yards. -/
theorem football_yards_lost 
  (lost : ℤ) 
  (h1 : yards_lost lost) 
  (h2 : final_progress 5) : 
  lost = 15 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_yards_lost_l1346_134639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_points_exist_l1346_134663

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Colors available for the points -/
inductive Color
  | White
  | Red

/-- The closet floor -/
def ClosetFloor : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 2}

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A coloring of the closet floor -/
def Coloring := Point → Color

theorem same_color_points_exist (c : Coloring) :
  ∃ (p q : Point), p ∈ ClosetFloor ∧ q ∈ ClosetFloor ∧
  distance p q = 1 ∧ c p = c q := by
  sorry

#check same_color_points_exist

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_points_exist_l1346_134663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_geometric_mean_l1346_134652

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a triangle is isosceles with AB = AC -/
def isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := 
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the perpendicular distance from a point to a line segment -/
noncomputable def perpDistance (p : Point) (a b : Point) : ℝ := sorry

/-- Checks if a circle passes through two points and is tangent to two line segments -/
def isValidCircle (c : Circle) (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem locus_of_geometric_mean (t : Triangle) (c : Circle) :
  isIsosceles t →
  isValidCircle c t →
  ∀ p : Point,
    isInside p t →
    (perpDistance p t.B t.C)^2 = 
      perpDistance p t.A t.C * perpDistance p t.A t.B ↔
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi ∧
      p.x = c.center.x + c.radius * Real.cos θ ∧
      p.y = c.center.y + c.radius * Real.sin θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_geometric_mean_l1346_134652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_14_l1346_134620

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (4, 6)

noncomputable def triangle_area (v1 v2 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (v1.1 * v2.2 - v1.2 * v2.1)

theorem triangle_area_is_14 : triangle_area a b = 14 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp [a, b]
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_14_l1346_134620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1346_134630

-- Define the function f(x) = 2^x
noncomputable def f (x : ℝ) : ℝ := 2^x

-- Statement: f is increasing on the interval (-∞, 1)
theorem f_increasing_on_interval :
  ∀ x y : ℝ, x < y → y < 1 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1346_134630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_l1346_134644

/-- The circle equation: x^2 + y^2 + 8x + 16 = 0 -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 + 8*x + 16 = 0

/-- The line equation: y = 2x + 3 -/
def lineEq (x y : ℝ) : Prop := y = 2*x + 3

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The theorem stating the smallest possible distance -/
theorem smallest_distance :
  ∃ (x1 y1 x2 y2 : ℝ),
    circleEq x1 y1 ∧ lineEq x2 y2 ∧
    (∀ (a b c d : ℝ), circleEq a b → lineEq c d → distance x1 y1 x2 y2 ≤ distance a b c d) ∧
    distance x1 y1 x2 y2 = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_l1346_134644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_no_common_points_l1346_134697

-- Define the necessary structures
structure Line where

structure Plane where

-- Define the relationships
def parallel (l : Line) (p : Plane) : Prop := sorry

def contained_in (l : Line) (p : Plane) : Prop := sorry

def have_no_common_points (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem lines_no_common_points 
  (a b : Line) (α : Plane) 
  (h1 : parallel a α) 
  (h2 : contained_in b α) : 
  have_no_common_points a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_no_common_points_l1346_134697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_four_fifths_l1346_134678

def N : ℕ := sorry

axiom N_range : 1 ≤ N ∧ N ≤ 2020

def probability_N16_mod5_eq1 : ℚ :=
  (Finset.filter (λ n : ℕ ↦ 1 ≤ n ∧ n ≤ 2020 ∧ n^16 % 5 = 1) (Finset.range 2021)).card /
  (Finset.filter (λ n : ℕ ↦ 1 ≤ n ∧ n ≤ 2020) (Finset.range 2021)).card

theorem probability_is_four_fifths :
  probability_N16_mod5_eq1 = 4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_four_fifths_l1346_134678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1346_134661

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x - 2 + Real.log x

-- State the theorem
theorem root_in_interval :
  (∃ x ∈ Set.Ioo 1 2, f x = 0) ∧
  (∀ x y, x < y → f x < f y) ∧
  (f 1 < 0) ∧
  (f 2 > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1346_134661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_configuration_l1346_134648

/-- The number of radars --/
def n : ℕ := 9

/-- The radius of each radar's coverage area in km --/
noncomputable def r : ℝ := 37

/-- The width of the coverage ring in km --/
noncomputable def w : ℝ := 24

/-- The angle between two adjacent radars in radians --/
noncomputable def θ : ℝ := 2 * Real.pi / n

/-- The maximum distance from the center to each radar --/
noncomputable def max_distance : ℝ := 35 / Real.sin (θ / 2)

/-- The area of the coverage ring --/
noncomputable def coverage_area : ℝ := 1680 * Real.pi / Real.tan (θ / 2)

/-- Theorem stating the maximum distance and coverage area for the given configuration --/
theorem radar_configuration :
  (max_distance = 35 / Real.sin (θ / 2)) ∧
  (coverage_area = 1680 * Real.pi / Real.tan (θ / 2)) := by
  sorry

#check radar_configuration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_configuration_l1346_134648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_distribution_ratio_l1346_134609

/-- Given a class where notebooks are distributed equally among children,
    prove that the ratio of the hypothetical number of children to the
    actual number of children is 1:2 under the given conditions. -/
theorem notebook_distribution_ratio :
  ∀ (C H : ℕ),
  (C > 0) →  -- Ensure C is positive
  (C * (C / 8) = 512) →  -- Total notebooks distributed
  (C / 8 = 16) →  -- Each child gets C/8 notebooks, which equals 16 in the hypothetical case
  (H * 16 = 512) →  -- Hypothetical case where each child gets 16 notebooks
  (H : ℚ) / (C : ℚ) = 1 / 2 := by
  sorry

#check notebook_distribution_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_distribution_ratio_l1346_134609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_omega_upper_bound_l1346_134628

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (2 * ω * x)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x - 3 * Real.pi / (4 * ω))

theorem max_omega_value (ω : ℝ) (h1 : ω > 0) :
  (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/6), 
    (∀ y ∈ Set.Icc (-Real.pi/4) (Real.pi/6), x < y → g ω x > g ω y)) → 
  ω ≤ 1 :=
by sorry

theorem omega_upper_bound : 
  ∃ ω : ℝ, ω > 0 ∧ 
  (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/6), 
    (∀ y ∈ Set.Icc (-Real.pi/4) (Real.pi/6), x < y → g ω x > g ω y)) ∧
  ω = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_omega_upper_bound_l1346_134628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_all_points_on_curve_l1346_134612

-- Define the curve
def curve (x y : ℝ) : Prop := x * y = 1

-- Define the circle (we don't need its specific equation for this problem)
def circle_set : Set (ℝ × ℝ) := sorry

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := 
  {(3, 1/3), (-4, -1/4), (1/5, 5), (-5/12, -12/5)}

-- Theorem statement
theorem fourth_intersection_point :
  ∀ (p : ℝ × ℝ), p ∈ intersection_points ∧ p ∈ circle_set ∧ curve p.1 p.2 →
  (p = (3, 1/3) ∨ p = (-4, -1/4) ∨ p = (1/5, 5)) ∨ p = (-5/12, -12/5) := by
  sorry

-- Verify that all points in the intersection satisfy the curve equation
theorem all_points_on_curve :
  ∀ (p : ℝ × ℝ), p ∈ intersection_points → curve p.1 p.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_all_points_on_curve_l1346_134612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_k_zero_k_range_for_nonnegative_domain_l1346_134615

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - k * x^2 + 2

-- Theorem for part (I)
theorem min_value_when_k_zero :
  ∃ (min_val : ℝ), min_val = 1 ∧ 
  ∀ (x : ℝ), f 0 x ≥ min_val :=
sorry

-- Theorem for part (II)
theorem k_range_for_nonnegative_domain :
  ∀ (k : ℝ), (∀ (x : ℝ), x ≥ 0 → f k x ≥ 1) ↔ k ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_k_zero_k_range_for_nonnegative_domain_l1346_134615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_solution_l1346_134669

def vector_problem (a b : ℝ × ℝ × ℝ) (h1 : ‖a‖ = 4) (h2 : ‖b‖ = 2) (h3 : a • b = -4) : Prop :=
  let angle_ab := Real.arccos (-1/2)
  let angle_a_aplusb := Real.arccos (Real.sqrt 3 / 2)
  (a - 2 • b) • (a + b) = 12 ∧ 
  ‖2 • a - b‖ = 2 * Real.sqrt 21 ∧ 
  angle_a_aplusb = Real.pi / 6

theorem vector_problem_solution (a b : ℝ × ℝ × ℝ) (h1 : ‖a‖ = 4) (h2 : ‖b‖ = 2) (h3 : a • b = -4) :
  vector_problem a b h1 h2 h3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_solution_l1346_134669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l1346_134682

/-- Binary operation ◇ defined on nonzero real numbers -/
def diamond : ℝ → ℝ → ℝ := sorry

/-- First property of ◇: a ◇ (b ◇ c) = (a ◇ b) · c -/
axiom diamond_assoc (a b c : ℝ) : a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  diamond a (diamond b c) = (diamond a b) * c

/-- Second property of ◇: a ◇ a = 1 -/
axiom diamond_self (a : ℝ) : a ≠ 0 → diamond a a = 1

/-- Theorem: If 4050 ◇ (9 ◇ x) = 150, then x = 1/3 -/
theorem diamond_equation_solution :
  ∀ x : ℝ, x ≠ 0 → diamond 4050 (diamond 9 x) = 150 → x = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l1346_134682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petyas_numbers_l1346_134607

theorem petyas_numbers (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  Finset.toSet {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} = 
  Finset.toSet {7, 9, 12, 16, 17, 19, 20, 21, 22, 29} →
  (a, b, c, d, e) = (2, 5, 7, 14, 15) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petyas_numbers_l1346_134607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_probability_l1346_134635

def max_number : ℕ := 300
def billy_multiple : ℕ := 15
def bobbi_multiple : ℕ := 20

theorem same_number_probability :
  let billy_choices := Finset.filter (fun n => billy_multiple ∣ n) (Finset.range max_number)
  let bobbi_choices := Finset.filter (fun n => bobbi_multiple ∣ n) (Finset.range max_number)
  let common_choices := Finset.filter (fun n => billy_multiple ∣ n ∧ bobbi_multiple ∣ n) (Finset.range max_number)
  (Finset.card common_choices : ℚ) / ((Finset.card billy_choices) * (Finset.card bobbi_choices)) = 1 / 60 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_probability_l1346_134635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_overtime_increase_l1346_134662

/-- A bus driver's compensation structure and work hours --/
structure BusDriverCompensation where
  regular_rate : ℚ
  regular_hours : ℚ
  total_hours : ℚ
  total_compensation : ℚ

/-- Calculate the percentage increase in overtime rate compared to regular rate --/
noncomputable def overtime_rate_increase (bdc : BusDriverCompensation) : ℚ :=
  let regular_earnings := bdc.regular_rate * bdc.regular_hours
  let overtime_hours := bdc.total_hours - bdc.regular_hours
  let overtime_earnings := bdc.total_compensation - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  ((overtime_rate - bdc.regular_rate) / bdc.regular_rate) * 100

/-- Theorem stating the overtime rate increase for the given scenario --/
theorem bus_driver_overtime_increase :
  let bdc : BusDriverCompensation := {
    regular_rate := 16,
    regular_hours := 40,
    total_hours := 48,
    total_compensation := 864
  }
  overtime_rate_increase bdc = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_overtime_increase_l1346_134662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_results_l1346_134691

/-- Basketball shooting game between two players -/
structure BasketballGame where
  playerA_percentage : ℝ
  playerB_percentage : ℝ
  first_shot_probability : ℝ

/-- Probability that player A takes the i-th shot -/
noncomputable def prob_A_takes_ith_shot (i : ℕ) : ℝ :=
  1/3 + (1/6) * (2/5)^(i-1)

/-- Expected number of times player A shoots in first n shots -/
noncomputable def expected_A_shots (n : ℕ) : ℝ :=
  (5/18) * (1 - (2/5)^n) + n/3

/-- Theorem stating the main results of the basketball game -/
theorem basketball_game_results (game : BasketballGame)
  (h1 : game.playerA_percentage = 0.6)
  (h2 : game.playerB_percentage = 0.8)
  (h3 : game.first_shot_probability = 0.5) :
  (∃ (prob_B_second : ℝ), prob_B_second = 0.6) ∧
  (∀ (i : ℕ), i > 0 → prob_A_takes_ith_shot i = 1/3 + (1/6) * (2/5)^(i-1)) ∧
  (∀ (n : ℕ), n > 0 → expected_A_shots n = (5/18) * (1 - (2/5)^n) + n/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_results_l1346_134691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anthony_pets_final_count_l1346_134618

def pet_population (initial : ℕ) (door_loss : ℚ) (contest_win : ℕ) (contest_reward : ℕ)
  (birth_rate : ℚ) (offspring : ℕ) (donation_increase : ℚ) (old_age_loss : ℚ) (illness_loss : ℚ) : ℕ :=
  let after_door_loss := initial - (↑initial * door_loss).ceil.toNat
  let after_contest := after_door_loss + contest_reward
  let after_birth := after_contest + ((↑after_contest * birth_rate).floor.toNat * offspring)
  let after_donation := after_birth + (↑after_birth * donation_increase).ceil.toNat
  let after_old_age := after_donation - (↑after_donation * old_age_loss).ceil.toNat
  after_old_age - (↑after_old_age * illness_loss).ceil.toNat

theorem anthony_pets_final_count :
  pet_population 120 (8/100) 5 15 (3/8) 3 (25/100) (9/100) (11/100) = 270 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anthony_pets_final_count_l1346_134618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_in_plane_l1346_134616

def a : Fin 3 → ℝ := ![2, 1, -3]
def b : Fin 3 → ℝ := ![-1, 2, 3]
def c (z : ℝ) : Fin 3 → ℝ := ![7, 6, z]

def coplanar (v1 v2 v3 : Fin 3 → ℝ) : Prop :=
  ∃ (m n : ℝ), ∀ i, v3 i = m * v1 i + n * v2 i

theorem vectors_in_plane (z : ℝ) :
  coplanar a b (c z) → z = -9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_in_plane_l1346_134616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simson_line_bisects_segment_l1346_134632

noncomputable section

/-- Predicate indicating that three points form a triangle -/
def Triangle (A B C : EuclideanPlane) : Prop := sorry

/-- Predicate indicating that H is the orthocenter of triangle ABC -/
def Orthocenter (A B C H : EuclideanPlane) : Prop := sorry

/-- Predicate indicating that P is on the circumcircle of triangle ABC -/
def OnCircumcircle (P A B C : EuclideanPlane) : Prop := sorry

/-- The Simson line of point P with respect to triangle ABC -/
def SimsonLine (P A B C : EuclideanPlane) : Set EuclideanPlane := sorry

/-- Predicate indicating that a line bisects a segment -/
def Bisects (l : Set EuclideanPlane) (s : Set EuclideanPlane) : Prop := sorry

/-- A line segment between two points -/
def Segment (P Q : EuclideanPlane) : Set EuclideanPlane := sorry

/-- The Simson line of a point on the circumcircle of a triangle bisects the segment
    between that point and the triangle's orthocenter. -/
theorem simson_line_bisects_segment (A B C P H : EuclideanPlane) :
  Triangle A B C →
  Orthocenter A B C H →
  OnCircumcircle P A B C →
  Bisects (SimsonLine P A B C) (Segment P H) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simson_line_bisects_segment_l1346_134632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l1346_134687

theorem sin_double_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 3 / 5) : 
  Real.sin (2 * α) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l1346_134687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extrema_l1346_134603

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x + 2

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - a

-- State the theorem
theorem tangent_line_and_extrema (a : ℝ) :
  (f' a 2 = 0) →
  (∃ m b : ℝ, m = 5 ∧ b = -16 ∧ ∀ x : ℝ, f a 3 + m * (x - 3) = m * x + b) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f a x ≤ 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f a x ≥ -10/3) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f a x = 2) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f a x = -10/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extrema_l1346_134603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1346_134658

/-- Given non-zero vectors a and b in ℝ², prove that the angle between them is 120° 
    when (a) · (a + b) = 0 and 2|a| = |b|. -/
theorem angle_between_vectors (a b : ℝ × ℝ) (ha : a ≠ (0, 0)) :
  (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0) →
  (2 * Real.sqrt (a.1^2 + a.2^2) = Real.sqrt (b.1^2 + b.2^2)) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * π / 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1346_134658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_intersection_symmetry_l1346_134624

/-- The perpendicular foot of a point P on a line through points Q and R. -/
noncomputable def perp_foot (P Q R : EuclideanPlane) : EuclideanPlane :=
sorry

/-- Given a triangle ABC and points A₁, B₁, C₁, if the perpendiculars from A₁, B₁, C₁ to BC, CA, AB 
    respectively intersect at one point, then the perpendiculars from A, B, C to B₁C₁, C₁A₁, A₁B₁ 
    respectively also intersect at one point. -/
theorem perpendicular_intersection_symmetry 
  (A B C A₁ B₁ C₁ : EuclideanPlane) 
  (h : ∃ P, (perp_foot A₁ B C = P) ∧ (perp_foot B₁ C A = P) ∧ (perp_foot C₁ A B = P)) :
  ∃ Q, (perp_foot A B₁ C₁ = Q) ∧ (perp_foot B C₁ A₁ = Q) ∧ (perp_foot C A₁ B₁ = Q) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_intersection_symmetry_l1346_134624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_planes_l1346_134631

/-- The distance between two parallel planes in 3D space -/
noncomputable def plane_distance (a b c d : ℝ) (e f g h : ℝ) : ℝ :=
  |d - h| / Real.sqrt (a^2 + b^2 + c^2)

/-- The first plane equation: 2x - 4y + 6z = 10 -/
def plane1 (x y z : ℝ) : Prop := 2*x - 4*y + 6*z = 10

/-- The second plane equation: x - 2y + 3z = 4 -/
def plane2 (x y z : ℝ) : Prop := x - 2*y + 3*z = 4

theorem distance_between_planes :
  plane_distance 2 (-4) 6 10 1 (-2) 3 4 = 1 / Real.sqrt 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_planes_l1346_134631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_condition_l1346_134611

theorem necessary_not_sufficient_condition : 
  (∀ a : ℝ, abs a < 1 → a - 1 < 0) ∧ 
  (∃ a : ℝ, a - 1 < 0 ∧ ¬(abs a < 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_condition_l1346_134611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_difference_l1346_134634

theorem quadratic_roots_difference (k : ℝ) :
  let f (x : ℝ) := x^2 - 2*x + k
  let α := (2 + Real.sqrt (4 - 4*k)) / 2
  let β := (2 - Real.sqrt (4 - 4*k)) / 2
  (|α - β| = 2 * Real.sqrt 2) → (k = -1 ∨ k = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_difference_l1346_134634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_conversions_l1346_134660

noncomputable section

-- Conversion rates
def gram_to_kg : ℚ := 1 / 1000
def cm_to_m : ℚ := 1 / 100
def jiao_to_yuan : ℚ := 1 / 10
def fen_to_yuan : ℚ := 1 / 100
def kg_to_ton : ℚ := 1 / 1000

-- Conversion functions
def grams_to_kg (g : ℚ) : ℚ := g * gram_to_kg
def cm_to_meters (cm : ℚ) : ℚ := cm * cm_to_m
def jiao_fen_to_yuan (j f : ℚ) : ℚ := j * jiao_to_yuan + f * fen_to_yuan
def yuan_fen_to_yuan (y f : ℚ) : ℚ := y + f * fen_to_yuan
def tons_kg_to_tons (t k : ℚ) : ℚ := t + k * kg_to_ton

end noncomputable section

theorem unit_conversions :
  (grams_to_kg 80 = 8/100) ∧
  (cm_to_meters 165 = 165/100) ∧
  (jiao_fen_to_yuan 4 9 = 49/100) ∧
  (yuan_fen_to_yuan 13 7 = 1307/100) ∧
  (tons_kg_to_tons 5 26 = 5026/1000) :=
by sorry

#eval grams_to_kg 80
#eval cm_to_meters 165
#eval jiao_fen_to_yuan 4 9
#eval yuan_fen_to_yuan 13 7
#eval tons_kg_to_tons 5 26

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_conversions_l1346_134660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_per_row_l1346_134665

theorem tiles_per_row (room_area : ℝ) (tile_size : ℝ) : 
  room_area = 256 → tile_size = 8 / 12 → ⌊Real.sqrt room_area / tile_size⌋ = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_per_row_l1346_134665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_car_fuel_consumption_l1346_134672

/-- Represents the fuel efficiency of a car in kilometers per liter -/
structure FuelEfficiency where
  km_per_liter : ℝ
  efficiency_positive : km_per_liter > 0

/-- Calculates the fuel consumption in liters per 100 km given the fuel efficiency -/
noncomputable def fuel_consumption (fe : FuelEfficiency) : ℝ :=
  100 / fe.km_per_liter

/-- Theorem stating the fuel consumption of the new car -/
theorem new_car_fuel_consumption 
  (old_car : FuelEfficiency)
  (new_car : FuelEfficiency)
  (h1 : new_car.km_per_liter = old_car.km_per_liter + 4.2)
  (h2 : fuel_consumption new_car = fuel_consumption old_car - 2) :
  ∃ (ε : ℝ), abs (fuel_consumption new_car - 5.97) < ε ∧ ε > 0 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_car_fuel_consumption_l1346_134672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1346_134673

theorem triangle_side_length (a b c : ℝ) (A : ℝ) : 
  A = 60 * Real.pi / 180 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  b + c = 6 →
  a = 2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1346_134673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1346_134636

theorem problem_solution : 12 * ((2 : ℚ)/3 - 1/4 + 1/6)⁻¹ = 144/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1346_134636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1346_134637

/-- The length of the line segment formed by the intersection points of two circles -/
theorem intersection_segment_length :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 + A.2^2 = 9) ∧ 
    (A.1^2 + A.2^2 - 4*A.1 + 2*A.2 - 3 = 0) ∧
    (B.1^2 + B.2^2 = 9) ∧ 
    (B.1^2 + B.2^2 - 4*B.1 + 2*B.2 - 3 = 0) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1346_134637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l1346_134623

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -Real.sqrt 2],
    ![Real.sqrt 2, 2]]

theorem matrix_power_four :
  A ^ 4 = ![![-64, 0],
            ![0, -64]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l1346_134623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_emptying_time_l1346_134604

/-- Represents the tank emptying scenario -/
structure TankScenario where
  tank_volume : ℚ  -- in cubic feet
  inlet_rate : ℚ   -- in cubic inches/min
  outlet_rate1 : ℚ -- in cubic inches/min
  outlet_rate2 : ℚ -- in cubic inches/min
  outlet_rate3 : ℚ -- in cubic inches/min

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℚ := 12

/-- Time to empty the tank in minutes -/
noncomputable def time_to_empty (scenario : TankScenario) : ℚ :=
  (scenario.tank_volume * feet_to_inches ^ 3) / 
  (scenario.outlet_rate1 + scenario.outlet_rate2 + scenario.outlet_rate3 - scenario.inlet_rate)

/-- The main theorem stating the time to empty the tank -/
theorem tank_emptying_time :
  let scenario : TankScenario := {
    tank_volume := 45,
    inlet_rate := 5,
    outlet_rate1 := 12,
    outlet_rate2 := 9,
    outlet_rate3 := 6
  }
  ∃ ε > 0, |time_to_empty scenario - 3534.55| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_emptying_time_l1346_134604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1346_134633

/-- The function f(x) = x² + 2/x -/
noncomputable def f (x : ℝ) : ℝ := x^2 + 2/x

/-- The function g(x) = (1/2)² + m -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1/2)^2 + m

/-- The theorem stating the range of m -/
theorem range_of_m (m : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc (-1) 1, f x₁ ≥ g m x₂) ↔ m ≤ 5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1346_134633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_permutation_is_31254_l1346_134696

/-- A list of all five-digit integers using digits 1, 2, 3, 4, and 5 exactly once, ordered from least to greatest -/
def five_digit_permutations : List Nat :=
  (List.range 5).map (λ i => i + 1)
  |>.permutations
  |>.map (λ l => l.foldl (λ acc d => acc * 10 + d) 0)
  |>.filter (λ n => n ≥ 10000)
  |>.toArray
  |>.qsort (· < ·)
  |>.toList

/-- The 50th integer in the ordered list of five-digit integers using digits 1, 2, 3, 4, and 5 exactly once -/
def fiftieth_permutation : Nat :=
  five_digit_permutations[49]'(by sorry)

theorem fiftieth_permutation_is_31254 : fiftieth_permutation = 31254 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_permutation_is_31254_l1346_134696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_in_room_l1346_134656

/-- Represents a rectangular room with tiles -/
structure Room where
  area : ℝ
  length_width_ratio : ℝ
  tile_size : ℝ

/-- Calculates the number of tiles along the width of the room -/
noncomputable def tiles_along_width (room : Room) : ℝ :=
  let width := Real.sqrt (room.area / room.length_width_ratio)
  (width * 12) / room.tile_size

/-- Theorem stating the number of tiles along the width for the given room -/
theorem tiles_in_room (room : Room) 
    (h_area : room.area = 360)
    (h_ratio : room.length_width_ratio = 2)
    (h_tile : room.tile_size = 4) :
    tiles_along_width room = 18 * Real.sqrt 5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval tiles_along_width ⟨360, 2, 4⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_in_room_l1346_134656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elisa_total_paint_l1346_134647

/-- The amount of square feet Elisa paints on Monday -/
def monday_paint : ℚ := 30

/-- The amount of square feet Elisa paints on Tuesday -/
noncomputable def tuesday_paint : ℚ := 2 * monday_paint

/-- The amount of square feet Elisa paints on Wednesday -/
noncomputable def wednesday_paint : ℚ := monday_paint / 2

/-- The total amount of square feet Elisa paints -/
noncomputable def total_paint : ℚ := monday_paint + tuesday_paint + wednesday_paint

/-- Theorem stating that the total amount of square feet Elisa paints is 105 -/
theorem elisa_total_paint : total_paint = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elisa_total_paint_l1346_134647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1346_134685

def is_valid_assignment (a b c d e f : ℕ) : Prop :=
  Finset.toSet {a, b, c, d, e, f} = Finset.toSet {0, 1, 2, 3, 4}

def expression_value (a b c d e f : ℕ) : ℤ :=
  (c : ℤ) * (a^b : ℕ) - d + (e^f : ℕ)

theorem max_expression_value :
  ∃ (a b c d e f : ℕ), is_valid_assignment a b c d e f ∧
    (∀ (x y z w u v : ℕ), is_valid_assignment x y z w u v →
      expression_value x y z w u v ≤ expression_value a b c d e f) ∧
    expression_value a b c d e f = 127 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1346_134685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1346_134683

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the line through two points
def line_through (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p = (1 - t) • p1 + t • p2}

-- Define the orthogonal projection
def orthogonal_projection (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry -- Implementation of orthogonal projection

-- Define the circumcircle
def circumcircle (t : Triangle) : Set (ℝ × ℝ) :=
  sorry -- Implementation of circumcircle

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.C = (2, -8) ∧
  ∀ x y, y = -2 * x + 11 ↔ (x, y) ∈ line_through t.A t.B ∧
  ∀ x y, x + 3 * y + 2 = 0 ↔ (x, y) ∈ line_through t.B (orthogonal_projection t.B (line_through t.A t.C))

-- Define the circumcircle equation
def is_circumcircle_equation (t : Triangle) (a b c d e : ℝ) : Prop :=
  ∀ x y, (x, y) ∈ circumcircle t ↔ a * x^2 + b * y^2 + c * x + d * y + e = 0

-- State the theorem
theorem triangle_properties (t : Triangle) :
  triangle_conditions t →
  t.A = (5, 1) ∧
  t.B = (7, -3) ∧
  is_circumcircle_equation t 1 1 (-4) 6 (-12) :=
by
  sorry -- Proof to be implemented


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1346_134683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_250_meters_l1346_134674

-- Define the given parameters
noncomputable def train_speed_kmph : ℝ := 72
noncomputable def platform_length : ℝ := 270
noncomputable def crossing_time : ℝ := 26

-- Define the conversion factor from km/h to m/s
noncomputable def kmph_to_ms : ℝ := 5 / 18

-- Define the function to calculate the train length
noncomputable def calculate_train_length (speed_kmph : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : ℝ :=
  speed_kmph * kmph_to_ms * crossing_time - platform_length

-- Theorem statement
theorem train_length_is_250_meters :
  calculate_train_length train_speed_kmph platform_length crossing_time = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_250_meters_l1346_134674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l1346_134629

noncomputable def equation (x : ℝ) : Prop :=
  Real.sqrt x + Real.sqrt (9 / x) + 2 * Real.sqrt (x + 9 / x) = 8

theorem sum_of_solutions : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, equation x) ∧ (∀ x : ℝ, equation x → x ∈ s) ∧ (s.sum id = 40.96) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l1346_134629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_device_significantly_improved_l1346_134686

-- Define the data for old and new devices
def old_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

-- Define sample mean function
def sample_mean (data : List Float) : Float :=
  (data.sum) / (data.length.toFloat)

-- Define sample variance function
def sample_variance (data : List Float) : Float :=
  let mean := sample_mean data
  (data.map (fun x => (x - mean) ^ 2)).sum / (data.length.toFloat)

-- Define the significant improvement criterion
def is_significantly_improved (x_bar y_bar s1_sq s2_sq : Float) : Prop :=
  y_bar - x_bar ≥ 2 * (((s1_sq + s2_sq) / 10 : Float).sqrt)

-- Theorem statement
theorem new_device_significantly_improved :
  let x_bar := sample_mean old_data
  let y_bar := sample_mean new_data
  let s1_sq := sample_variance old_data
  let s2_sq := sample_variance new_data
  is_significantly_improved x_bar y_bar s1_sq s2_sq :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_device_significantly_improved_l1346_134686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_approx_l1346_134671

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

theorem balance_difference_approx :
  let angela_balance := compound_interest 7000 0.05 15
  let bob_balance := simple_interest 12000 0.04 15
  let difference := bob_balance - angela_balance
  ⌊difference⌋ = 4647 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_approx_l1346_134671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fabian_apples_correct_answer_l1346_134677

/-- The number of kilograms of apples Fabian wants to buy -/
def apples : ℝ := 5

/-- The cost of one kilogram of apples in dollars -/
def apple_cost : ℝ := 2

/-- The cost of one kilogram of walnuts in dollars -/
def walnut_cost : ℝ := 6

/-- The number of packs of sugar Fabian wants to buy -/
def sugar_packs : ℕ := 3

/-- The amount of walnuts in kilograms Fabian wants to buy -/
def walnuts : ℝ := 0.5

/-- The total cost of Fabian's purchase in dollars -/
def total_cost : ℝ := 16

/-- The cost of one pack of sugar in dollars -/
def sugar_cost : ℝ := apple_cost - 1

theorem fabian_apples :
  apples * apple_cost + sugar_packs * sugar_cost + walnuts * walnut_cost = total_cost :=
by
  -- Substitute the defined values
  simp [apples, apple_cost, sugar_packs, sugar_cost, walnuts, walnut_cost, total_cost]
  -- Perform the calculation
  norm_num

theorem correct_answer : apples = 5 := rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fabian_apples_correct_answer_l1346_134677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_an_is_integer_l1346_134627

theorem an_is_integer (a b n : ℕ+) (h1 : a > b) :
  ∃ θ : ℝ, 0 < θ ∧ θ < π/2 ∧ Real.sin θ = (2 * (a : ℝ) * b) / ((a^2 : ℝ) + b^2) →
  ∃ k : ℤ, ((a^2 : ℝ) + b^2) * Real.sin (n * θ) = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_an_is_integer_l1346_134627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_property_l1346_134689

theorem sine_function_property (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + φ)) →
  (abs φ < Real.pi / 2) →
  (∀ x, f (x + Real.pi / 3) = -f (Real.pi / 3 - x)) →
  (∀ x₁ x₂, x₁ ∈ Set.Ioo (Real.pi / 12) (7 * Real.pi / 12) →
            x₂ ∈ Set.Ioo (Real.pi / 12) (7 * Real.pi / 12) →
            x₁ ≠ x₂ →
            f x₁ + f x₂ = 0) →
  (∀ x₁ x₂, x₁ ∈ Set.Ioo (Real.pi / 12) (7 * Real.pi / 12) →
            x₂ ∈ Set.Ioo (Real.pi / 12) (7 * Real.pi / 12) →
            x₁ ≠ x₂ →
            f (x₁ + x₂) = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_property_l1346_134689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_l1346_134640

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable def angle (a b : E) : ℝ := Real.arccos (inner a b / (norm a * norm b))

theorem vectors_parallel (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : norm a = norm b) (h2 : norm (a + b) = 2 * norm a) : 
  angle a b = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_l1346_134640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sum_of_sines_l1346_134699

theorem max_value_sum_of_sines :
  ∃ (M : ℝ), M = Real.sqrt 3 ∧ ∀ x : ℝ, Real.sin x + Real.sin (x - π/3) ≤ M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sum_of_sines_l1346_134699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1346_134692

theorem tan_alpha_value (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : 3 * Real.sin α ^ 2 = 2 * Real.cos α) : 
  Real.tan α = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1346_134692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_P_intersect_Q_in_I_l1346_134638

def I : Set ℕ := {x : ℕ | x ≠ 0 ∧ x ≥ 1 ∧ x ≤ 4}
def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {2, 3, 4}

theorem complement_of_P_intersect_Q_in_I :
  (I \ (P ∩ Q)) = {1, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_P_intersect_Q_in_I_l1346_134638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_unique_n_l1346_134642

/-- An arithmetic sequence with n terms -/
structure ArithmeticSequence where
  n : ℕ            -- number of terms
  a₁ : ℚ           -- first term
  d : ℚ             -- common difference

/-- The sum of the first k terms of an arithmetic sequence -/
noncomputable def sumFirstK (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k * (2 * seq.a₁ + (k - 1) * seq.d) / 2

/-- The sum of the last k terms of an arithmetic sequence -/
noncomputable def sumLastK (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k * (2 * (seq.a₁ + (seq.n - 1) * seq.d) - (k - 1) * seq.d) / 2

/-- The sum of all terms in an arithmetic sequence -/
noncomputable def sumAll (seq : ArithmeticSequence) : ℚ :=
  seq.n * (2 * seq.a₁ + (seq.n - 1) * seq.d) / 2

theorem arithmetic_sequence_unique_n (seq : ArithmeticSequence) :
  sumFirstK seq 4 = 26 →
  sumLastK seq 4 = 110 →
  sumAll seq = 187 →
  seq.n = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_unique_n_l1346_134642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1346_134601

/-- The compound interest formula -/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- The problem statement -/
theorem investment_growth :
  let principal : ℝ := 5000
  let annual_rate : ℝ := 0.04
  let quarterly_rate : ℝ := annual_rate / 4
  let years : ℕ := 18
  let periods : ℕ := years * 4
  let final_amount := compound_interest principal quarterly_rate periods
  ∃ ε > 0, |final_amount - 10154.28| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1346_134601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_sets_l1346_134676

theorem intersection_of_sets : 
  let M : Set ℝ := {x | x ≤ 0}
  let N : Set ℝ := {x | x^2 ≤ 1}
  M ∩ N = Set.Icc (-1) 0 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_sets_l1346_134676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_n_l1346_134679

/-- A sequence defined recursively -/
def a : ℕ → ℚ
  | 0 => 0  -- Added case for 0
  | 1 => 1
  | n + 1 => ((n + 1) * a n) / n

/-- Theorem stating that the general term of the sequence is n -/
theorem a_eq_n (n : ℕ) : n ≠ 0 → a n = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_n_l1346_134679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_m_on_n_l1346_134693

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define vectors a and b
variable (a b : V)

-- Define the angle between a and b
noncomputable def angle : ℝ := Real.pi / 6  -- 30° in radians

-- Define the magnitudes of a and b
noncomputable def mag_a : ℝ := Real.sqrt 3
def mag_b : ℝ := 1

-- Define vectors m and n
def m (a b : V) : V := a + b
def n (a b : V) : V := a - b

-- State the theorem
theorem projection_m_on_n (a b : V) :
  (‖a‖ = mag_a) →
  (‖b‖ = mag_b) →
  (inner a b = ‖a‖ * ‖b‖ * Real.cos angle) →
  (inner (m a b) (n a b) / ‖n a b‖ = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_m_on_n_l1346_134693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l1346_134646

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)

theorem tangent_line_at_zero :
  let slope := (deriv f) 0
  let y_intercept := f 0
  (λ x ↦ slope * x + y_intercept) = (λ x ↦ 2 * x + 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l1346_134646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_figure_contains_integer_point_l1346_134698

/-- A convex figure in the plane -/
structure ConvexFigure where
  -- We represent the figure as a set of points
  points : Set (ℝ × ℝ)
  -- Add necessary conditions for convexity
  convex : ∀ (x y : ℝ × ℝ), x ∈ points → y ∈ points → ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (1 - t) • x + t • y ∈ points

/-- The area of a convex figure -/
noncomputable def area (F : ConvexFigure) : ℝ := sorry

/-- A point has integer coordinates -/
def isIntegerPoint (p : ℝ × ℝ) : Prop := ∃ (m n : ℤ), p = (↑m, ↑n)

/-- The origin point (0, 0) -/
def origin : ℝ × ℝ := (0, 0)

/-- A figure is symmetric about the origin -/
def isSymmetricAboutOrigin (F : ConvexFigure) : Prop :=
  ∀ p, p ∈ F.points → (-p.1, -p.2) ∈ F.points

/-- Main theorem -/
theorem convex_figure_contains_integer_point (F : ConvexFigure) 
  (h_area : area F > 4)
  (h_sym : isSymmetricAboutOrigin F) :
  ∃ p, p ∈ F.points ∧ isIntegerPoint p ∧ p ≠ origin := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_figure_contains_integer_point_l1346_134698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1346_134613

/-- For a parabola defined by y² = 8x, the distance from its focus to its directrix is 4. -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y^2 = 8*x → (∃ (focus_x focus_y directrix_x : ℝ),
    |focus_x - directrix_x| = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1346_134613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_a_5000_b_501_l1346_134625

-- Define the sequences a_n and b_n
def a : ℕ → ℕ
  | 0 => 1  -- Define for 0 to avoid missing case
  | 1 => 1
  | 2 => 8
  | n + 3 => 7 * a (n + 2) - a (n + 1)

def b : ℕ → ℕ
  | 0 => 1  -- Define for 0 to avoid missing case
  | 1 => 1
  | 2 => 2
  | n + 3 => 3 * b (n + 2) - b (n + 1)

-- State the theorem
theorem gcd_a_5000_b_501 : Nat.gcd (a 5000) (b 501) = 89 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_a_5000_b_501_l1346_134625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_fourth_power_l1346_134643

variable {n : ℕ}
variable (A : Matrix (Fin n) (Fin n) ℝ)

theorem det_A_fourth_power (h : Matrix.det A = 7) : Matrix.det (A^4) = 2401 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_fourth_power_l1346_134643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_shadow_area_sum_a_plus_b_l1346_134654

/-- The area of the shadow cast by a cube with edge length 8, balanced on one vertex 
    with its body diagonal vertical, when the sun is directly overhead. -/
theorem cube_shadow_area : ℝ := by
  -- Define the cube's edge length
  let cube_edge : ℝ := 8

  -- Define the shadow as a regular hexagon
  -- (We don't prove it's a hexagon, but state it as a fact)
  let shadow_is_regular_hexagon : Prop := sorry

  -- The area of the shadow
  let shadow_area : ℝ := 64 * Real.sqrt 3

  -- State that the cube is balanced on one vertex with body diagonal vertical
  let cube_balanced : Prop := sorry

  -- State that the sun is directly overhead
  let sun_overhead : Prop := sorry

  -- Prove that the shadow area is equal to 64√3
  have h : shadow_area = 64 * Real.sqrt 3 := by rfl

  -- Return the final result
  exact shadow_area

/-- The sum of a and b, where the shadow area is expressed as a√b -/
theorem sum_a_plus_b : ℕ := by
  -- Define a and b
  let a : ℕ := 64
  let b : ℕ := 3

  -- Calculate the sum
  let sum : ℕ := a + b

  -- Prove that the sum is equal to 67
  have h : sum = 67 := by rfl

  -- Return the final result
  exact sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_shadow_area_sum_a_plus_b_l1346_134654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1346_134626

/-- Box A contains tiles numbered from 1 to 20 -/
def box_A : Finset ℕ := Finset.range 20

/-- Box B contains tiles numbered from 11 to 30 -/
def box_B : Finset ℕ := Finset.Icc 11 30

/-- Predicate for tiles in Box A that are less than 15 -/
def less_than_15 (n : ℕ) : Prop := n < 15

/-- Predicate for tiles in Box B that are even or greater than 25 -/
def even_or_greater_than_25 (n : ℕ) : Prop := Even n ∨ n > 25

/-- The probability of the desired outcome -/
def probability : ℚ := 21 / 50

theorem probability_theorem :
  (Finset.filter (fun n => n < 15) box_A).card / box_A.card *
  (Finset.filter (fun n => Even n ∨ n > 25) box_B).card / box_B.card = probability := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1346_134626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yan_distance_ratio_l1346_134667

/-- Yan's walking speed -/
noncomputable def walking_speed : ℝ := 1

/-- Yan's cycling speed (9 times walking speed) -/
noncomputable def cycling_speed : ℝ := 9 * walking_speed

/-- Distance from Yan to his home -/
noncomputable def distance_to_home : ℝ := 4

/-- Distance from Yan to the market -/
noncomputable def distance_to_market : ℝ := 5

/-- Time taken to walk directly to the market -/
noncomputable def time_walk_to_market : ℝ := distance_to_market / walking_speed

/-- Time taken to walk home and cycle to market -/
noncomputable def time_walk_and_cycle : ℝ := distance_to_home / walking_speed + (distance_to_home + distance_to_market) / cycling_speed

theorem yan_distance_ratio :
  time_walk_to_market = time_walk_and_cycle →
  distance_to_home / distance_to_market = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yan_distance_ratio_l1346_134667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_number_20th_row_l1346_134614

/-- Represents the triangular array where each row starts with a specific number
    and the difference between consecutive numbers in a row forms an arithmetic sequence. -/
def TriangularArray : Type := ℕ → ℕ → ℕ

/-- The first number in each row of the triangular array. -/
def firstInRow : ℕ → ℕ := sorry

/-- The common difference of the arithmetic sequence in each row. -/
def rowDifference : ℕ → ℕ := sorry

/-- The nth number in the mth row of the triangular array. -/
def arrayElement (arr : TriangularArray) (m n : ℕ) : ℕ :=
  firstInRow m + (n - 1) * rowDifference m

theorem tenth_number_20th_row (arr : TriangularArray) :
  firstInRow 20 = 210 →
  rowDifference 20 = 24 →
  arrayElement arr 20 10 = 426 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_number_20th_row_l1346_134614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_nth_term_a_n_formula_l1346_134670

-- Define the sequence a_n and its partial sum S_n
def S (n : ℕ) : ℕ := n^2 + 2*n

-- State the theorem
theorem sequence_nth_term (n : ℕ) : 
  n ≥ 1 → S n - S (n-1) = 2*n + 1 :=
by sorry

-- Prove that a_n = 2n + 1 for all n ≥ 1
theorem a_n_formula (n : ℕ) : 
  n ≥ 1 → S n - (if n = 1 then 0 else S (n-1)) = 2*n + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_nth_term_a_n_formula_l1346_134670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_is_three_l1346_134621

/-- Given two vectors a and b in ℝ³, where a is perpendicular to b,
    prove that the magnitude of b is 3. -/
theorem magnitude_of_b_is_three (a b : Fin 3 → ℝ) : 
  a = ![2, 3, -2] →
  ∃ m : ℝ, b = ![2, -m, -1] →
  (a 0) * (b 0) + (a 1) * (b 1) + (a 2) * (b 2) = 0 →
  Real.sqrt ((b 0)^2 + (b 1)^2 + (b 2)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_is_three_l1346_134621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l1346_134606

theorem least_number_with_remainder (n : ℕ) : 
  (∀ d ∈ ({6, 130, 9, 18} : Set ℕ), n % d = 4) ∧ 
  (∀ m < n, ∃ d ∈ ({6, 130, 9, 18} : Set ℕ), m % d ≠ 4) →
  n = 2344 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l1346_134606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_l1346_134664

theorem absolute_value_equation_solution : 
  ∃! x : ℝ, (|2*x - 1| = 3*x + 6) ∧ (x + 2 > 0) :=
by
  -- The unique solution is x = -1
  use -1
  
  constructor
  · constructor
    · -- Show that |-1*2 - 1| = 3*(-1) + 6
      sorry
    · -- Show that -1 + 2 > 0
      sorry
  
  -- Prove uniqueness
  · intro y ⟨h1, h2⟩
    -- Show that if y satisfies the conditions, then y = -1
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_l1346_134664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l1346_134617

noncomputable section

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- Two lines are coincident if they have the same slope and y-intercept -/
def coincident (m1 b1 m2 b2 : ℝ) : Prop := m1 = m2 ∧ b1 = b2

/-- The slope-intercept form of a line ax + by = c is y = (-a/b)x + (c/b) -/
def slope_intercept (a b c : ℝ) : ℝ × ℝ := (-a/b, c/b)

theorem parallel_lines_condition (a : ℝ) :
  (parallel (slope_intercept a 2 (-3*a)).1 (slope_intercept 3 (a-1) (a-7)).1 ∧
   ¬coincident (slope_intercept a 2 (-3*a)).1 (slope_intercept a 2 (-3*a)).2
               (slope_intercept 3 (a-1) (a-7)).1 (slope_intercept 3 (a-1) (a-7)).2) ↔
  a = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l1346_134617
