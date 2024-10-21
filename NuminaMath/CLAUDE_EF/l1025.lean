import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1025_102581

noncomputable def f (x : ℝ) := Real.sqrt (4 - Real.sqrt (7 - Real.sqrt x))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | 0 ≤ x ∧ x ≤ 49} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1025_102581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grisha_win_probability_expected_flips_l1025_102535

/-- Represents the outcome of a single coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the result of the game -/
inductive GameResult
| GrishaWins
| VanyaWins

/-- The coin flipping rule as described in the problem -/
def coinFlipRule (flips : List CoinFlip) : Option GameResult :=
  match flips with
  | [] => none
  | (CoinFlip.Heads :: rest) => 
      if rest.length % 2 == 0 then some GameResult.GrishaWins else coinFlipRule rest
  | (CoinFlip.Tails :: rest) => 
      if rest.length % 2 == 1 then some GameResult.VanyaWins else coinFlipRule rest

/-- The probability of Grisha winning -/
noncomputable def grishaWinProbability : ℝ := 1/3

/-- The expected number of coin flips until the outcome is decided -/
noncomputable def expectedFlips : ℝ := 2

/-- Theorem stating that the probability of Grisha winning is 1/3 -/
theorem grisha_win_probability :
  grishaWinProbability = 1/3 := by sorry

/-- Theorem stating that the expected number of coin flips is 2 -/
theorem expected_flips :
  expectedFlips = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grisha_win_probability_expected_flips_l1025_102535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_ratio_l1025_102591

/-- The ratio of the area of the inscribed circle to the area of a right triangle -/
theorem inscribed_circle_area_ratio (b h : ℝ) (h1 : 0 < b) (h2 : 0 < h) : 
  let c := Real.sqrt (b^2 + h^2)
  let r := (b * h) / (b + h + c)
  let circle_area := π * r^2
  let triangle_area := (1/2) * b * h
  circle_area / triangle_area = (2 * π * b^2 * h^2) / ((b + h + c)^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_ratio_l1025_102591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l1025_102596

/-- The area of a circular sector with central angle α and radius R -/
noncomputable def sectorArea (α : ℝ) (R : ℝ) : ℝ := (1/2) * α * R^2

/-- The perimeter of a circular sector with central angle α and radius R -/
noncomputable def sectorPerimeter (α : ℝ) (R : ℝ) : ℝ := 2 * R + α * R

theorem sector_max_area (c : ℝ) (h : c > 0) :
  ∃ (α : ℝ) (R : ℝ),
    sectorPerimeter α R = c ∧
    (∀ (α' : ℝ) (R' : ℝ),
      sectorPerimeter α' R' = c →
      sectorArea α R ≥ sectorArea α' R') ∧
    α = 2 ∧
    sectorArea α R = c^2 / 16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l1025_102596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_regular_pentagon_l1025_102523

/-- A regular pentagon inscribed in a circle -/
structure RegularPentagonInCircle where
  /-- The side length of the pentagon -/
  side_length : ℝ
  /-- The side length is positive -/
  side_length_pos : side_length > 0

/-- The arc length intercepted by one side of the pentagon -/
noncomputable def arc_length (p : RegularPentagonInCircle) : ℝ :=
  2 * Real.pi

theorem arc_length_of_regular_pentagon (p : RegularPentagonInCircle) 
  (h : p.side_length = 5) : arc_length p = 2 * Real.pi := by
  -- Unfold the definition of arc_length
  unfold arc_length
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_regular_pentagon_l1025_102523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_april_canoes_l1025_102500

def canoes_per_month (n : ℕ) : ℕ :=
  5 * 3^n

def total_canoes (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) canoes_per_month

theorem april_canoes :
  total_canoes 4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_april_canoes_l1025_102500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l1025_102592

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The focus of the hyperbola -/
def focus : ℝ × ℝ := (5, 0)

/-- The asymptote equation -/
def asymptote (x y : ℝ) : Prop := 3 * x + 4 * y = 0

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  let (x, y) := p
  abs (A * x + B * y + C) / Real.sqrt (A^2 + B^2)

theorem distance_focus_to_asymptote :
  distance_point_to_line focus 3 4 0 = 3 :=
by
  -- Unfold definitions and simplify
  simp [distance_point_to_line, focus]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l1025_102592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1025_102506

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (f 0 = -1) ∧
  (∀ x : ℝ, f x ≤ 2) ∧
  (∀ k : ℤ, f (k * Real.pi + Real.pi / 3) = 2) ∧
  (∀ k : ℤ, ∀ x : ℝ, k * Real.pi - Real.pi / 4 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12 →
    ∀ y : ℝ, k * Real.pi - Real.pi / 4 ≤ y ∧ y ≤ x → f y ≤ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1025_102506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_amount_approx_l1025_102557

/-- The price of rice per pound -/
noncomputable def rice_price : ℝ := 1.10

/-- The price of peas per pound -/
noncomputable def peas_price : ℝ := 0.55

/-- The total amount of rice and peas bought in pounds -/
noncomputable def total_amount : ℝ := 30

/-- The total cost of the purchase -/
noncomputable def total_cost : ℝ := 23.50

/-- The amount of rice bought -/
noncomputable def rice_amount : ℝ := (total_cost - peas_price * total_amount) / (rice_price - peas_price)

/-- Theorem stating that the rice amount is approximately 12.7 pounds -/
theorem rice_amount_approx : 
  ∀ ε > 0, |rice_amount - 12.7| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_amount_approx_l1025_102557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_function_l1025_102518

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_a_for_decreasing_function 
  (f : ℝ → ℝ) 
  (h_decreasing : DecreasingFunction f) 
  (h_inequality : ∀ a, f (1 - a) < f (2 * a - 1)) :
  Set.range (λ a : ℝ => a) = Set.Iio 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_function_l1025_102518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangents_l1025_102579

-- Define the circle
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (3, 2)

-- Define the two potential tangent lines through P
def tangent_line1 (x y : ℝ) : Prop := 12*x - 5*y - 26 = 0
def tangent_line2 (x y : ℝ) : Prop := y - 2 = 0

theorem circle_and_tangents :
  (∃ x y : ℝ, circle_O x y ∧ tangent_line x y) ∧
  (∀ x y : ℝ, circle_O x y ∧ (tangent_line1 x y ∨ tangent_line2 x y) → 
    ((x - point_P.1)^2 + (y - point_P.2)^2)^2 = 
    16 * ((x - point_P.1)^2 + (y - point_P.2)^2 - 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangents_l1025_102579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_boat_speed_l1025_102511

/-- Two boats traveling towards each other -/
structure BoatProblem where
  boat1_speed : ℝ
  boat2_speed : ℝ
  initial_distance : ℝ
  distance_before_collision : ℝ
  time_before_collision : ℝ

/-- The conditions of the problem -/
noncomputable def problem_conditions : BoatProblem where
  boat1_speed := 25
  boat2_speed := 5  -- This is not given directly, but we need to define it for Lean
  initial_distance := 20
  distance_before_collision := 0.5
  time_before_collision := 1/60  -- 1 minute in hours

/-- Theorem stating that the speed of the first boat is 25 miles/hr -/
theorem first_boat_speed (p : BoatProblem) 
  (h1 : p.boat1_speed = problem_conditions.boat1_speed)
  (h2 : p.initial_distance = problem_conditions.initial_distance)
  (h3 : p.distance_before_collision = problem_conditions.distance_before_collision)
  (h4 : p.time_before_collision = problem_conditions.time_before_collision) :
  p.boat1_speed = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_boat_speed_l1025_102511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qin_jiushao_v4_l1025_102526

def qin_jiushao (coeffs : List ℝ) (x : ℝ) : List ℝ :=
  match coeffs with
  | [] => []
  | head :: tail => tail.scanl (fun acc a => acc * x + a) head

def polynomial : List ℝ := [7, 6, 5, 4, 3, 2, 1]

theorem qin_jiushao_v4 :
  (qin_jiushao polynomial 3).get! 4 = 789 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_qin_jiushao_v4_l1025_102526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_money_value_l1025_102561

/-- The amount of money Mr. Black has -/
noncomputable def black_money : ℝ := 75

/-- The amount of money Mr. Green has -/
noncomputable def green_money : ℝ := black_money / 4

/-- The amount of money Mr. White has -/
noncomputable def white_money : ℝ := green_money * 1.2

/-- Theorem stating that Mr. White has $22.50 -/
theorem white_money_value : white_money = 22.5 := by
  -- Unfold the definitions
  unfold white_money green_money black_money
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_money_value_l1025_102561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1025_102532

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.c * Real.cos t.B + t.b * Real.cos t.C = 3 * t.a * Real.cos t.B)
  (h2 : t.a * t.c * Real.cos t.B = 2) : 
  Real.cos t.B = 1/3 ∧ t.b ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1025_102532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ngon_concurrent_lines_l1025_102515

-- Define the regular n-gon and related points
def regular_ngon (n : ℕ) : Set (ℝ × ℝ) := sorry

-- Define the points Xᵢ
def X (i : ℕ) (n : ℕ) : ℝ × ℝ := sorry

-- Define circle Γ
def Γ : Set (ℝ × ℝ) := sorry

-- Define points Tᵢ and Sᵢ
def T (i : ℕ) (n : ℕ) : ℝ × ℝ := sorry
def S (i : ℕ) (n : ℕ) : ℝ × ℝ := sorry

-- Define the property of being concurrent
def concurrent (lines : List (Set (ℝ × ℝ))) : Prop := sorry

-- Define a line connecting two points
def connect (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem ngon_concurrent_lines (n : ℕ) :
  let A := regular_ngon n
  let x_lines := List.range n |>.map (λ i => connect (X i n) (T i n))
  let s_lines := List.range n |>.map (λ i => connect (X i n) (S i n))
  (∀ i, X i n ∈ Γ) →
  concurrent x_lines ∧ concurrent s_lines := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ngon_concurrent_lines_l1025_102515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_theorem_l1025_102563

/-- Given a function f(x) = aˣ + a⁻ˣ where a > 0 and a ≠ 1, if f(1) = 3, then f(0) + f(1) + f(2) = 12 -/
theorem function_sum_theorem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  let f := fun x : ℝ => a^x + a^(-x)
  f 1 = 3 → f 0 + f 1 + f 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_theorem_l1025_102563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l1025_102538

/-- Two circles touching externally at point D -/
structure ExternallyTouchingCircles where
  R : ℝ  -- radius of the first circle
  r : ℝ  -- radius of the second circle
  h : R > 0 ∧ r > 0  -- both radii are positive

/-- External tangent to two circles -/
noncomputable def ExternalTangent (c : ExternallyTouchingCircles) :=
  { AB : ℝ // AB = 2 * Real.sqrt (c.R * c.r) }

/-- Volume of solid generated by rotating triangle ABD -/
noncomputable def TriangleVolume (c : ExternallyTouchingCircles) (t : ExternalTangent c) : ℝ :=
  (8 * Real.pi * c.R^2 * c.r^2) / (3 * (c.R + c.r))

/-- Volume of solid generated by rotating the region bounded by arcs AD, BD, and line AB -/
noncomputable def ArcRegionVolume (c : ExternallyTouchingCircles) (t : ExternalTangent c) : ℝ :=
  (4 * Real.pi * c.R^2 * c.r^2) / (3 * (c.R + c.r))

/-- The main theorem stating the ratio of volumes -/
theorem volume_ratio (c : ExternallyTouchingCircles) (t : ExternalTangent c) :
  TriangleVolume c t / ArcRegionVolume c t = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l1025_102538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_product_900_l1025_102560

/-- A function that returns the digits of a positive integer -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- A function that returns true if a number is a five-digit positive integer -/
def isFiveDigitPositive (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n ≤ 99999

/-- A function that returns the product of a list of natural numbers -/
def productOfList : List ℕ → ℕ
  | [] => 1
  | (h :: t) => h * productOfList t

/-- The count of five-digit positive integers with digit product 900 -/
def countFiveDigitProduct900 : ℕ :=
  let validNumbers := List.range 90000
    |>.map (· + 10000)
    |>.filter (fun n => productOfList (digits n) = 900)
  validNumbers.length

theorem count_five_digit_product_900 :
  countFiveDigitProduct900 = 210 := by
  sorry

#eval countFiveDigitProduct900

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_product_900_l1025_102560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_overall_percentage_l1025_102508

/-- Calculates the overall percentage given three subject percentages -/
noncomputable def overallPercentage (subject1 : ℝ) (subject2 : ℝ) (subject3 : ℝ) : ℝ :=
  (subject1 + subject2 + subject3) / 3

/-- Theorem stating that given percentages of 60, 80, and 85 for three subjects,
    the overall percentage is 75 -/
theorem student_overall_percentage :
  overallPercentage 60 80 85 = 75 := by
  -- Unfold the definition of overallPercentage
  unfold overallPercentage
  -- Simplify the arithmetic
  simp [add_div]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_overall_percentage_l1025_102508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_quadratic_l1025_102572

def is_lattice_point (p : ℤ × ℤ) : Prop := true

def quadratic_function (x : ℤ) : ℚ :=
  (x^2 : ℚ) / 10 - (x : ℚ) / 10 + 9 / 5

def satisfies_condition (p : ℤ × ℤ) : Prop :=
  is_lattice_point p ∧ 
  (p.2 : ℚ) = quadratic_function p.1 ∧
  (p.2 : ℤ) ≤ p.1.natAbs

theorem lattice_points_on_quadratic : 
  {p : ℤ × ℤ | satisfies_condition p} = 
  {(2,2), (4,3), (7,6), (9,9), (-6,6), (-3,3)} := by sorry

#check lattice_points_on_quadratic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_quadratic_l1025_102572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_test_max_missable_l1025_102550

/-- Calculates the maximum number of problems that can be missed while still passing a test -/
def max_missable_problems (total_problems : ℕ) (passing_percentage : ℚ) : ℕ :=
  (((1 : ℚ) - passing_percentage) * total_problems).floor.toNat

/-- Theorem stating the maximum number of missable problems for the specific test conditions -/
theorem geometry_test_max_missable :
  max_missable_problems 50 (3/4) = 12 := by
  -- Unfold the definition of max_missable_problems
  unfold max_missable_problems
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_test_max_missable_l1025_102550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_fixed_point_l1025_102501

/-- A parabola with vertex at the origin and directrix x = -1 -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  vertex_at_origin : equation 0 0
  directrix : ℝ → Prop
  directrix_def : ∀ x, directrix x ↔ x = -1

/-- A line intersecting the parabola at two distinct points -/
structure IntersectingLine (p : Parabola) where
  line : ℝ → ℝ → Prop
  point_a : ℝ × ℝ
  point_b : ℝ × ℝ
  distinct : point_a ≠ point_b
  on_parabola_a : p.equation point_a.1 point_a.2
  on_parabola_b : p.equation point_b.1 point_b.2
  on_line_a : line point_a.1 point_a.2
  on_line_b : line point_b.1 point_b.2

/-- The dot product condition -/
def dotProductCondition (p : Parabola) (l : IntersectingLine p) : Prop :=
  l.point_a.1 * l.point_b.1 + l.point_a.2 * l.point_b.2 = -4

theorem parabola_intersection_fixed_point (p : Parabola) (l : IntersectingLine p) 
  (h : dotProductCondition p l) : 
  ∃ (x y : ℝ), x = 2 ∧ y = 0 ∧ l.line x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_fixed_point_l1025_102501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_point_l1025_102553

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 3 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (3, -2)

/-- The given point -/
def given_point : ℝ × ℝ := (2, 5)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_from_center_to_point :
  distance circle_center given_point = Real.sqrt 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_point_l1025_102553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1025_102588

noncomputable def f (x : ℝ) := Real.log (3 * x + 1)

theorem domain_of_f :
  Set.range (fun x : ℝ => x) ∩ {x | f x ∈ Set.range Real.log} = Set.Ioi (-1/3 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1025_102588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_l1025_102533

theorem cube_root_equation (m n : ℕ) (h1 : m = 48) (h2 : n = 288) :
  ((m : ℝ) ^ (1/3) + (n : ℝ) ^ (1/3) - 1) ^ 2 = 49 + 20 * 6 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_l1025_102533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_fractions_l1025_102598

noncomputable def f1 (x : ℝ) : ℝ := 1 / (3 * x^2)
noncomputable def f2 (x : ℝ) : ℝ := 1 / (6 * x^3)
noncomputable def f3 (x : ℝ) : ℝ := 1 / (9 * x)
noncomputable def lcm_result (x : ℝ) : ℝ := 1 / (18 * x^3)

theorem lcm_of_fractions (x : ℝ) (hx : x ≠ 0) :
  ∃ (k1 k2 k3 : ℚ), k1 ≠ 0 ∧ k2 ≠ 0 ∧ k3 ≠ 0 ∧
  k1 * lcm_result x = f1 x ∧
  k2 * lcm_result x = f2 x ∧
  k3 * lcm_result x = f3 x ∧
  ∀ (y : ℝ), (∃ (m1 m2 m3 : ℚ), m1 ≠ 0 ∧ m2 ≠ 0 ∧ m3 ≠ 0 ∧
    m1 * y = f1 x ∧ m2 * y = f2 x ∧ m3 * y = f3 x) →
  ∃ (n : ℚ), n ≠ 0 ∧ n * lcm_result x = y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_fractions_l1025_102598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_decrease_l1025_102517

-- Define the original side length of the equilateral triangle
noncomputable def original_side : ℝ := Real.sqrt 400

-- Define the new side length after decreasing by 6 cm
noncomputable def new_side : ℝ := original_side - 6

-- Define the function to calculate the area of an equilateral triangle
noncomputable def area (side : ℝ) : ℝ := (side ^ 2 * Real.sqrt 3) / 4

-- State the theorem
theorem area_decrease :
  area original_side - area new_side = 51 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_decrease_l1025_102517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x15_is_166_l1025_102594

/-- The coefficient of x^15 in the expansion of (1 + x + x^2 + ... + x^20)(1 + x + x^2 + ... + x^10)^2 -/
def coefficient_x15 : ℕ :=
  (Finset.range 21).sum (fun i => (Finset.range 11).sum (fun j => (Finset.range 11).sum (fun k =>
    if i + j + k = 15 then 1 else 0)))

/-- The geometric series (1 + x + x^2 + ... + x^n) -/
noncomputable def geometric_sum (n : ℕ) : Polynomial ℚ :=
  (Finset.range (n + 1)).sum (fun i => Polynomial.X ^ i)

theorem coefficient_x15_is_166 :
  (geometric_sum 20 * (geometric_sum 10) ^ 2).coeff 15 = 166 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x15_is_166_l1025_102594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_close_in_unit_circle_l1025_102527

open Set

-- Define a type for points in a 2D plane
def Point := ℝ × ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define a unit circle
def unitCircle : Set Point :=
  {p : Point | p.1^2 + p.2^2 ≤ 1}

-- State the theorem
theorem two_points_close_in_unit_circle 
  (points : Finset Point) 
  (h_card : points.card = 6) 
  (h_in_circle : ∀ p, p ∈ points → p ∈ unitCircle) : 
  ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ distance p q ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_close_in_unit_circle_l1025_102527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_to_largest_volume_ratio_l1025_102529

/-- Represents a right circular cone sliced into four pieces of equal height -/
structure SlicedCone where
  height : ℝ
  baseRadius : ℝ

/-- Calculates the volume of a cone given its height and base radius -/
noncomputable def coneVolume (h : ℝ) (r : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Theorem: The ratio of the volume of the second-largest piece to the largest piece is 19/37 -/
theorem second_to_largest_volume_ratio (cone : SlicedCone) : 
  let v1 := coneVolume (4 * cone.height) (4 * cone.baseRadius) - 
            coneVolume (3 * cone.height) (3 * cone.baseRadius)
  let v2 := coneVolume (3 * cone.height) (3 * cone.baseRadius) - 
            coneVolume (2 * cone.height) (2 * cone.baseRadius)
  v2 / v1 = 19 / 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_to_largest_volume_ratio_l1025_102529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_of_0_18_l1025_102582

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define 'a' and 'b' as real numbers
variable (a b : ℝ)

-- Define the given conditions
axiom a_def : lg 2 = a
axiom b_def : lg 3 = b

-- State the theorem
theorem lg_of_0_18 : lg 0.18 = a + 2 * b - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_of_0_18_l1025_102582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rounds_is_four_l1025_102585

/-- Represents the distribution of rounds played by golfers -/
structure RoundDistribution where
  one_round : Nat
  two_rounds : Nat
  four_rounds : Nat
  five_rounds : Nat
  six_rounds : Nat

/-- Calculates the average number of rounds played and rounds to the nearest integer -/
def roundedAverageRounds (dist : RoundDistribution) : Nat :=
  let totalRounds := dist.one_round * 1 + dist.two_rounds * 2 + dist.four_rounds * 4 + 
                     dist.five_rounds * 5 + dist.six_rounds * 6
  let totalGolfers := dist.one_round + dist.two_rounds + dist.four_rounds + 
                      dist.five_rounds + dist.six_rounds
  let average : Rat := (totalRounds : Rat) / (totalGolfers : Rat)
  (average + 1/2).floor.toNat

theorem average_rounds_is_four (dist : RoundDistribution) 
  (h1 : dist.one_round = 4)
  (h2 : dist.two_rounds = 3)
  (h3 : dist.four_rounds = 4)
  (h4 : dist.five_rounds = 2)
  (h5 : dist.six_rounds = 6) :
  roundedAverageRounds dist = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rounds_is_four_l1025_102585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1025_102552

/-- The set of digits to be used -/
def digits : Finset Nat := {2, 4, 7, 9}

/-- A function that checks if a number is a valid three-digit integer formed from the given digits -/
def isValidNumber (n : Nat) : Bool :=
  100 ≤ n ∧ n < 1000 ∧ 
  (n / 100) ∈ digits ∧
  ((n / 10) % 10) ∈ digits ∧
  (n % 10) ∈ digits ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10)

/-- The set of all valid three-digit numbers formed from the given digits -/
def validNumbers : Finset Nat :=
  Finset.filter (fun n => isValidNumber n) (Finset.range 1000)

theorem count_valid_numbers : validNumbers.card = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1025_102552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_funny_number_l1025_102536

def is_funny (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 0 → d ∣ n → Nat.Prime (d + 2)

def num_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun x => x > 0 ∧ x ∣ n) (Finset.range (n + 1))).card

theorem largest_funny_number :
  ∃ (n : ℕ),
    n > 0 ∧
    is_funny n ∧
    (∀ m : ℕ, m > 0 → is_funny m → num_divisors m ≤ num_divisors n) ∧
    n = 135 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_funny_number_l1025_102536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_quantifier_l1025_102549

variable {α : Type*} [LinearOrder α] [Semiring α]

theorem negation_of_universal_quantifier (S : Set α) :
  (¬ ∀ x ∈ S, x^3 > 0) ↔ (∃ x ∈ S, x^3 ≤ 0) :=
by
  apply Iff.intro
  · intro h
    push_neg at h
    exact h
  · intro h
    push_neg
    exact h

#check negation_of_universal_quantifier

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_quantifier_l1025_102549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_with_hole_l1025_102548

theorem sphere_volume_with_hole (π : ℝ) (h : π > 0) : 
  (4/3 * π * (10^3 - 5^3) - π * 2^2 * 20) = (3260/3) * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_with_hole_l1025_102548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_carriages_l1025_102534

/-- Proves that a train with the given specifications has 24 carriages -/
theorem train_carriages : ∀ (n : ℕ),
  (let train_speed : ℝ := 60;
   let bridge_length : ℝ := 1.5;
   let crossing_time : ℝ := 3;
   let carriage_length : ℝ := 0.06;
   let train_length : ℝ := (n + 1) * carriage_length;
   let speed_km_min : ℝ := train_speed / 60;
   speed_km_min * crossing_time = bridge_length + train_length) →
  n = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_carriages_l1025_102534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_four_l1025_102545

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (|x - 2|) - m

-- Theorem statement
theorem sum_of_roots_is_four (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), f m x₁ = 0 ∧ f m x₂ = 0 ∧ x₁ + x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_four_l1025_102545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l1025_102571

noncomputable def f (x : ℝ) : ℝ := (x - 15*x^2 + 50*x^3 - 10*x^4) / (8 - 3*x^3)

theorem f_nonnegative_iff (x : ℝ) : f x ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio (2 * Real.rpow 3 (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l1025_102571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1025_102539

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - 13 * Real.pi / 4)

theorem f_properties :
  (f (Real.pi / 8) = 0) ∧
  (∀ x : ℝ, f x = 2 * Real.sin (2 * (x - 5 * Real.pi / 8))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1025_102539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_misha_is_lying_l1025_102562

/-- Represents the possible goal counts a player can claim -/
inductive GoalClaim
  | one
  | two
  | three
  | five

/-- Represents a player in the football match -/
structure Player where
  isTruthTeller : Bool
  goalClaim : GoalClaim

/-- Represents the football match -/
structure FootballMatch where
  players : Finset Player
  truthTellersScore : Nat
  liarsScore : Nat

/-- The sum of goals claimed by truth-tellers -/
def sumTruthTellersClaims (m : FootballMatch) : Nat :=
  m.players.sum fun p => 
    if p.isTruthTeller then
      match p.goalClaim with
      | GoalClaim.one => 1
      | GoalClaim.two => 2
      | GoalClaim.three => 3
      | GoalClaim.five => 5
    else 0

/-- Misha's claim -/
def mishasClaim : GoalClaim := GoalClaim.two

/-- Theorem stating that Misha is lying -/
theorem misha_is_lying (m : FootballMatch) 
  (h1 : m.players.card = 20)
  (h2 : (m.players.filter (fun p => p.isTruthTeller)).card = 10)
  (h3 : (m.players.filter (fun p => !p.isTruthTeller)).card = 10)
  (h4 : m.truthTellersScore = 20)
  (h5 : m.liarsScore = 17)
  (h6 : ∃ p ∈ m.players, p.goalClaim = mishasClaim) :
  ∃ p ∈ m.players, p.goalClaim = mishasClaim ∧ !p.isTruthTeller :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_misha_is_lying_l1025_102562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_max_and_sin_range_l1025_102509

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, 2 * Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (3 * Real.sin x + 4 * Real.cos x, -Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

def acute_triangle (A B C : ℝ) : Prop := 0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2 ∧ A + B + C = Real.pi

theorem vector_dot_product_max_and_sin_range 
  (A B C a b c : ℝ) 
  (h_triangle : acute_triangle A B C)
  (h_f : f (B/2 + Real.pi/4) = 4*c/a + 2) :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x ≤ f x_max ∧ f x_max = 4*Real.sqrt 2 + 2) ∧
  (∀ (y : ℝ), Real.sin B * Real.sin C = y → Real.sqrt 2 / 2 < y ∧ y ≤ (2 + Real.sqrt 2) / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_max_and_sin_range_l1025_102509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1025_102595

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x + 5

-- State the theorem
theorem f_max_min :
  ∃ (x_max x_min : ℝ),
    (0 ≤ x_max ∧ x_max ≤ 2) ∧
    (0 ≤ x_min ∧ x_min ≤ 2) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 → f x ≤ f x_max) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 → f x_min ≤ f x) ∧
    f x_max = 5/2 ∧
    f x_min = 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l1025_102595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_for_unit_radian_chord_l1025_102586

/-- Given a circle where the chord length corresponding to a central angle of 1 radian is 2,
    the arc length corresponding to this central angle is 1/sin(0.5). -/
theorem arc_length_for_unit_radian_chord (r : ℝ) : 
  (2 * r * Real.sin (1 / 2) = 2) → (r = 1 / Real.sin (1 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_for_unit_radian_chord_l1025_102586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1025_102567

-- Define the line
def line (a : ℝ) (x : ℝ) : ℝ := a * x + 1

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Theorem statement
theorem line_intersects_circle (a : ℝ) : 
  ∃ x y : ℝ, line a x = y ∧ circle_eq x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1025_102567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ratio_l1025_102568

/-- Curve C defined by parametric equations -/
noncomputable def C (α : ℝ) : ℝ × ℝ := (2 + Real.cos α, 3 + Real.sin α)

/-- Point A in polar coordinates -/
noncomputable def A : ℝ × ℝ := (3, Real.pi / 2)

/-- Distance from origin O to tangent point N on curve C -/
noncomputable def ON : ℝ := 2 * Real.sqrt 3

/-- Distance from point A to tangent point M on curve C -/
noncomputable def AM : ℝ := Real.sqrt 3

/-- Theorem stating the ratio of ON to AM is 2 -/
theorem tangent_ratio : ON / AM = 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ratio_l1025_102568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_equation_l1025_102544

/-- Given a circle with radius 1 and center at (1,0) in polar coordinates,
    its polar equation is ρ = 2cos θ -/
theorem circle_polar_equation (ρ θ : ℝ) :
  (ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2 = 1 ↔ ρ = 2 * Real.cos θ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_equation_l1025_102544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_given_medians_l1025_102541

/-- The area of a triangle given its three medians -/
noncomputable def triangle_area_from_medians (m₁ m₂ m₃ : ℝ) : ℝ :=
  let s := (m₁ + m₂ + m₃) / 2
  (4 / 3) * Real.sqrt (s * (s - m₁) * (s - m₂) * (s - m₃))

/-- Theorem: A triangle with medians 3, 4, and 5 has an area of 8 -/
theorem triangle_area_with_given_medians :
  triangle_area_from_medians 3 4 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_given_medians_l1025_102541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l1025_102569

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*Real.log x

-- Define the monotonicity property
def isMonotonic (a : ℝ) : Prop :=
  (a ≤ 0 → ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a > 0 → ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → x₂ < Real.sqrt a → f a x₁ > f a x₂) ∧
  (a > 0 → ∀ x₁ x₂, Real.sqrt a < x₁ → x₁ < x₂ → f a x₁ < f a x₂)

-- Define the minimum value function g(a)
noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ 1 then 1 else a - a * Real.log a

-- Define the property for unique solution when a > 0
def hasUniqueSolution (a : ℝ) : Prop :=
  a > 0 ∧ ∃! x, f a x = 2*a*x

-- Main theorem
theorem main_theorem (a : ℝ) :
  isMonotonic a ∧
  (∀ x, x ≥ 1 → f a x ≥ g a) ∧
  (hasUniqueSolution a ↔ a = 1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l1025_102569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannon_problem_l1025_102514

/-- The number of cannons -/
def n : ℕ := 10

/-- The probability of each cannon hitting the target -/
noncomputable def p : ℚ := 1/2

/-- The score for hitting the target -/
def score : ℕ := 2

/-- The probability of hitting the target exactly k times -/
noncomputable def prob_hit (k : ℕ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1-p)^(n-k)

/-- The expected value of the total score -/
noncomputable def expected_score : ℚ := n * p * score

theorem cannon_problem :
  prob_hit 3 = 15/128 ∧ expected_score = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannon_problem_l1025_102514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_square_area_l1025_102551

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The area of a square with side length equal to one-third of 20 cm, rounded to two decimal places, is 44.44 cm^2. -/
theorem small_square_area : 
  let original_side : ℝ := 20
  let small_side : ℝ := original_side / 3
  let exact_area : ℝ := small_side ^ 2
  round_to_hundredth exact_area = 44.44 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_square_area_l1025_102551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_neg_six_l1025_102528

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then x^3 - x else -((-x)^3 - (-x))

theorem f_neg_two_equals_neg_six :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x > 0, f x = x^3 - x) →  -- definition of f for x > 0
  f (-2) = -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_neg_six_l1025_102528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_equals_negative_one_l1025_102575

theorem sin_2alpha_equals_negative_one (α : ℝ) 
  (h1 : Real.sin α - Real.cos α = Real.sqrt 2) 
  (h2 : α ∈ Set.Ioo 0 Real.pi) : 
  Real.sin (2 * α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_equals_negative_one_l1025_102575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_degrees_l1025_102599

noncomputable def rotation_matrix (θ : Real) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

theorem rotation_150_degrees :
  rotation_matrix (150 * π / 180) = ![![-Real.sqrt 3 / 2, -1 / 2],
                                     ![1 / 2, -Real.sqrt 3 / 2]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_degrees_l1025_102599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1025_102593

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 + Real.sin x)^10 + (1 - Real.sin x)^10

-- State the theorem
theorem f_max_value :
  ∃ (M : ℝ), M = 1024 ∧ ∀ x, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → f x ≤ M :=
by
  -- We'll use 1024 as our maximum value
  let M := 1024
  
  -- Prove that M satisfies the conditions
  use M
  
  constructor
  · -- Prove M = 1024
    rfl
    
  · -- Prove ∀ x, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → f x ≤ M
    intro x hx
    sorry  -- The actual proof would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1025_102593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_percentage_approx_l1025_102546

/-- Calculates the markup percentage given the cost price, discount rates, and profit percentage. -/
noncomputable def calculate_markup_percentage (cost_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (profit_percentage : ℝ) : ℝ :=
  let selling_price := cost_price * (1 + profit_percentage / 100)
  let marked_price := selling_price / ((1 - discount1 / 100) * (1 - discount2 / 100))
  let markup := marked_price - cost_price
  (markup / cost_price) * 100

/-- Theorem stating that the markup percentage is approximately 63.54% given the specified conditions. -/
theorem markup_percentage_approx (cost_price : ℝ) (h1 : cost_price = 225) 
  (discount1 : ℝ) (h2 : discount1 = 10)
  (discount2 : ℝ) (h3 : discount2 = 15)
  (profit_percentage : ℝ) (h4 : profit_percentage = 25) :
  ∃ ε > 0, |calculate_markup_percentage cost_price discount1 discount2 profit_percentage - 63.54| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_percentage_approx_l1025_102546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_bound_l1025_102516

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 / 8 - Real.log x

-- State the theorem
theorem function_inequality_implies_bound (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, ∀ t ∈ Set.Icc 0 2, f x < 4 - a * t) → a < 31/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_bound_l1025_102516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_calculation_l1025_102565

/-- Calculates the total interest earned on an investment split between two rates -/
theorem investment_interest_calculation
  (total_investment : ℝ)
  (investment_at_rate1 : ℝ)
  (rate1 : ℝ)
  (rate2 : ℝ)
  (h1 : total_investment = 12000)
  (h2 : investment_at_rate1 = 5500)
  (h3 : rate1 = 0.07)
  (h4 : rate2 = 0.09)
  (h5 : investment_at_rate1 ≤ total_investment) :
  (investment_at_rate1 * rate1 + (total_investment - investment_at_rate1) * rate2) = 970 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_calculation_l1025_102565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_coordinates_point_B_l1025_102504

/-- Define the is_midpoint relation -/
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

/-- Given a point A and a midpoint M of segment AB, calculate the sum of coordinates of point B -/
theorem sum_coordinates_point_B (A M B : ℝ × ℝ) : 
  A = (10, 4) → M = (4, 8) → is_midpoint M A B → A.1 + A.2 + B.1 + B.2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_coordinates_point_B_l1025_102504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_equals_2_sqrt_61_l1025_102554

-- Define the points
def start : ℝ × ℝ := (-5, 6)
def mid : ℝ × ℝ := (0, 0)
def finish : ℝ × ℝ := (6, -5)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem total_distance_equals_2_sqrt_61 :
  distance start mid + distance mid finish = 2 * Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_equals_2_sqrt_61_l1025_102554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1025_102547

noncomputable def f (x : ℝ) := 2 * |Real.sin x|

theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1025_102547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_theorem_l1025_102558

def f (x : ℝ) : ℝ := (1 - 2*x)^8
def g (x : ℝ) : ℝ := (1 + x)^9 * (1 - 2*x)^8

theorem coefficient_theorem :
  (∃ p : Polynomial ℝ, (∀ x, f x = p.eval x) ∧ p.coeff 3 = -448) ∧
  (∃ q : Polynomial ℝ, (∀ x, g x = q.eval x) ∧ q.coeff 2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_theorem_l1025_102558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1025_102584

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sin x ^ 2

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1025_102584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sweater_markup_theorem_l1025_102556

/-- Markup percentage from wholesale to retail price -/
noncomputable def markup_percentage (wholesale : ℝ) (retail : ℝ) : ℝ :=
  (retail - wholesale) / wholesale * 100

/-- Discounted price as a fraction of retail price -/
noncomputable def discounted_price_fraction (discount_percentage : ℝ) : ℝ :=
  1 - discount_percentage / 100

theorem sweater_markup_theorem (wholesale : ℝ) (retail : ℝ) 
  (h1 : wholesale > 0) 
  (h2 : discounted_price_fraction 70 * retail = 1.4 * wholesale) : 
  markup_percentage wholesale retail = 367 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sweater_markup_theorem_l1025_102556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1025_102589

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

-- Define the range of f
def range_f : Set ℝ := {y | ∃ x, f x = y}

-- Theorem statement
theorem f_properties :
  (∀ y, y ∈ range_f → -3 ≤ y ∧ y ≤ 3) ∧
  (∀ a b, a ∈ range_f → b ∈ range_f → 3 * |a + b| ≤ |a * b + 9|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1025_102589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_calculation_l1025_102510

/-- Calculates the final salary after a raise and a pay cut -/
noncomputable def final_salary (initial : ℝ) (raise_percent : ℝ) (cut_percent : ℝ) : ℝ :=
  initial * (1 + raise_percent / 100) * (1 - cut_percent / 100)

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem salary_calculation :
  round_to_nearest (final_salary 3500 10 15) = 3273 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_calculation_l1025_102510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_real_root_equation_l1025_102503

theorem positive_real_root_equation (a : ℝ) : 
  a > 0 ∧ (2 : ℝ)^(a + 1) = (8 : ℝ)^((1/a) - (1/3)) → a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_real_root_equation_l1025_102503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_abcd_l1025_102505

theorem max_sum_abcd (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a + b ^ 2 + c ^ 3 + d ^ 4 = 90 →
  d > 1 →
  (∀ w x y z : ℕ+, 
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    w + x ^ 2 + y ^ 3 + z ^ 4 = 90 →
    z > 1 →
    a + b + c + d ≥ w + x + y + z) →
  a + b + c + d = 70 := by
  sorry

#check max_sum_abcd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_abcd_l1025_102505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_star_equation_l1025_102587

-- Define the ⋆ operation
noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a + b) / Real.sqrt (a - b)

-- Theorem statement
theorem solve_star_equation :
  ∃ (x : ℝ), x > 20 ∧ star x 20 = 3 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_star_equation_l1025_102587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_is_unique_l1025_102570

/-- Represents a 2D point --/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a 2D vector --/
structure Vec2D where
  x : ℚ
  y : ℚ

/-- Represents a line in parametric form --/
structure ParametricLine where
  point : Point
  direction : Vec2D

def line1 : ParametricLine := {
  point := { x := 1, y := 4 },
  direction := { x := -2, y := 3 }
}

def line2 : ParametricLine := {
  point := { x := 2, y := 5 },
  direction := { x := 3, y := 1 }
}

def intersection_point : Point := {
  x := 3/7,
  y := 34/7
}

/-- Theorem stating that the given point is the unique intersection of the two lines --/
theorem intersection_point_is_unique :
  ∃! t u : ℚ,
    (line1.point.x + t * line1.direction.x = intersection_point.x) ∧
    (line1.point.y + t * line1.direction.y = intersection_point.y) ∧
    (line2.point.x + u * line2.direction.x = intersection_point.x) ∧
    (line2.point.y + u * line2.direction.y = intersection_point.y) :=
by
  sorry

#check intersection_point_is_unique

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_is_unique_l1025_102570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_ratio_on_interval_l1025_102573

noncomputable def f (x : ℝ) := 2 * x / (x - 2)

theorem max_min_ratio_on_interval :
  ∀ M m : ℝ,
  (∀ x ∈ Set.Icc 3 4, f x ≤ M) →
  (∃ x ∈ Set.Icc 3 4, f x = M) →
  (∀ x ∈ Set.Icc 3 4, m ≤ f x) →
  (∃ x ∈ Set.Icc 3 4, f x = m) →
  m^2 / M = 8/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_ratio_on_interval_l1025_102573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_negative_one_l1025_102597

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + 1) * (x + a)) / x

theorem odd_function_implies_a_equals_negative_one :
  (∀ x : ℝ, x ≠ 0 → f a x = -f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_negative_one_l1025_102597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PQ_l1025_102559

noncomputable section

-- Define the curve C in polar coordinates
def curve_C (θ : ℝ) : ℝ := 4 * Real.sin (θ - Real.pi/3)

-- Define point Q
def point_Q (φ : ℝ) : ℝ × ℝ := (Real.cos φ, Real.sin φ)

-- Define a point P on curve C
def point_P (θ : ℝ) : ℝ × ℝ := (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ)

-- State the theorem
theorem max_distance_PQ :
  ∀ θ φ : ℝ, Real.sqrt ((point_P θ).1 - (point_Q φ).1)^2 + ((point_P θ).2 - (point_Q φ).2)^2 ≤ 5 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PQ_l1025_102559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_count_l1025_102525

/-- The type of a finite collection of 100 distinct real numbers -/
def DistinctHundredReals := {s : Finset ℝ // s.card = 100}

/-- The function representing the left side minus the right side of the equation -/
noncomputable def f (x : ℝ) (points : DistinctHundredReals) : ℝ :=
  let (a, b) := (points.val.toList.take 50, points.val.toList.drop 50)
  (a.map (fun ai => |x - ai|)).sum - (b.map (fun bi => |x - bi|)).sum

/-- The theorem stating that the maximum number of roots is 49 -/
theorem max_roots_count (points : DistinctHundredReals) :
  (∃ (roots : Finset ℝ), roots.card = 49 ∧ ∀ x ∈ roots, f x points = 0) ∧
  ¬∃ (roots : Finset ℝ), roots.card > 49 ∧ ∀ x ∈ roots, f x points = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_count_l1025_102525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_eq_line_segment_length_l1025_102564

-- Define the ellipse parameters
noncomputable def a : ℝ := 3
noncomputable def c : ℝ := 2 * Real.sqrt 2

-- Define the line parameters
def m : ℝ := 1
def b : ℝ := 2

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := x^2/9 + y^2 = 1

-- Define the line equation
def line_eq (x y : ℝ) : Prop := y = m*x + b

-- Theorem for the standard equation of the ellipse
theorem ellipse_standard_eq : 
  ∀ x y : ℝ, ellipse_eq x y ↔ x^2/9 + y^2 = 1 := by sorry

-- Theorem for the length of the line segment
theorem line_segment_length : 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse_eq x₁ y₁ ∧ 
    ellipse_eq x₂ y₂ ∧
    line_eq x₁ y₁ ∧
    line_eq x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6 * Real.sqrt 3 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_eq_line_segment_length_l1025_102564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_guarantee_l1025_102583

/-- Represents a card in the deck -/
structure Card where
  suit : Fin 4
  rank : Fin 9
deriving Fintype, DecidableEq

/-- The deck of cards -/
def Deck : Finset Card := Finset.univ

/-- A valid distribution of cards between two players -/
structure Distribution where
  player1 : Finset Card
  player2 : Finset Card
  valid : player1 ∪ player2 = Deck ∧ player1 ∩ player2 = ∅ ∧ player1.card = 18 ∧ player2.card = 18

/-- A sequence of played cards -/
def GameSequence := List Card

/-- Checks if a card can be played after the previous card -/
def canPlay (prev next : Card) : Prop :=
  prev.suit = next.suit ∨ prev.rank = next.rank

/-- Counts the number of valid plays in a game sequence for the second player -/
def countValidPlays (dist : Distribution) (seq : GameSequence) : Nat :=
  sorry

/-- The main theorem: the second player can always guarantee at least 15 points -/
theorem second_player_guarantee (dist : Distribution) : 
  ∃ (strategy : GameSequence → Card), 
    ∀ (seq : GameSequence), 
      countValidPlays dist (seq.append [strategy seq]) ≥ 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_guarantee_l1025_102583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_squared_less_than_two_to_n_negation_l1025_102520

theorem factorial_squared_less_than_two_to_n_negation :
  (∀ n : ℕ, (Nat.factorial n)^2 < 2^n) ↔ 
  ¬(∃ n : ℕ, (Nat.factorial n)^2 ≥ 2^n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_squared_less_than_two_to_n_negation_l1025_102520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1025_102542

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1) * x + a

-- Part 1: Solution set for a = 2
def solution_set : Set ℝ := {x | x < 1 ∨ x > 2}

theorem part1 : {x : ℝ | f 2 x > 0} = solution_set := by sorry

-- Part 2: Range of a
noncomputable def upper_bound : ℝ := 2 * Real.sqrt 2 + 3

theorem part2 : ∀ a : ℝ, (∀ x ∈ Set.Ioi 1, f a x + 2*x ≥ 0) → a ≤ upper_bound := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1025_102542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_properties_l1025_102580

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  |16 + 6*x - x^2 - y^2| + |6*x| = 16 + 12*x - x^2 - y^2

def equation2 (a x y : ℝ) : Prop :=
  (a + 15)*y + 15*x - a = 0

-- Define the area of the figure
noncomputable def figureArea : ℝ := 25*Real.pi - 25*Real.arcsin 0.8 + 12

-- Define the values of a for which the system has exactly one solution
def uniqueSolutionValues : Set ℝ := {-20, -12}

-- Theorem statement
theorem system_properties :
  (∀ x y : ℝ, equation1 x y → x ≥ 0 ∧ (x - 3)^2 + y^2 ≤ 25) ∧
  (∀ a : ℝ, a ∈ uniqueSolutionValues ↔
    ∃! p : ℝ × ℝ, equation1 p.1 p.2 ∧ equation2 a p.1 p.2) := by
  sorry

#check system_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_properties_l1025_102580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_clearing_time_approx_7_69_seconds_l1025_102578

-- Define the trains' lengths in meters
noncomputable def train1_length : ℝ := 110
noncomputable def train2_length : ℝ := 200

-- Define the trains' speeds in km/h
noncomputable def train1_speed : ℝ := 80
noncomputable def train2_speed : ℝ := 65

-- Define the conversion factor from km/h to m/s
noncomputable def km_h_to_m_s : ℝ := 1000 / 3600

-- Calculate the total distance
noncomputable def total_distance : ℝ := train1_length + train2_length

-- Calculate the relative speed in m/s
noncomputable def relative_speed : ℝ := (train1_speed + train2_speed) * km_h_to_m_s

-- Calculate the time for trains to clear each other
noncomputable def clearing_time : ℝ := total_distance / relative_speed

-- Theorem statement
theorem trains_clearing_time_approx_7_69_seconds :
  abs (clearing_time - 7.69) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_clearing_time_approx_7_69_seconds_l1025_102578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_transportation_cost_optimal_speed_upper_bound_l1025_102543

/-- Represents the total transportation cost function -/
noncomputable def total_cost (v : ℝ) (a : ℝ) : ℝ := 800 * (1/4 * v + a / v)

/-- The domain of the speed -/
def speed_domain (v : ℝ) : Prop := 0 < v ∧ v ≤ 100

theorem optimal_transportation_cost (a : ℝ) (h : 0 < a ∧ a ≤ 2500) :
  ∃ (v : ℝ), speed_domain v ∧ 
    (∀ (u : ℝ), speed_domain u → total_cost v a ≤ total_cost u a) ∧
    v = 2 * Real.sqrt a ∧
    total_cost v a = 800 * Real.sqrt a :=
by sorry

theorem optimal_speed_upper_bound (a : ℝ) (h : a > 2500) :
  ∃ (v : ℝ), speed_domain v ∧ 
    (∀ (u : ℝ), speed_domain u → total_cost v a ≤ total_cost u a) ∧
    v = 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_transportation_cost_optimal_speed_upper_bound_l1025_102543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_division_problem_l1025_102574

/-- Represents the number of days it takes for a person to complete the entire work -/
def CompletionDays : Type := ℕ

/-- Represents the number of days a person works on the job -/
def WorkDays : Type := ℕ

/-- Calculate the work done by a person given their completion days and work days -/
def workDone (completionDays workDays : ℕ) : ℚ :=
  (workDays : ℚ) / completionDays

theorem work_division_problem 
  (a_completion_days b_completion_days b_remaining_days : ℕ) 
  (a_work_days : ℕ) :
  a_completion_days = 15 →
  b_completion_days = 18 →
  b_remaining_days = 12 →
  workDone a_completion_days a_work_days + workDone b_completion_days b_remaining_days = 1 →
  a_work_days = 5 := by
  sorry

#eval workDone 15 5 + workDone 18 12  -- This should evaluate to 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_division_problem_l1025_102574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quilt_length_is_seven_l1025_102507

/-- Represents the properties of a rectangular quilt --/
structure Quilt where
  width : ℚ
  costPerSquareFoot : ℚ
  totalCost : ℚ

/-- Calculates the length of a quilt given its properties --/
def quiltLength (q : Quilt) : ℚ :=
  (q.totalCost / q.costPerSquareFoot) / q.width

/-- Theorem stating that a quilt with the given properties has a length of 7 feet --/
theorem quilt_length_is_seven (q : Quilt) 
  (h1 : q.width = 8)
  (h2 : q.costPerSquareFoot = 40)
  (h3 : q.totalCost = 2240) : 
  quiltLength q = 7 := by
  sorry

#eval quiltLength { width := 8, costPerSquareFoot := 40, totalCost := 2240 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quilt_length_is_seven_l1025_102507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_trig_identity_l1025_102555

noncomputable def a (α : Real) : Real × Real := (Real.cos α, -2)
noncomputable def b (α : Real) : Real × Real := (Real.sin α, 1)

theorem parallel_vectors_trig_identity (α : Real) 
  (h : ∃ k : Real, a α = k • b α) : 
  2 * Real.sin α * Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_trig_identity_l1025_102555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_five_and_six_l1025_102512

theorem subsets_containing_five_and_six (S : Finset Nat) :
  S = {1, 2, 3, 4, 5, 6} →
  (Finset.filter (fun A => 5 ∈ A ∧ 6 ∈ A) (Finset.powerset S)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_five_and_six_l1025_102512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_operations_divisibility_l1025_102590

theorem polynomial_operations_divisibility (p : ℕ → ℤ) (a b : ℤ) :
  (∀ n, n > 8 → p n = 0) →  -- p is a polynomial of degree at most 8
  p 8 = 1 →                 -- coefficient of x^8 is 1
  p 7 = 1 →                 -- coefficient of x^7 is 1
  (∀ n, n < 7 → p n = 0) →  -- all lower degree terms are 0
  (∃ (k : ℕ) (q : ℕ → ℤ), 
    (∀ n, q n = (n * p n) ∨ q n = ((n + 1) * p (n + 1))) ∧
    q 1 = a ∧ q 0 = b) →    -- a and b result from operations
  49 ∣ (a - b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_operations_divisibility_l1025_102590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_necessary_not_sufficient_l1025_102524

/-- The function f(x) = cos²(ax) - sin²(ax) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.cos (a * x))^2 - (Real.sin (a * x))^2

/-- The minimum positive period of a function -/
noncomputable def min_positive_period (g : ℝ → ℝ) : ℝ := sorry

/-- Theorem: a=1 is a necessary but not sufficient condition for f to have a minimum positive period of π -/
theorem a_eq_one_necessary_not_sufficient :
  (∀ a : ℝ, min_positive_period (f a) = π → a = 1) ∧
  (∃ a : ℝ, a = 1 ∧ min_positive_period (f a) ≠ π) := by
  sorry

#check a_eq_one_necessary_not_sufficient

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_necessary_not_sufficient_l1025_102524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_of_eggs_l1025_102519

-- Define the type for the number of eggs
def NumEggs := Nat

-- Define the capacity of each container
def ContainerCapacity : Nat := 10

-- Define the number of containers with missing eggs
def ContainersWithMissingEggs : Nat := 3

-- Define the minimum number of eggs stated in the problem
def MinimumEggs : Nat := 100

-- Define a function to calculate the total number of eggs
def totalEggs (numContainers : Nat) : Nat :=
  numContainers * ContainerCapacity - ContainersWithMissingEggs

-- Theorem statement
theorem smallest_number_of_eggs :
  ∃ (n : Nat), 
    totalEggs n > MinimumEggs ∧
    (∀ (m : Nat), totalEggs m > MinimumEggs → totalEggs n ≤ totalEggs m) ∧
    totalEggs n = 107 := by
  sorry

#eval totalEggs 11  -- This should output 107

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_of_eggs_l1025_102519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_m_value_l1025_102537

/-- A function f(x) = ax^n + b is linear if and only if n = 1 and a ≠ 0 -/
def IsLinearFunction (a n : ℝ) : Prop := n = 1 ∧ a ≠ 0

/-- The given function y = (m+2)x^(m^2-3) + m-2 -/
noncomputable def f (m x : ℝ) : ℝ := (m + 2) * (x ^ (m^2 - 3)) + m - 2

theorem linear_function_m_value :
  ∃! m : ℝ, IsLinearFunction (m + 2) (m^2 - 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_m_value_l1025_102537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l1025_102513

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem x0_value (x0 : ℝ) (h : x0 > 0) :
  (deriv f x0 = 2) → x0 = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_l1025_102513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_for_given_parameters_l1025_102577

/-- The time taken for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed : ℝ) : ℝ :=
  train_length / train_speed

/-- Theorem stating that the time taken for a 300m long train traveling at 108 m/s to cross an electric pole is 300/108 seconds -/
theorem train_crossing_time_for_given_parameters :
  train_crossing_time 300 108 = 300 / 108 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_for_given_parameters_l1025_102577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_side_length_difference_l1025_102521

/-- A circle with radius R divided into ten equal parts -/
structure DecagonCircle where
  R : ℝ
  center : ℂ
  division_points : Fin 10 → ℂ

/-- The regular decagon formed by connecting adjacent points -/
def regular_decagon (c : DecagonCircle) : Fin 10 → ℂ := c.division_points

/-- The star-shaped decagon formed by connecting every third point -/
def star_decagon (c : DecagonCircle) : Fin 10 → ℂ := 
  fun i => c.division_points ((i * 3) % 10)

/-- Side length of the regular decagon -/
noncomputable def regular_side_length (c : DecagonCircle) : ℝ :=
  Complex.abs (c.division_points 1 - c.division_points 0)

/-- Side length of the star-shaped decagon -/
noncomputable def star_side_length (c : DecagonCircle) : ℝ :=
  Complex.abs (c.division_points 3 - c.division_points 0)

/-- The theorem to be proved -/
theorem decagon_side_length_difference (c : DecagonCircle) :
  star_side_length c - regular_side_length c = c.R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_side_length_difference_l1025_102521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_regions_is_six_l1025_102576

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = 2 * x
def line2 (x y : ℝ) : Prop := y = (1 / 2) * x
def line3 (x y : ℝ) : Prop := y = x

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a region as a set of points
def Region := Set Point

-- Define the set of all regions formed by the lines
noncomputable def regions : Finset Region := sorry

-- Theorem stating that the number of regions is 6
theorem num_regions_is_six : regions.card = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_regions_is_six_l1025_102576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_angle_slope_l1025_102530

noncomputable def slope_of_line (a b c : ℝ) : ℝ := a / -b

noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m

theorem double_angle_slope (a b c : ℝ) (h : b ≠ 0) :
  let m₁ := slope_of_line a b c
  let θ₁ := inclination_angle m₁
  let θ₂ := 2 * θ₁
  Real.tan θ₂ = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_angle_slope_l1025_102530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_root_of_three_minus_pi_cubed_l1025_102566

theorem eighth_root_of_three_minus_pi_cubed : 
  ((3 - Real.pi) ^ 8 : ℝ) ^ (1/8 : ℝ) = Real.pi - 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_root_of_three_minus_pi_cubed_l1025_102566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chlorous_acid_molecular_weight_l1025_102540

/-- The molecular weight of Chlorous acid, given that 6 moles weigh 408 grams -/
theorem chlorous_acid_molecular_weight :
  (408 : ℝ) / 6 = 68 := by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chlorous_acid_molecular_weight_l1025_102540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_f_l1025_102522

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * Real.arccos x) + 4 * Real.arcsin (Real.sin (x / 2))

-- Theorem for the domain and range of f
theorem domain_and_range_of_f :
  (∀ x : ℝ, f x ≠ 0 → abs x ≤ 1) ∧
  (∀ y : ℝ, (∃ x : ℝ, f x = y) → -3/2 ≤ y ∧ y ≤ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_f_l1025_102522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solution_count_l1025_102531

theorem integer_solution_count : ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ |7*x + 2| ≤ 9) ∧ Finset.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solution_count_l1025_102531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l1025_102502

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - (3/2) * x^2 + 1

-- State the theorem
theorem tangent_line_and_inequality (a : ℝ) (h : a > 0) :
  -- Part 1: Tangent line equation when a = 1
  (a = 1 → ∀ x : ℝ, (6 * x - 9 = f 1 2 + (deriv (f 1)) 2 * (x - 2))) ∧
  -- Part 2: Condition for f(x) < a² on [-1, 1/2]
  (a > 1 ↔ ∀ x : ℝ, x ∈ Set.Icc (-1) (1/2) → f a x < a^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l1025_102502
