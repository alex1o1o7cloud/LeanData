import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1072_107217

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Define the domain
def domain : Set ℝ := Set.Icc 1 5

-- Theorem statement
theorem range_of_f :
  Set.range (fun x ↦ f x) ∩ (Set.image f domain) = Set.Ico 2 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1072_107217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_hyperbola_properties_l1072_107231

/-- Represents an ellipse with equation 3x^2 + y^2 = 18 -/
structure Ellipse where
  equation : ∀ x y : ℝ, 3 * x^2 + y^2 = 18

/-- Represents a hyperbola derived from the ellipse -/
structure Hyperbola where
  ellipse : Ellipse
  equation : ∀ x y : ℝ, y^2 / 6 - x^2 / 12 = 1

/-- Predicate to check if two points are the foci of an ellipse -/
def are_foci (e : Ellipse) (f₁ f₂ : ℝ × ℝ) : Prop :=
  sorry

/-- Function to calculate the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating the properties of the ellipse and its derived hyperbola -/
theorem ellipse_and_hyperbola_properties (e : Ellipse) :
  (∃ f₁ f₂ : ℝ × ℝ, f₁ = (0, -2) ∧ f₂ = (0, 2) ∧ are_foci e f₁ f₂) ∧
  (eccentricity e = Real.sqrt 6 / 3) ∧
  (∃ h : Hyperbola, h.ellipse = e) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_hyperbola_properties_l1072_107231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_geq_two_l1072_107229

-- Define the triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0

-- Define the altitude AD
noncomputable def altitude (t : RightTriangle) : ℝ × ℝ := sorry

-- Define the incenters I₁ and I₂
noncomputable def incenter_ABD (t : RightTriangle) : ℝ × ℝ := sorry
noncomputable def incenter_ACD (t : RightTriangle) : ℝ × ℝ := sorry

-- Define points K and L
noncomputable def point_K (t : RightTriangle) : ℝ × ℝ := sorry
noncomputable def point_L (t : RightTriangle) : ℝ × ℝ := sorry

-- Define areas of triangles ABC and AKL
noncomputable def area_ABC (t : RightTriangle) : ℝ := sorry
noncomputable def area_AKL (t : RightTriangle) : ℝ := sorry

-- Theorem statement
theorem area_ratio_geq_two (t : RightTriangle) :
  (area_ABC t) / (area_AKL t) ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_geq_two_l1072_107229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_arithmetic_sum_l1072_107211

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n • a₁ + (n * (n - 1) / 2) • d

/-- Theorem: Sum of first n terms of the specific arithmetic sequence -/
theorem specific_arithmetic_sum (n : ℕ) :
  arithmetic_sum (-2) 3 n = n • (3 • n - 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_arithmetic_sum_l1072_107211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1072_107289

/-- Represents the dimensions of a rectangular floor -/
structure FloorDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a square tile -/
structure TileDimensions where
  side : ℝ

/-- Calculates the shaded area on a floor with given dimensions and tile properties -/
noncomputable def shadedArea (floor : FloorDimensions) (tile : TileDimensions) : ℝ :=
  let totalArea := floor.length * floor.width
  let tileArea := tile.side * tile.side
  let whiteCircleArea := Real.pi * 1^2
  let shadedTileArea := tileArea - whiteCircleArea
  let numTiles := (floor.length / tile.side) * (floor.width / tile.side)
  numTiles * shadedTileArea

theorem shaded_area_calculation :
  let floor := FloorDimensions.mk 16 20
  let tile := TileDimensions.mk 2
  shadedArea floor tile = 320 - 80 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1072_107289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_divisible_by_1963_l1072_107218

theorem fraction_divisible_by_1963 (n : ℕ) :
  ∃ (k m : ℤ),
    13 * 733^n + 1950 * 582^n = 1963 * k ∧
    333^n - 733^n - 1068^n + 431^n = 1963 * m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_divisible_by_1963_l1072_107218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_lateral_surface_area_l1072_107216

/-- The lateral surface area of a regular pyramid with side length 1 and base perimeter 4 is √3. -/
theorem regular_pyramid_lateral_surface_area :
  ∀ (p : Real),
    p > 0 →
    p = 1 →
    4 * p = 4 →
    4 * (1/2 * p^2 * Real.sin (π/3)) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_lateral_surface_area_l1072_107216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1072_107220

-- Define the function pairs
noncomputable def f1 (x : ℝ) : ℝ := x
noncomputable def g1 (x : ℝ) : ℝ := Real.sqrt (x^2)

def f2 (x : ℝ) : ℝ := x
noncomputable def g2 (x : ℝ) : ℝ := x^2 / x

def f3 (x : ℝ) : ℝ := x^2
def g3 (t : ℝ) : ℝ := t^2

noncomputable def f4 (x : ℝ) : ℝ := Real.sqrt (x + 1) * Real.sqrt (x - 1)
noncomputable def g4 (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)

-- Theorem statement
theorem function_equality :
  (∃ x, f1 x ≠ g1 x) ∧
  (∃ x, f2 x ≠ g2 x) ∧
  (∀ x, f3 x = g3 x) ∧
  (∃ x, f4 x ≠ g4 x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1072_107220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_db_length_l1072_107200

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the lengths of the sides and diagonal
def AB : ℝ := 5
def BC : ℝ := 17
def CD : ℝ := 5
def DA : ℝ := 9
def DB : ℕ := 13

-- Theorem statement
theorem db_length : DB = 13 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_db_length_l1072_107200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_left_oar_probability_l1072_107252

-- Define the probability type as a real number between 0 and 1
def Probability := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

-- Define the probability of being able to row the canoe
noncomputable def prob_can_row : Probability := ⟨0.84, by norm_num⟩

-- Define the probability that an oar works
noncomputable def prob_oar_works : Probability := ⟨0.6, by norm_num⟩

-- State the theorem
theorem left_oar_probability :
  (1 - (1 - prob_oar_works.val) * (1 - prob_oar_works.val) = prob_can_row.val) →
  prob_oar_works.val = 0.6 := by
  intro h
  -- The proof goes here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_left_oar_probability_l1072_107252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l1072_107298

/-- The sum of the infinite series (n^3 + n^2 - n) / ((n + 3)!) from n = 1 to infinity equals 1/6 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (n^3 + n^2 - n : ℝ) / Nat.factorial (n + 3)) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l1072_107298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_efficiency_l1072_107213

noncomputable def tank_capacity : ℚ := 12
noncomputable def fraction_remaining : ℚ := 2/3
noncomputable def distance_to_work : ℚ := 10

theorem fuel_efficiency :
  let fuel_used : ℚ := tank_capacity * (1 - fraction_remaining)
  let total_distance : ℚ := 2 * distance_to_work
  total_distance / fuel_used = 5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_efficiency_l1072_107213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_and_even_l1072_107238

-- Define the function f(x) = -lg |x|
noncomputable def f (x : ℝ) : ℝ := -Real.log (abs x) / Real.log 2

-- State the theorem
theorem f_monotone_decreasing_and_even :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x > f y) ∧
  (∀ x : ℝ, f x = f (-x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_and_even_l1072_107238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1234_formula_l1072_107278

/-- The base function f₁ -/
noncomputable def f₁ (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

/-- The recursive definition of fₙ -/
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => λ x => x  -- f₀(x) = x (for completeness)
  | 1 => f₁
  | n + 1 => λ x => f₁ (f n x)

/-- The main theorem stating that f₁₂₃₄(x) = 1 / (1 - x) -/
theorem f_1234_formula (x : ℝ) : f 1234 x = 1 / (1 - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1234_formula_l1072_107278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_minus_circles_area_l1072_107206

/-- The area of a 4 × 3 rectangle minus two non-overlapping circles with diameters 2 and 1 is closest to 8 -/
theorem rectangle_minus_circles_area : ℝ :=
  let rectangle_area : ℝ := 4 * 3
  let circle1_area : ℝ := Real.pi * (2/2)^2
  let circle2_area : ℝ := Real.pi * (1/2)^2
  let remaining_area : ℝ := rectangle_area - circle1_area - circle2_area

  -- We assume the circles don't overlap
  have circles_dont_overlap : circle1_area + circle2_area ≤ rectangle_area := by sorry

  -- The theorem
  have closest_to_eight : 
    ∀ n : ℤ, |remaining_area - 8| ≤ |remaining_area - ↑n| ∨ n = 8 := by sorry

  -- Return the remaining area
  remaining_area


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_minus_circles_area_l1072_107206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_num_divisors_squared_l1072_107255

/-- The number of pairs of positive integers (a, b) satisfying ab/(a + b) = n -/
def f (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = n * (p.1 + p.2)) 
    (Finset.product (Finset.range (n^2 + 1)) (Finset.range (n^2 + 1)))).card

/-- The number of divisors of n^2 -/
def num_divisors_squared (n : ℕ) : ℕ :=
  (Finset.filter (fun d : ℕ => d > 0 ∧ n^2 % d = 0) (Finset.range (n^2 + 1))).card

theorem f_equals_num_divisors_squared (n : ℕ) (h : n > 0) : f n = num_divisors_squared n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_num_divisors_squared_l1072_107255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1072_107215

-- Define the constants
noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2

-- State the theorem
theorem log_inequality : a * b < a + b ∧ a + b < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1072_107215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_difference_l1072_107249

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_sum_difference
  (n : ℕ) :
  let a₁ : ℝ := 1
  let d : ℝ := 2
  let S := sum_arithmetic_sequence a₁ d
  S (n + 2) - S n = 36 → n = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_difference_l1072_107249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1072_107246

theorem equation_solutions (x : ℝ) : 
  (x > 0 ∧ x^(Real.log x / Real.log 10) = x^5 / 10000) ↔ (x = 10 ∨ x = 10000) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1072_107246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1072_107257

noncomputable section

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = p * x ∧ p > 0

def line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p / 4, 0)

-- Define the y-intercept of the line
def y_intercept (b : ℝ) : ℝ × ℝ := (0, b)

-- Define the area of a triangle given three points
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem parabola_line_intersection (p : ℝ) :
  (∃ b : ℝ, 
    line 2 b (p/4) 0 ∧  -- Line passes through focus
    triangle_area (0, 0) (y_intercept b) (focus p) = 1) →
  p = 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1072_107257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_propositions_l1072_107276

theorem complex_number_propositions :
  (∀ (a b c : ℂ), (a^2 + b^2).re > (c^2).re → (a^2 + b^2 - c^2).re > 0) ∧
  (∃ (a b c : ℂ), (a^2 + b^2 - c^2).re > 0 ∧ ¬((a^2 + b^2).re > (c^2).re)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_propositions_l1072_107276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_correct_l1072_107256

/-- Rational Woman's path -/
noncomputable def rational_path (t : ℝ) : ℝ × ℝ :=
  (2 * Real.cos t, 2 * Real.sin t)

/-- Irrational Woman's path -/
noncomputable def irrational_path (t : ℝ) : ℝ × ℝ :=
  (3 + 3 * Real.cos (t / 2), 3 * Real.sin (t / 2))

/-- The smallest possible distance between points on the two paths -/
def smallest_distance : ℝ := 1

theorem smallest_distance_correct :
  ∀ t₁ t₂ : ℝ,
  let (x₁, y₁) := rational_path t₁
  let (x₂, y₂) := irrational_path t₂
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≥ smallest_distance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_correct_l1072_107256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomials_l1072_107233

theorem gcd_of_polynomials (a : ℤ) (h : ∃ k : ℤ, a = 2 * k * 1009) :
  Int.gcd (2 * a^2 + 31 * a + 58) (a + 15) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomials_l1072_107233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kelsey_videos_l1072_107267

/-- The number of videos watched by three friends --/
def total_videos : ℕ := 411

/-- The difference in videos watched between Kelsey and Ekon --/
def kelsey_ekon_diff : ℕ := 43

/-- The difference in videos watched between Uma and Ekon --/
def uma_ekon_diff : ℕ := 17

/-- Theorem stating that Kelsey watched 160 videos --/
theorem kelsey_videos : ∃ k : ℕ, k = 160 := by
  let uma_videos := (total_videos + kelsey_ekon_diff - uma_ekon_diff) / 3
  let kelsey_videos := uma_videos + kelsey_ekon_diff - uma_ekon_diff
  use kelsey_videos
  sorry

#eval (total_videos + kelsey_ekon_diff - uma_ekon_diff) / 3 + kelsey_ekon_diff - uma_ekon_diff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kelsey_videos_l1072_107267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_line_equation_l1072_107293

-- Define the point M
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance function
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Define the distance from a point to a line y = k
def distanceToLine (p : Point) (k : ℝ) : ℝ :=
  |p.y - k|

-- Define the focus F
def F : Point :=
  ⟨0, 1⟩

-- Define the point P
def P : Point :=
  ⟨0, -1⟩

-- Define the origin O
def O : Point :=
  ⟨0, 0⟩

-- Define the trajectory of M
def trajectoryM (p : Point) : Prop :=
  p.x^2 = 4 * p.y

-- Define the line l
def lineL (x : ℝ) : ℝ :=
  2 * x - 1

-- State the theorem
theorem trajectory_and_line_equation :
  ∀ (M A B : Point),
  (∀ (M : Point), distanceToLine M (-2) = distance M F + 1) →
  trajectoryM A ∧ trajectoryM B →
  A.y = lineL A.x ∧ B.y = lineL B.x →
  (A.y / A.x + B.y / B.x = 2) →
  (∀ (M : Point), trajectoryM M ↔ M.x^2 = 4 * M.y) ∧
  (∀ (x : ℝ), lineL x = 2 * x - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_line_equation_l1072_107293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unit_cost_l1072_107244

-- Define the cost function
noncomputable def cost_function (x : ℝ) : ℝ := x^2 + 2*x + 9

-- Define the unit cost function
noncomputable def unit_cost (x : ℝ) : ℝ := (cost_function x) / x

-- Theorem statement
theorem min_unit_cost :
  ∀ x : ℝ, x > 0 → unit_cost x ≥ 8 ∧ 
  (unit_cost x = 8 ↔ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unit_cost_l1072_107244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lt_b_lt_c_l1072_107295

-- Define a, b, and c
noncomputable def a : ℝ := (1/2) ^ (3/4)
noncomputable def b : ℝ := (3/4) ^ (1/2)
noncomputable def c : ℝ := Real.log 3 / Real.log 2

-- Theorem statement
theorem a_lt_b_lt_c : a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lt_b_lt_c_l1072_107295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_4_l1072_107223

/-- A function f with the given properties -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem f_value_at_pi_4 (ω φ : ℝ) 
  (h1 : ω > 0) 
  (h2 : 0 < φ) (h3 : φ < π) 
  (h4 : ∃ (k : ℤ), f ω φ ((2*k+1)*π/(2*ω)) = f ω φ ((2*k-1)*π/(2*ω))) 
  (h5 : Real.tan φ = Real.sqrt 3 / 3) :
  f ω φ (π/4) = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_4_l1072_107223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_average_speed_l1072_107292

/-- Calculates the average speed of a cyclist's trip given two segments with different speeds and distances. -/
theorem cyclist_average_speed (d1 d2 v1 v2 : ℝ) (h1 : d1 = 9) (h2 : d2 = 12) (h3 : v1 = 12) (h4 : v2 = 9) :
  ∃ ε > 0, |((d1 + d2) / (d1 / v1 + d2 / v2)) - 10.1| < ε :=
by
  -- Define the total distance, total time, and average speed
  let total_distance := d1 + d2
  let total_time := d1 / v1 + d2 / v2
  let average_speed := total_distance / total_time

  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_average_speed_l1072_107292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_problem_l1072_107221

/-- Represents the possible directions an ant can move --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents the path of an ant --/
def AntPath := List Direction

/-- Calculates the final position of an ant given its path and q --/
def finalPosition (path : AntPath) (q : ℚ) : ℚ × ℚ :=
  sorry

/-- Determines if two paths are different --/
def differentPaths (path1 path2 : AntPath) : Prop :=
  sorry

/-- Theorem: The only possible value for q is 1 --/
theorem ant_problem (q : ℚ) :
  (q > 0) →
  (∃ (n : ℕ) (path1 path2 : AntPath),
    (path1.length = n) ∧
    (path2.length = n) ∧
    (differentPaths path1 path2) ∧
    (finalPosition path1 q = finalPosition path2 q)) →
  q = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_problem_l1072_107221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_upper_bound_l1072_107280

theorem sine_upper_bound : ∀ x : ℝ, Real.sin x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_upper_bound_l1072_107280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l1072_107279

-- Define the points P and Q
def P : ℝ × ℝ := (2, 1)
def Q : ℝ × ℝ := (1, 4)

-- Define the line on which R lies
def line_R (x y : ℝ) : Prop := x - y = 3

-- Define the area of a triangle given three points
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

-- Theorem statement
theorem triangle_PQR_area :
  ∃ (R : ℝ × ℝ), line_R R.1 R.2 ∧ triangle_area P Q R = 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l1072_107279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_between_c_and_b_l1072_107236

noncomputable def a : ℝ := (Real.sqrt 2 / 2) * (Real.sin (17 * Real.pi / 180) + Real.cos (17 * Real.pi / 180))
noncomputable def b : ℝ := 2 * (Real.cos (13 * Real.pi / 180))^2 - 1
noncomputable def c : ℝ := Real.sqrt 3 / 2

theorem a_between_c_and_b : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_between_c_and_b_l1072_107236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_in_new_basis_l1072_107212

open InnerProductSpace

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the orthonormal basis vectors
variable (a b c : V)

-- Define the orthonormality condition
axiom orthonormal : Orthonormal ℝ ![a, b, c]

-- Define vector p in terms of the original basis
def p (a b c : V) : V := 3 • a + 2 • b + 1 • c

-- Define the new basis vectors
def new_basis_1 (a : V) : V := a
def new_basis_2 (b c : V) : V := b + c
def new_basis_3 (b c : V) : V := b - c

-- State the theorem
theorem unit_vector_in_new_basis (a b c : V) (h : Orthonormal ℝ ![a, b, c]) :
  ∃ (k : ℝ), k • p a b c = (3*Real.sqrt 14/14) • new_basis_1 a + 
                           (3*Real.sqrt 14/28) • new_basis_2 b c + 
                           (Real.sqrt 14/28) • new_basis_3 b c ∧
  ‖k • p a b c‖ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_in_new_basis_l1072_107212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_existence_l1072_107245

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (a * x) - a * Real.sin x

-- Define the set of values for a
def A : Set ℝ := {a | a < -1/2 ∨ a > 1/2 ∨ a = 0}

-- Theorem statement
theorem root_existence (a : ℝ) : 
  (∃ x ∈ Set.Ioo 0 (2 * Real.pi), f a x = 0) ↔ a ∈ A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_existence_l1072_107245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1072_107273

-- Define the function f as noncomputable
noncomputable def f (x θ : Real) : Real :=
  Real.sin x ^ 2 + Real.sqrt 3 * Real.tan θ * Real.cos x + (Real.sqrt 3 / 8) * Real.tan θ - 3 / 2

-- State the theorem
theorem f_properties :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2),
    f x (Real.pi / 3) ≤ 15 / 8 ∧
    f 0 (Real.pi / 3) = 15 / 8) ∧
  (∃ θ ∈ Set.Icc 0 (Real.pi / 3),
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x θ ≤ -1 / 8) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x θ = -1 / 8) ∧
    θ = Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1072_107273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_players_can_force_tie_l1072_107285

/-- Represents a player in the card game -/
inductive Player : Type
| Ana : Player
| Benito : Player

/-- The game state -/
structure GameState :=
  (round : Nat)
  (ana_cards : List Nat)
  (benito_cards : List Nat)

/-- The game setup -/
def initial_state : GameState :=
  { round := 0
  , ana_cards := [0]
  , benito_cards := [] }

/-- The total number of rounds in the game -/
def total_rounds : Nat := 2020

/-- Calculates the score of a player given their cards -/
def score (cards : List Nat) : Nat :=
  cards.sum

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Bool

/-- Helper function to play the game (added implementation) -/
def play_game (strategy1 strategy2 : Strategy) (initial : GameState) (rounds : Nat) : GameState :=
  sorry

/-- Determines if a strategy leads to a tie or better -/
def is_tie_or_better (strategy : Strategy) : Prop :=
  ∀ (opponent_strategy : Strategy),
    let final_state := play_game strategy opponent_strategy initial_state total_rounds
    score final_state.ana_cards ≥ score final_state.benito_cards

/-- The main theorem: both players can force at least a tie -/
theorem both_players_can_force_tie :
  ∃ (ana_strategy benito_strategy : Strategy),
    is_tie_or_better ana_strategy ∧ is_tie_or_better benito_strategy :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_players_can_force_tie_l1072_107285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrench_twice_hammer_l1072_107227

/-- The weight of a hammer -/
def H : ℝ := sorry

/-- The weight of a wrench -/
def W : ℝ := sorry

/-- Hammers and wrenches have uniform weights -/
axiom uniform_weights : H > 0 ∧ W > 0

/-- The total weight of 2 hammers and 2 wrenches is one-third of 8 hammers and 5 wrenches -/
axiom weight_relation : 2 * H + 2 * W = (1 / 3) * (8 * H + 5 * W)

/-- Theorem: The weight of one wrench is twice the weight of one hammer -/
theorem wrench_twice_hammer : W = 2 * H := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrench_twice_hammer_l1072_107227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l1072_107219

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 ≤ 7} = Set.Icc (-4) 3 := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f x a ≥ 2*a + 1} = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l1072_107219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_X_l1072_107224

-- Define the total number of students
def total_students : ℕ := 12

-- Define the number of male students
def male_students : ℕ := 7

-- Define the number of female students
def female_students : ℕ := 5

-- Define the number of students to be selected
def selected_students : ℕ := 3

-- Define X as the number of male students selected
def X : Fin 4 := sorry

-- Define the probability mass function for X
noncomputable def pmf_X : Fin 4 → ℝ
| 0 => (Nat.choose female_students selected_students : ℝ) / (Nat.choose total_students selected_students : ℝ)
| 1 => (Nat.choose male_students 1 * Nat.choose female_students 2 : ℝ) / (Nat.choose total_students selected_students : ℝ)
| 2 => (Nat.choose male_students 2 * Nat.choose female_students 1 : ℝ) / (Nat.choose total_students selected_students : ℝ)
| 3 => (Nat.choose male_students selected_students : ℝ) / (Nat.choose total_students selected_students : ℝ)

-- Theorem stating that the expected value of X is 7/4
theorem expected_value_X :
  Finset.sum (Finset.range 4) (fun i => (i : ℝ) * pmf_X i) = 7/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_X_l1072_107224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l1072_107207

-- Define the cost function
noncomputable def C (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10 * x^2 + 100 * x
  else if x ≥ 40 then 501 * x + 10000 / x - 4500
  else 0

-- Define the profit function
noncomputable def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then -10 * x^2 + 400 * x - 2500
  else if x ≥ 40 then 2000 - (x + 10000 / x)
  else 0

-- Theorem statement
theorem max_profit_at_100 :
  ∀ x : ℝ, x > 0 → L x ≤ 1800 ∧ L 100 = 1800 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l1072_107207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_at_2007_l1072_107288

open BigOperators

def Q (n : ℕ) : ℚ :=
  ∏ k in Finset.range (n - 3), 2 * (1 - 1 / (k + 4 : ℚ))

theorem Q_value_at_2007 :
  Q 2007 = (3 * 2^2004 : ℚ) / 2007 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_at_2007_l1072_107288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1072_107266

open Set Real

def A : Set ℝ := {x | (2 : ℝ)^x > 1}
def B : Set ℝ := {x | -1 < x ∧ x < 1}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem problem_solution :
  (∀ x, x ∈ A ∪ B ↔ x > -1) ∧
  (∀ x, x ∈ (𝒰 \ A) ∩ B ↔ -1 < x ∧ x ≤ 0) ∧
  (∀ a, B ∪ C a = C a → a ≥ 2) :=
by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1072_107266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l1072_107261

theorem line_segment_length : ∃ (length : Real), length = 8 * Real.sqrt 2 := by
  -- Define the endpoints
  let x₁ : Real := 4
  let y₁ : Real := 1
  let x₂ : Real := 12
  let y₂ : Real := 9

  -- Define the length of the line segment
  let length := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

  -- State the existence of the length and its value
  use length

  -- Prove the equality (skipped with sorry)
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l1072_107261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_equation_solution_l1072_107237

theorem arcsin_equation_solution :
  ∃! x : ℝ, 
    x ∈ Set.Icc (-1 : ℝ) 1 ∧ 
    (3 * x) ∈ Set.Icc (-1 : ℝ) 1 ∧
    Real.arcsin x + Real.arcsin (3 * x) = π / 2 ∧
    x = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_equation_solution_l1072_107237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_self_describing_sequences_l1072_107250

def SelfDescribingSequence (x : List ℕ) : Prop :=
  ∀ j, j < x.length → x.get ⟨j, by sorry⟩ = (x.filter (· = j)).length

theorem self_describing_sequences :
  ∀ x : List ℕ, SelfDescribingSequence x →
    (x = [2, 0, 2, 0] ∨
     x = [1, 2, 1, 0] ∨
     x = [2, 1, 2, 0, 0] ∨
     ∃ p : ℕ, p ≥ 3 ∧ x = p :: 2 :: 1 :: (List.replicate (p - 3) 0 ++ [1, 0, 0, 0])) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_self_describing_sequences_l1072_107250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_equals_sum_of_squares_of_first_four_divisors_l1072_107247

def is_divisor (d n : ℕ) : Bool := n % d = 0

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d => is_divisor d n)

theorem unique_n_equals_sum_of_squares_of_first_four_divisors :
  ∃! n : ℕ,
    n > 0 ∧
    (divisors n).length ≥ 4 ∧
    let d := divisors n
    n = (d.get! 0) ^ 2 + (d.get! 1) ^ 2 + (d.get! 2) ^ 2 + (d.get! 3) ^ 2 ∧
    n = 130 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_equals_sum_of_squares_of_first_four_divisors_l1072_107247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l1072_107296

theorem trig_simplification (x y : ℝ) : 
  (Real.cos x)^2 + (Real.cos (x - y))^2 - 2 * (Real.cos x) * (Real.cos y) * (Real.cos (x - y)) = (Real.sin x)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l1072_107296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_problem_l1072_107283

theorem power_of_two_problem (y : ℝ) (h : (2:ℝ)^(2*y) = 64) : (2:ℝ)^(-y) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_problem_l1072_107283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1072_107275

noncomputable def f (x : ℝ) : ℝ := 1 / ⌊x^2 - 9*x + 20⌋

theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | x < 4 ∨ (4 < x ∧ x < 5) ∨ 5 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1072_107275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_cube_volume_ratio_and_sum_l1072_107202

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A cube whose vertices are the centers of the faces of a regular dodecahedron -/
structure CenterFaceCube where
  dodecahedron : RegularDodecahedron

/-- The volume of a regular dodecahedron -/
noncomputable def volume_dodecahedron (d : RegularDodecahedron) : ℝ :=
  (15 + 7 * Real.sqrt 5) * d.side_length ^ 3 / 4

/-- The volume of a center face cube -/
noncomputable def volume_center_face_cube (c : CenterFaceCube) : ℝ :=
  c.dodecahedron.side_length ^ 3

/-- The ratio of volumes between a regular dodecahedron and its center face cube -/
noncomputable def volume_ratio (d : RegularDodecahedron) (c : CenterFaceCube) : ℝ :=
  volume_dodecahedron d / volume_center_face_cube c

theorem dodecahedron_cube_volume_ratio_and_sum 
  (d : RegularDodecahedron) (c : CenterFaceCube) : 
  volume_ratio d c = (15 + 7 * Real.sqrt 5) / 4 ∧ 
  (15 + 7 * Real.sqrt 5 + 4 : ℝ) = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_cube_volume_ratio_and_sum_l1072_107202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_on_foot_l1072_107287

/-- Given a journey with the specified conditions, prove the distance traveled on foot -/
theorem distance_on_foot (total_journey : ℝ) (rail_fraction : ℝ) (bus_fraction : ℝ)
  (h_total : total_journey = 130)
  (h_rail : rail_fraction = 3/5)
  (h_bus : bus_fraction = 17/20)
  : total_journey * (1 - rail_fraction) * (1 - bus_fraction) = 7.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_on_foot_l1072_107287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_less_than_one_std_dev_above_mean_l1072_107226

-- Define a symmetric distribution
structure SymmetricDistribution (μ σ : ℝ) where
  pdf : ℝ → ℝ
  symmetric : ∀ x, pdf (μ + x) = pdf (μ - x)

-- Define the property that 68% of the distribution lies within one standard deviation of the mean
def WithinOneStdDev (μ σ : ℝ) (d : SymmetricDistribution μ σ) : Prop :=
  ∃ (p : ℝ), p = 0.68 ∧ sorry

-- Theorem statement
theorem distribution_less_than_one_std_dev_above_mean
  (μ σ : ℝ) (d : SymmetricDistribution μ σ) (h : WithinOneStdDev μ σ d) :
  ∃ (p : ℝ), p = 0.84 ∧ sorry := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_less_than_one_std_dev_above_mean_l1072_107226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_difference_l1072_107240

noncomputable def promotion_A (price : ℝ) : ℝ := price + price / 2

noncomputable def promotion_B (price : ℝ) : ℝ := price + (price - 10)

theorem savings_difference (shoe_price : ℝ) (h : shoe_price = 30) :
  promotion_B shoe_price - promotion_A shoe_price = 5 := by
  simp [promotion_A, promotion_B, h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_difference_l1072_107240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisected_segment_length_l1072_107259

/-- Predicate stating that E and F trisect the line segment AG -/
def trisects (E F A G : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A E = dist E F ∧ dist E F = dist F G

/-- Definition of midpoint -/
def is_midpoint (N A G : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist N A = dist N G ∧ 2 * dist N A = dist A G

/-- Given a line segment AG with points E and F trisecting it, and N as its midpoint,
    prove that if NF = 10, then the length of AG is 30. -/
theorem trisected_segment_length (A G E F N : EuclideanSpace ℝ (Fin 2))
    (h1 : trisects E F A G)
    (h2 : is_midpoint N A G)
    (h3 : dist N F = 10) : dist A G = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisected_segment_length_l1072_107259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1072_107282

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 4)

theorem f_properties :
  -- The smallest positive period is π/2
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (S : ℝ), S > 0 → (∀ (x : ℝ), f (x + S) = f x) → T ≤ S)) ∧
  -- The smallest positive period is exactly π/2
  (∀ (x : ℝ), f (x + Real.pi / 2) = f x) ∧
  -- f(π/3) = 2√3
  f (Real.pi / 3) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1072_107282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_max_sum_l1072_107268

theorem min_value_of_max_sum (a b c d e f g : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → e ≥ 0 → f ≥ 0 → g ≥ 0 →
  a + b + c + d + e + f + g = 1 →
  let M := max (a + b + c) (max (b + c + d) (max (c + d + e) (max (d + e + f) (e + f + g))))
  ∀ ε > 0, M ≥ 1/3 - ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_max_sum_l1072_107268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sum_is_seven_l1072_107262

/-- The number of marbles in the bag -/
def n : ℕ := 6

/-- The set of marble numbers -/
def marbles : Finset ℕ := Finset.range n

/-- The set of all pairs of different marbles -/
def marblePairs : Finset (ℕ × ℕ) :=
  (marbles.product marbles).filter (fun p => p.1 < p.2)

/-- The sum of a pair of marbles -/
def pairSum (p : ℕ × ℕ) : ℕ := p.1 + p.2

/-- The expected value of the sum of two randomly drawn marbles -/
noncomputable def expectedSum : ℚ :=
  (marblePairs.sum (fun p => (pairSum p : ℚ)) : ℚ) / marblePairs.card

theorem expected_sum_is_seven :
  expectedSum = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sum_is_seven_l1072_107262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hike_time_calculation_l1072_107272

/-- Calculates the total time of a round-trip hike given the distance, pace to destination, and pace returning. -/
noncomputable def hikeTime (distance : ℝ) (paceTo : ℝ) (paceFrom : ℝ) : ℝ :=
  distance / paceTo + distance / paceFrom

/-- Theorem: Given a hike with specified conditions, the total time is 5 hours. -/
theorem hike_time_calculation :
  let distance : ℝ := 12
  let paceTo : ℝ := 4
  let paceFrom : ℝ := 6
  hikeTime distance paceTo paceFrom = 5 := by
  -- Unfold the definition of hikeTime
  unfold hikeTime
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hike_time_calculation_l1072_107272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_ratio_theorem_l1072_107234

/-- An octagon with specific properties -/
structure Octagon where
  area : ℝ
  bisector_line_ratio : ℝ
  lower_square_area : ℝ
  lower_triangle_base : ℝ

/-- The theorem statement -/
theorem octagon_ratio_theorem (O : Octagon) 
  (h1 : O.area = 10)
  (h2 : O.bisector_line_ratio = 1/2)
  (h3 : O.lower_square_area = 1)
  (h4 : O.lower_triangle_base = 5) :
  ∃ (x y : ℝ), x + y = 5 ∧ x / y = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_ratio_theorem_l1072_107234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_pythagorean_triple_SL2Z_no_fourth_power_triple_SL2Z_l1072_107235

-- Define SL₂(ℤ)
def SL2Z : Type := {A : Matrix (Fin 2) (Fin 2) ℤ // Matrix.det A = 1}

-- Statement for A² + B² = C²
theorem no_pythagorean_triple_SL2Z :
  ¬ ∃ (A B C : SL2Z), A.1 ^ 2 + B.1 ^ 2 = C.1 ^ 2 := by sorry

-- Statement for A⁴ + B⁴ = C⁴
theorem no_fourth_power_triple_SL2Z :
  ¬ ∃ (A B C : SL2Z), A.1 ^ 4 + B.1 ^ 4 = C.1 ^ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_pythagorean_triple_SL2Z_no_fourth_power_triple_SL2Z_l1072_107235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_GAB_l1072_107209

/-- Curve C in rectangular coordinates -/
def curve_C (x y : ℝ) : Prop := y^2 = 8*x

/-- Line l passing through (2, 0) and (0, -2) -/
def line_l (x y : ℝ) : Prop := y = x - 2

/-- Point G -/
def point_G : ℝ × ℝ := (-2, 0)

theorem area_triangle_GAB :
  ∃ (A B : ℝ × ℝ),
    curve_C A.1 A.2 ∧
    curve_C B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    (let S := abs ((point_G.1 - A.1) * (B.2 - A.2) - (point_G.2 - A.2) * (B.1 - A.1)) / 2;
     S = 16 * Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_GAB_l1072_107209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l1072_107204

theorem complex_absolute_value :
  let z : ℂ := (2 + Complex.I) / Complex.I
  Complex.abs (z + 1) = 2 * Real.sqrt 2 := by
  -- Unfold the definition of z
  let z := (2 + Complex.I) / Complex.I
  -- Simplify z
  have h1 : z = 1 - 2 * Complex.I := by sorry
  -- Calculate |z + 1|
  have h2 : Complex.abs (z + 1) = Complex.abs (2 - 2 * Complex.I) := by sorry
  -- Use the definition of complex absolute value
  have h3 : Complex.abs (2 - 2 * Complex.I) = Real.sqrt (2^2 + (-2)^2) := by sorry
  -- Simplify
  have h4 : Real.sqrt (2^2 + (-2)^2) = 2 * Real.sqrt 2 := by sorry
  -- Combine the steps
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l1072_107204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_quiz_show_theorem_l1072_107274

/-- Represents the Olympic knowledge quiz show game -/
structure OlympicQuizShow where
  correct_prob : ℝ  -- Probability of answering a question correctly
  max_questions : ℕ -- Maximum number of questions
  required_correct : ℕ -- Number of correct answers required to enter final
  required_incorrect : ℕ -- Number of incorrect answers to be eliminated

/-- Represents the number of questions answered by a contestant -/
def QuestionCount : Type := Fin 3

/-- Distribution of the number of questions answered -/
def distribution (game : OlympicQuizShow) : QuestionCount → ℝ := sorry

/-- Expected value of the number of questions answered -/
def expected_value (game : OlympicQuizShow) : ℝ := sorry

/-- Helper function to calculate the probability of entering the final round -/
def probability_of_entering_final (game : OlympicQuizShow) : ℝ := sorry

/-- Main theorem about the Olympic quiz show -/
theorem olympic_quiz_show_theorem (game : OlympicQuizShow) 
  (h1 : game.correct_prob = 2/3)
  (h2 : game.max_questions = 5)
  (h3 : game.required_correct = 3)
  (h4 : game.required_incorrect = 3) :
  let final_prob := 64/81
  let dist := distribution game
  let exp_val := expected_value game
  (final_prob = probability_of_entering_final game) ∧
  (dist ⟨0, by norm_num⟩ = 1/3 ∧ dist ⟨1, by norm_num⟩ = 10/27 ∧ dist ⟨2, by norm_num⟩ = 8/27) ∧
  (exp_val = 107/27) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_quiz_show_theorem_l1072_107274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1072_107203

/-- Calculates the length of a train given its speed and the time and distance it takes to cross a bridge. -/
noncomputable def train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_ms := train_speed * (1000 / 3600)  -- Convert km/h to m/s
  speed_ms * crossing_time - bridge_length

/-- Theorem stating that a train with speed 96 km/h crossing a 300-meter bridge in 15 seconds has a length of approximately 100.05 meters. -/
theorem train_length_calculation :
  let train_speed : ℝ := 96  -- km/h
  let bridge_length : ℝ := 300  -- meters
  let crossing_time : ℝ := 15  -- seconds
  abs (train_length train_speed bridge_length crossing_time - 100.05) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1072_107203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l1072_107242

theorem order_of_numbers : Real.sqrt 3 > (1/2)^3 ∧ (1/2)^3 > Real.log 3 / Real.log (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l1072_107242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_10_choose_3_l1072_107228

theorem binomial_10_choose_3 : (Nat.choose 10 3) = 120 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_10_choose_3_l1072_107228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_function_l1072_107258

theorem min_value_exponential_function (x : ℝ) :
  (2 : ℝ)^(2*x) - 5 * (2 : ℝ)^x + 6 ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_function_l1072_107258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_negative_2023_l1072_107291

theorem opposite_of_negative_2023 : 
  (-(2023 : ℤ)).neg = (2023 : ℤ) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_negative_2023_l1072_107291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AP_distance_l1072_107281

-- Define the cube and points
variable (A B C D G P : EuclideanSpace ℝ (Fin 3))

-- Define the cube properties
def IsCube (A B C D : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- AG is a diagonal through the center of the cube
def IsDiagonal (A G : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Axioms for the cube and diagonal
axiom is_cube : IsCube A B C D
axiom AG_diagonal : IsDiagonal A G

-- Define the distances
axiom BP_dist : ‖B - P‖ = 60 * Real.sqrt 10
axiom CP_dist : ‖C - P‖ = 60 * Real.sqrt 5
axiom DP_dist : ‖D - P‖ = 120 * Real.sqrt 2
axiom GP_dist : ‖G - P‖ = 36 * Real.sqrt 7

-- Theorem to prove
theorem AP_distance : ‖A - P‖ = 192 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AP_distance_l1072_107281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_beta_properties_l1072_107222

theorem angle_beta_properties (β : Real) 
  (h1 : π/2 < β ∧ β < π) -- β is in the second quadrant
  (h2 : 2 * Real.tan β^2 / (3 * Real.tan β + 2) = 1) : -- β satisfies the equation
  (Real.sin (β + 3*π/2) = 2*Real.sqrt 5/5) ∧ 
  ((2/3) * Real.sin β^2 + Real.cos β * Real.sin β = -1/15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_beta_properties_l1072_107222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transformation_l1072_107248

-- Define the number of piles
def num_piles : ℕ := 2023

-- Define a function to represent the i-th prime number
noncomputable def nth_prime (i : ℕ) : ℕ := sorry

-- Define the initial configuration of piles
noncomputable def initial_piles : Fin num_piles → ℕ := λ i => nth_prime i.val.succ

-- Define the operations
def split_pile (piles : Fin num_piles → ℕ) (i : Fin num_piles) (j k : ℕ) : Fin num_piles → ℕ := sorry

def merge_piles (piles : Fin num_piles → ℕ) (i j : Fin num_piles) : Fin num_piles → ℕ := sorry

-- Define the target configuration
def target_piles : Fin num_piles → ℕ := λ _ => num_piles

-- The main theorem
theorem impossible_transformation :
  ¬ ∃ (sequence : List (Fin num_piles × Fin num_piles)),
    ∃ (final_piles : Fin num_piles → ℕ),
    (sequence.foldl (λ piles (i, j) => merge_piles piles i j) initial_piles) = final_piles ∧
    final_piles = target_piles :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transformation_l1072_107248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_class_triangle_has_45_degree_angle_l1072_107260

/-- A triangle ABC is a V-class triangle if there exists a triangle A₁B₁C₁ such that
    cos A / sin A₁ = cos B / sin B₁ = cos C / sin C₁ = 1 --/
def is_v_class_triangle (A B C A₁ B₁ C₁ : ℝ) : Prop :=
  Real.cos A / Real.sin A₁ = 1 ∧ Real.cos B / Real.sin B₁ = 1 ∧ Real.cos C / Real.sin C₁ = 1

/-- The sum of angles in a triangle is π --/
axiom angle_sum (A B C : ℝ) : A + B + C = Real.pi

theorem v_class_triangle_has_45_degree_angle 
  (A B C A₁ B₁ C₁ : ℝ) 
  (h_v_class : is_v_class_triangle A B C A₁ B₁ C₁) : 
  A = Real.pi/4 ∨ B = Real.pi/4 ∨ C = Real.pi/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_class_triangle_has_45_degree_angle_l1072_107260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1072_107277

/-- Set of digits that are allowed in the numbers -/
def allowedDigits : Finset Nat := {0, 3, 5, 6, 7, 9}

/-- Check if a number contains only allowed digits -/
def hasOnlyAllowedDigits (n : Nat) : Bool :=
  n.digits 10 |>.all (·∈ allowedDigits)

/-- The set of numbers from 1 to 999 with only allowed digits -/
def validNumbers : Finset Nat :=
  Finset.filter (fun n => n ≥ 1 ∧ n ≤ 999 ∧ hasOnlyAllowedDigits n) (Finset.range 1000)

/-- The main theorem stating that there are 215 valid numbers -/
theorem count_valid_numbers : validNumbers.card = 215 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1072_107277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_at_origin_l1072_107254

noncomputable def curve (x : ℝ) : ℝ := x^3 - Real.sqrt 3 * x + 1

noncomputable def tangent_slope (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (deriv f) x

noncomputable def slope_angle (m : ℝ) : ℝ :=
  Real.arctan (-m) * (180 / Real.pi)

theorem tangent_slope_angle_at_origin :
  slope_angle (tangent_slope curve 0) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_at_origin_l1072_107254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_fraction_parts_l1072_107263

/-- Represents a repeating decimal with a three-digit repetend -/
def RepeatingDecimal (a b c : Nat) : ℚ :=
  (a * 100 + b * 10 + c : ℚ) / 999

/-- The fraction representation of 0.036̅ -/
def x : ℚ := RepeatingDecimal 0 3 6

theorem product_of_fraction_parts : ∃ (n d : ℕ), x = n / d ∧ Nat.Coprime n d ∧ n * d = 444 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_fraction_parts_l1072_107263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_vertical_shift_shifted_line_equation_l1072_107297

/-- Given a line y = mx + b, shifting it vertically by k units results in y = mx + (b + k) -/
theorem line_vertical_shift (m b k : ℝ) : 
  (fun x : ℝ => m * x + (b + k)) = (fun x : ℝ => m * x + b + k) := by
  funext x
  ring

/-- The line y = -2x - 1 is obtained by shifting y = -2x - 4 upward by 3 units -/
theorem shifted_line_equation : 
  (fun x : ℝ => -2 * x - 1) = (fun x : ℝ => -2 * x - 4 + 3) := by
  funext x
  ring

#check shifted_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_vertical_shift_shifted_line_equation_l1072_107297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_kept_80_l1072_107265

/-- The amount Tim won in the raffle -/
noncomputable def raffle_win : ℚ := 100

/-- The percentage Tim gave away to his friend -/
noncomputable def percentage_given : ℚ := 20

/-- The amount Tim kept after giving away a percentage to his friend -/
noncomputable def amount_kept : ℚ := raffle_win - (percentage_given / 100) * raffle_win

/-- Theorem stating that the amount Tim kept is $80 -/
theorem tim_kept_80 : amount_kept = 80 := by
  -- Unfold the definitions
  unfold amount_kept raffle_win percentage_given
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_kept_80_l1072_107265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1072_107251

noncomputable def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 2

def line_equation (x y : ℝ) : Prop := x - y + 3 = 0

noncomputable def distance_point_line (x₀ y₀ a b c : ℝ) : ℝ := 
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

theorem distance_circle_center_to_line : 
  distance_point_line (-2) 2 1 (-1) 3 = (3 * Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1072_107251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l1072_107214

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + 2*y + 3*z = 5)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -2)
  (z_ge : z ≥ -3) :
  (Real.sqrt (x + 1) + Real.sqrt (2*y + 4) + Real.sqrt (3*z + 9) ≤ Real.sqrt 57) ∧ 
  (∃ x₀ y₀ z₀ : ℝ, x₀ + 2*y₀ + 3*z₀ = 5 ∧ 
    x₀ ≥ -1 ∧ y₀ ≥ -2 ∧ z₀ ≥ -3 ∧
    Real.sqrt (x₀ + 1) + Real.sqrt (2*y₀ + 4) + Real.sqrt (3*z₀ + 9) = Real.sqrt 57) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l1072_107214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_cycle_finite_long_cycle_exists_l1072_107241

/-- A permutation of n elements -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Application of a permutation k times -/
def applyKTimes (p : Permutation n) (k : ℕ) : Permutation n :=
  match k with
  | 0 => id
  | k + 1 => p ∘ (applyKTimes p k)

/-- Theorem: For any permutation, there exists a finite number of applications that returns all elements to their original positions -/
theorem permutation_cycle_finite (n : ℕ) (p : Permutation n) : 
  ∃ k : ℕ, k > 0 ∧ applyKTimes p k = id := by
  sorry

/-- For n = 98, there exists a permutation with a cycle longer than 300,000 years (assuming 365.25 days per year) -/
theorem long_cycle_exists : 
  ∃ (p : Permutation 98), ∀ k : ℕ, k ≤ (300000 : ℕ) * 365 → applyKTimes p k ≠ id := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_cycle_finite_long_cycle_exists_l1072_107241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_six_in_rolls_l1072_107299

theorem no_six_in_rolls (rolls : Finset ℕ) (avg : ℚ) (var : ℚ) :
  rolls.card = 5 →
  avg = 2 →
  var = 31/10 →
  (rolls.sum (λ x => (x : ℚ))) / rolls.card = avg →
  (rolls.sum (λ x => ((x : ℚ) - avg) ^ 2)) / rolls.card = var →
  ∀ x ∈ rolls, x ≠ 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_six_in_rolls_l1072_107299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_sum_l1072_107205

/-- Three points in 3D space are collinear if they all lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), p2 = p1 + t₁ • (p3 - p1) ∧
                  p3 = p1 + t₂ • (p3 - p1)

/-- Given three collinear points (2, a, b), (a, 3, b), and (a, b, 4), prove that a + b = 7. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_sum_l1072_107205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_45deg_l1072_107294

/-- The area of a figure formed by rotating a semicircle around one of its ends -/
noncomputable def rotated_semicircle_area (R : ℝ) (α : ℝ) : ℝ :=
  (Real.pi * R^2) / 2

/-- Theorem: The area of a figure formed by rotating a semicircle of radius R 
    around one of its ends by 45° is equal to πR²/2 -/
theorem rotated_semicircle_area_45deg (R : ℝ) (h : R > 0) :
  rotated_semicircle_area R (Real.pi/4) = (Real.pi * R^2) / 2 := by
  sorry

#check rotated_semicircle_area_45deg

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_45deg_l1072_107294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gunther_cleaning_time_l1072_107290

/-- Gunther's apartment cleaning problem -/
theorem gunther_cleaning_time (
  free_time vacuum_time dusting_time mopping_time bathroom_time window_time cat_brush_time num_cats : ℕ
) : free_time = 265 ∧
    vacuum_time = 45 ∧
    dusting_time = 60 ∧
    mopping_time = 30 ∧
    bathroom_time = 40 ∧
    window_time = 15 ∧
    cat_brush_time = 5 ∧
    num_cats = 4 →
  free_time - (vacuum_time + dusting_time + mopping_time + bathroom_time + window_time + cat_brush_time * num_cats) = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gunther_cleaning_time_l1072_107290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1072_107270

-- Define the function y = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- Define the slope of the line y = -2x - 4
def m : ℝ := -2

-- Define the point of tangency
def x₀ : ℝ := 1
noncomputable def y₀ : ℝ := f x₀

-- State the theorem
theorem tangent_line_equation :
  ∃ (a b c : ℝ), 
    (∀ x, f x = Real.sqrt x) →
    (a * x₀ + b * y₀ + c = 0) ∧
    (a * 1 + b * (-m) = 0) ∧
    (a = 1 ∧ b = -2 ∧ c = 1) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1072_107270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_omelette_has_four_eggs_l1072_107284

/-- Represents the number of eggs in a larger omelette -/
def larger_omelette_eggs : ℕ → Prop := sorry

/-- Represents the total number of eggs used -/
def total_eggs : ℕ := 84

/-- Represents the number of 3-egg omelettes ordered -/
def three_egg_omelettes : ℕ := 5 + 3

/-- Represents the number of larger omelettes ordered -/
def larger_omelettes : ℕ := 7 + 8

/-- Represents the number of eggs used for 3-egg omelettes -/
def eggs_for_three_egg : ℕ := three_egg_omelettes * 3

/-- Represents the number of eggs used for larger omelettes -/
def eggs_for_larger : ℕ := total_eggs - eggs_for_three_egg

theorem larger_omelette_has_four_eggs : larger_omelette_eggs 4 :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_omelette_has_four_eggs_l1072_107284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l1072_107210

/-- An inverse proportion function with parameter k -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := (k - 2) / x

/-- Predicate to check if a point (x, y) is in the second or fourth quadrant -/
def in_second_or_fourth_quadrant (x y : ℝ) : Prop := (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)

/-- Theorem stating the properties of the inverse proportion function -/
theorem inverse_proportion_properties (k : ℝ) :
  (∀ x ≠ 0, in_second_or_fourth_quadrant x (inverse_proportion k x)) →
  (k < 2 ∧
   ∀ y₁ y₂ : ℝ, 
     inverse_proportion k (-4) = y₁ → 
     inverse_proportion k (-1) = y₂ → 
     y₁ < y₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l1072_107210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1072_107232

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then
    2 * x - x^2
  else if -4 ≤ x ∧ x < 0 then
    x^2 + 6 * x
  else
    0  -- This value doesn't matter as it's outside our domain of interest

-- State the theorem about the value range of f
theorem f_range : 
  (∀ y ∈ Set.range f, -9 ≤ y ∧ y ≤ 1) ∧ 
  (∃ x₁ ∈ Set.Icc (-4 : ℝ) 2, f x₁ = -9) ∧
  (∃ x₂ ∈ Set.Icc (-4 : ℝ) 2, f x₂ = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1072_107232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_polynomial_properties_l1072_107286

/-- A polynomial is reciprocal if its coefficients are symmetric. -/
def IsReciprocal (p : Polynomial ℚ) : Prop :=
  ∀ i, p.coeff i = p.coeff (p.natDegree - i)

/-- The theorem statement. -/
theorem reciprocal_polynomial_properties
  (n : ℕ)
  (hn : Odd n)
  (P : Polynomial ℚ)
  (hP : P.natDegree = n)
  (hrecip : IsReciprocal P) :
  P.eval (-1) = 0 ∧
  ∃ Q : Polynomial ℚ, Q.natDegree = n - 1 ∧ IsReciprocal Q ∧ P = (X + 1) * Q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_polynomial_properties_l1072_107286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_value_l1072_107208

theorem sin_double_angle_special_value (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin (2 * α) = (2 * Real.sqrt 3 / 3) * Real.sin α) : 
  Real.sin (2 * α) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_value_l1072_107208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_odd_solution_equivalence_sum_of_inverse_image_points_sine_odd_is_odd_l1072_107230

def SineOdd (g : ℝ → ℝ) : Prop := ∀ x, Real.sin (g (-x)) = -Real.sin (g x)

def MonoIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

def SurjectiveToReal (f : ℝ → ℝ) : Prop := ∀ y, ∃ x, f x = y

theorem sine_odd_solution_equivalence (g : ℝ → ℝ) (h : SineOdd g) :
  ∀ u₀, (Real.sin (g u₀) = 1 ↔ Real.sin (g (-u₀)) = -1) := by sorry

theorem sum_of_inverse_image_points (f : ℝ → ℝ) (h₁ : MonoIncreasing f) (h₂ : SineOdd f) 
  (h₃ : SurjectiveToReal f) (h₄ : f 0 = 0) (a b : ℝ) (ha : f a = Real.pi/2) (hb : f b = -Real.pi/2) :
  a + b = 0 := by sorry

theorem sine_odd_is_odd (f : ℝ → ℝ) (h₁ : MonoIncreasing f) (h₂ : SineOdd f) 
  (h₃ : SurjectiveToReal f) (h₄ : f 0 = 0) :
  ∀ x, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_odd_solution_equivalence_sum_of_inverse_image_points_sine_odd_is_odd_l1072_107230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eddie_age_l1072_107201

-- Define the ages as natural numbers
variable (Eddie Becky Irene : ℕ)

-- Define the relationships between ages
def becky_four_times_younger (Eddie Becky : ℕ) : Prop := Eddie = 4 * Becky
def irene_twice_becky (Irene Becky : ℕ) : Prop := Irene = 2 * Becky
def irene_age (Irene : ℕ) : Prop := Irene = 46

-- Theorem to prove Eddie's age
theorem eddie_age :
  becky_four_times_younger Eddie Becky →
  irene_twice_becky Irene Becky →
  irene_age Irene →
  Eddie = 92 := by
  intro h1 h2 h3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eddie_age_l1072_107201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_example_function_satisfies_property_general_solution_satisfies_property_l1072_107253

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f y + 2 * x) = x + f x + f y

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesProperty f → ∃ a : ℝ, ∀ x : ℝ, f x = x + a := by
  sorry

-- Example of a function satisfying the property
def example_function (a : ℝ) : ℝ → ℝ := fun x ↦ x + a

-- Prove that the example function satisfies the property
theorem example_function_satisfies_property (a : ℝ) :
  SatisfiesProperty (example_function a) := by
  intro x y
  simp [example_function]
  ring

-- Prove that any function of the form f(x) = x + a satisfies the property
theorem general_solution_satisfies_property (a : ℝ) :
  SatisfiesProperty (fun x ↦ x + a) := by
  intro x y
  simp
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_example_function_satisfies_property_general_solution_satisfies_property_l1072_107253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_notebook_price_budget_constraint_satisfied_l1072_107269

/-- The greatest possible whole-dollar price per notebook --/
def greatest_price (budget : ℕ) (num_notebooks : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) : ℕ :=
  let available := budget - entrance_fee
  let max_spend := (available : ℚ) / (1 + tax_rate)
  (max_spend / num_notebooks).floor.toNat

/-- Theorem stating the greatest possible price per notebook --/
theorem greatest_notebook_price :
  greatest_price 160 18 5 (5/100) = 8 := by
  sorry

/-- Verify that the solution satisfies the budget constraint --/
theorem budget_constraint_satisfied :
  let price := greatest_price 160 18 5 (5/100)
  ((18 * price : ℚ) * (1 + 5/100) + 5 : ℚ) ≤ 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_notebook_price_budget_constraint_satisfied_l1072_107269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_counterfeit_in_two_weighings_l1072_107264

/-- Represents a coin, which can be either genuine or counterfeit -/
inductive Coin
| genuine
| counterfeit

/-- Represents the result of a weighing -/
inductive WeighResult
| left_heavier
| right_heavier
| balanced

/-- A weighing function that compares two lists of coins -/
def weigh (left : List Coin) (right : List Coin) : WeighResult :=
  sorry

/-- A strategy function that determines which coins to weigh next based on previous results -/
def strategy (coins : List Coin) (first_weigh : WeighResult) : (List Coin × List Coin) :=
  sorry

/-- The main theorem stating that the counterfeit coin can be identified in two weighings -/
theorem identify_counterfeit_in_two_weighings 
  (coins : List Coin) 
  (h_count : coins.length = 8) 
  (h_one_counterfeit : ∃! c, c ∈ coins ∧ c = Coin.counterfeit) :
  ∃ (first_weigh : List Coin × List Coin) 
    (second_weigh : WeighResult → List Coin × List Coin),
    ∀ (result1 : WeighResult) (result2 : WeighResult),
    ∃! c, c ∈ coins ∧ c = Coin.counterfeit ∧ 
      (weigh (first_weigh.1) (first_weigh.2) = result1) ∧
      (weigh ((second_weigh result1).1) ((second_weigh result1).2) = result2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_counterfeit_in_two_weighings_l1072_107264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1072_107243

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := Set.Icc (-1) 2

-- Theorem statement
theorem range_of_f :
  Set.range (fun x ↦ f x) ∩ (Set.image f domain) = Set.Icc 2 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1072_107243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninety_eighth_term_is_27_l1072_107239

def next_term (n : ℕ) : ℕ :=
  if n < 10 then n * 9
  else if n % 2 = 0 then n / 2
  else n - 5

def sequence_term (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => next_term (sequence_term start n)

theorem ninety_eighth_term_is_27 :
  sequence_term 98 97 = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninety_eighth_term_is_27_l1072_107239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_eleven_is_zero_l1072_107225

/-- A three-digit palindrome is a number between 100 and 999 of the form aba where a and b are single digits and a ≠ 0 -/
def ThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 100 * a + 10 * b + a

/-- The set of all three-digit palindromes -/
def ThreeDigitPalindromeSet : Set ℕ :=
  {n : ℕ | ThreeDigitPalindrome n}

/-- The set of three-digit palindromes divisible by 11 -/
def DivisibleByElevenSet : Set ℕ :=
  {n ∈ ThreeDigitPalindromeSet | n % 11 = 0}

/-- The number of three-digit palindromes -/
def NumThreeDigitPalindromes : ℕ := 90

/-- The number of three-digit palindromes divisible by 11 -/
def NumDivisibleByEleven : ℕ := 0

/-- The probability of a randomly chosen three-digit palindrome being divisible by 11 -/
def ProbabilityDivisibleByEleven : ℚ :=
  (NumDivisibleByEleven : ℚ) / (NumThreeDigitPalindromes : ℚ)

theorem probability_divisible_by_eleven_is_zero :
  ProbabilityDivisibleByEleven = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_eleven_is_zero_l1072_107225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_line_relations_l1072_107271

structure Plane where

structure Line where

axiom perpendicular : Plane → Plane → Prop

axiom intersects : Plane → Plane → Prop

axiom not_perpendicular : Plane → Plane → Prop

axiom contains : Plane → Line → Prop

axiom parallel : Line → Plane → Prop

axiom perpendicular_line : Line → Plane → Prop

theorem plane_line_relations 
  (α β γ : Plane) 
  (h1 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h2 : perpendicular β γ)
  (h3 : intersects α γ)
  (h4 : not_perpendicular α γ) :
  (∃ a : Line, contains α a ∧ parallel a γ) ∧
  (∃ c : Line, contains γ c ∧ perpendicular_line c β) ∧
  (¬ ∀ b : Line, contains β b → perpendicular_line b γ) ∧
  (¬ ∀ b : Line, contains β b → parallel b γ) ∧
  (¬ ∃ a : Line, contains α a ∧ perpendicular_line a γ) ∧
  (¬ ∀ c : Line, contains γ c → parallel c α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_line_relations_l1072_107271
