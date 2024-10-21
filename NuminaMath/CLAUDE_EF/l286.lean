import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_series_sum_9_177_8_l286_28666

noncomputable def arithmetic_series_sum (a₁ : ℝ) (aₙ : ℝ) (d : ℝ) : ℝ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_9_177_8 :
  arithmetic_series_sum 9 177 8 = 2046 := by
  unfold arithmetic_series_sum
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_series_sum_9_177_8_l286_28666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_volume_ratio_l286_28611

/-- Represents a pyramid with a square base -/
structure SquarePyramid where
  baseEdge : ℝ
  height : ℝ

/-- Calculate the volume of a pyramid given its base area and height -/
noncomputable def pyramidVolume (baseArea : ℝ) (height : ℝ) : ℝ := (1 / 3) * baseArea * height

/-- Calculate the volume of a smaller similar pyramid given the original volume and the ratio of heights -/
noncomputable def smallerPyramidVolume (originalVolume : ℝ) (heightRatio : ℝ) : ℝ := originalVolume * heightRatio^3

/-- The main theorem stating the volume ratio of the remaining solid -/
theorem remaining_volume_ratio (p : SquarePyramid) 
  (h1 : p.baseEdge = 20)
  (h2 : p.height = 40) : 
  let originalVolume := pyramidVolume (p.baseEdge^2) p.height
  let smallerVolume1 := smallerPyramidVolume originalVolume (1/3)
  let smallerVolume2 := smallerPyramidVolume originalVolume (1/5)
  let remainingVolume := originalVolume - (smallerVolume1 + smallerVolume2)
  remainingVolume / originalVolume = 3223 / 3375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_volume_ratio_l286_28611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_parallel_lines_a_distance_parallel_lines_l286_28629

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 1 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a - 2) * y + a = 0

-- Define perpendicularity of lines
def perpendicular (f g : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ a : ℝ, ∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ, 
    f a x₁ y₁ → f a x₂ y₂ → x₁ ≠ x₂ →
    g a x₃ y₃ → g a x₄ y₄ → x₃ ≠ x₄ →
    (y₂ - y₁) / (x₂ - x₁) * (y₄ - y₃) / (x₄ - x₃) = -1

-- Define parallel lines
def parallel (f g : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ a : ℝ, ∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ, 
    f a x₁ y₁ → f a x₂ y₂ → x₁ ≠ x₂ →
    g a x₃ y₃ → g a x₄ y₄ → x₃ ≠ x₄ →
    (y₂ - y₁) / (x₂ - x₁) = (y₄ - y₃) / (x₄ - x₃)

-- Define the distance between parallel lines
noncomputable def distance (f g : ℝ → ℝ → ℝ → Prop) (a : ℝ) : ℝ :=
  |1 - 9| / Real.sqrt (3^2 + 3^2)

-- Theorem statements
theorem perpendicular_lines_a (a : ℝ) :
  perpendicular l₁ l₂ → a = 3/2 := sorry

theorem parallel_lines_a (a : ℝ) :
  parallel l₁ l₂ → a = 3 := sorry

theorem distance_parallel_lines (a : ℝ) :
  parallel l₁ l₂ → distance l₁ l₂ a = 4 * Real.sqrt 2 / 3 := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_parallel_lines_a_distance_parallel_lines_l286_28629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_l286_28648

theorem sine_symmetry (φ : ℝ) (h1 : 0 < φ) (h2 : φ ≤ π) : 
  (∀ x, Real.sin (2*x + φ) = Real.sin (2*(π/4 - x) + φ)) → φ = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_l286_28648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l286_28690

theorem gcd_problem (x : ℤ) (h : 19845 ∣ x) :
  Int.gcd ((3 * x + 5) * (7 * x + 2) * (13 * x + 7) * (2 * x + 10)) x = 700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l286_28690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l286_28698

/-- The ellipse E is defined by x²/1 + y²/b² = 1 where 0 < b < 1 -/
def Ellipse (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 / b^2 = 1}

/-- F₁ and F₂ are the left and right foci of the ellipse -/
noncomputable def LeftFocus (b : ℝ) : ℝ × ℝ := (-Real.sqrt (1 - b^2), 0)
noncomputable def RightFocus (b : ℝ) : ℝ × ℝ := (Real.sqrt (1 - b^2), 0)

/-- A line passing through F₁ intersects E at points A and B -/
def IntersectionPoints (b : ℝ) (A B : ℝ × ℝ) : Prop :=
  A ∈ Ellipse b ∧ B ∈ Ellipse b ∧ ∃ (t : ℝ), A = LeftFocus b + t • (B - LeftFocus b)

/-- |AF₁| = 4|BF₁| -/
noncomputable def DistanceRatio (b : ℝ) (A B : ℝ × ℝ) : Prop :=
  Real.sqrt ((A.1 - (LeftFocus b).1)^2 + (A.2 - (LeftFocus b).2)^2) = 
  4 * Real.sqrt ((B.1 - (LeftFocus b).1)^2 + (B.2 - (LeftFocus b).2)^2)

/-- AF₂ is perpendicular to the x-axis -/
def Perpendicular (b : ℝ) (A : ℝ × ℝ) : Prop :=
  A.1 - (RightFocus b).1 = 0

theorem ellipse_equation (b : ℝ) (A B : ℝ × ℝ) 
  (h1 : 0 < b) (h2 : b < 1)
  (h3 : IntersectionPoints b A B)
  (h4 : DistanceRatio b A B)
  (h5 : Perpendicular b A) : 
  b^2 = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l286_28698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intercept_theorem_l286_28657

/-- A circle in the Cartesian plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in the plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The distance from a point to a line y = x -/
noncomputable def distanceToLineYEqX (p : ℝ × ℝ) : ℝ :=
  abs (p.2 - p.1) / Real.sqrt 2

theorem circle_intercept_theorem (P : Circle) :
  (P.radius^2 - P.center.2^2 = 2) ∧ 
  (P.radius^2 - P.center.1^2 = 3) →
  (∀ (x y : ℝ), y^2 - x^2 = 1 → ∃ (R : ℝ), Circle.mk (x, y) R = P) ∧
  (distanceToLineYEqX P.center = Real.sqrt 2 / 2 →
    (P.center = (0, 1) ∧ P.radius = Real.sqrt 3) ∨
    (P.center = (0, -1) ∧ P.radius = Real.sqrt 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intercept_theorem_l286_28657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_residuals_value_l286_28609

def regression_equation (x : ℝ) : ℝ := 2 * x + 1

def experimental_data : List (ℝ × ℝ) := [(2, 4.9), (3, 7.1), (4, 9.1)]

def residual_calc (point : ℝ × ℝ) : ℝ :=
  point.2 - regression_equation point.1

def sum_squared_residuals (data : List (ℝ × ℝ)) : ℝ :=
  (data.map residual_calc).map (λ x => x * x) |>.sum

theorem sum_squared_residuals_value :
  sum_squared_residuals experimental_data = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_residuals_value_l286_28609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_not_in_ages_l286_28612

/-- Represents the 4-digit number on the exhibit plaque -/
def exhibit_number : ℕ := sorry

/-- Represents the set of ages of Mr. Thompson's children -/
def children_ages : Finset ℕ := sorry

/-- Represents Mr. Thompson's age -/
def thompson_age : ℕ := sorry

/-- The exhibit number has two digits that appear twice -/
axiom exhibit_number_form : ∃ a b : ℕ, exhibit_number = a * 1000 + a * 100 + b * 10 + b

/-- The exhibit number is divisible by each child's age -/
axiom divisible_by_ages : ∀ age : ℕ, age ∈ children_ages → exhibit_number % age = 0

/-- The last two digits of the exhibit number represent Mr. Thompson's age -/
axiom thompson_age_last_digits : thompson_age = exhibit_number % 100

/-- There are 9 children with different ages -/
axiom nine_different_ages : children_ages.card = 9 ∧ (∀ a b : ℕ, a ∈ children_ages → b ∈ children_ages → a = b → a = b)

/-- The oldest child is 10 years old -/
axiom oldest_child_10 : 10 ∈ children_ages

/-- Theorem: 7 is not in the set of ages of Mr. Thompson's children -/
theorem seven_not_in_ages : 7 ∉ children_ages := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_not_in_ages_l286_28612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l286_28681

/-- The maximum distance from any point on the unit circle to the line x + y = 2 is √2 + 1 -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let line := {p : ℝ × ℝ | p.1 + p.2 = 2}
  ∃ (d : ℝ), d = Real.sqrt 2 + 1 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ circle → 
      ∀ (q : ℝ × ℝ), q ∈ line → 
        Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ d ∧
        ∃ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ circle ∧ q' ∈ line ∧ 
          Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l286_28681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_equilateral_triangle_area_l286_28601

/-- Structure for a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Structure for a triangle -/
structure Triangle where
  vertices : List (ℝ × ℝ)

/-- The area of the largest equilateral triangle inscribed in a circle -/
theorem largest_inscribed_equilateral_triangle_area
  (C : Circle) -- Circle C
  (h : C.radius = 8) -- The radius is 8 cm
  (T : Triangle) -- Equilateral triangle T
  (inscribed : T.vertices.length = 3 ∧ ∀ v ∈ T.vertices, (v.1 - C.center.1)^2 + (v.2 - C.center.2)^2 ≤ C.radius^2) -- T is inscribed in C
  (vertex_at_center : C.center ∈ T.vertices) -- One vertex of T is at the center of C
  : 64 * Real.sqrt 3 = -- The area of T
    let side := 2 * C.radius -- The side length of T
    let height := C.radius * Real.sqrt 3 -- The height of T
    (1/2) * side * height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_equilateral_triangle_area_l286_28601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l286_28614

/-- The circle equation -/
def circle_eq (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*a*y + a^2 = 0

/-- The line equation -/
def line_eq (x y : ℝ) : Prop :=
  x - 2*y + 1 = 0

/-- The chord length is 2 -/
def chord_length_is_2 (a : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_eq x1 y1 a ∧ circle_eq x2 y2 a ∧
    line_eq x1 y1 ∧ line_eq x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = 4

theorem circle_line_intersection (a : ℝ) :
  chord_length_is_2 a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l286_28614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l286_28692

/-- Given a function g(x) = (x-3) / (x^2 + cx + d) with vertical asymptotes at x = -1 and x = 3,
    prove that the sum of c and d is -5. -/
theorem asymptote_sum (c d : ℝ) (g : ℝ → ℝ) : 
  (∀ x : ℝ, g x = (x - 3) / (x^2 + c*x + d)) →
  (∀ x : ℝ, x ≠ -1 ∧ x ≠ 3 → g x ≠ 0) →
  (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x + 1| ∧ |x + 1| < δ → |g x| > 1/ε) →
  (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x - 3| ∧ |x - 3| < δ → |g x| > 1/ε) →
  c + d = -5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l286_28692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_sum_product_l286_28630

theorem not_prime_sum_product (a b c d : ℕ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0)
  (h5 : a * c + b * d = (b + d + a - c) * (b + d - a + c)) :
  ¬ Nat.Prime (a * b + c * d) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_sum_product_l286_28630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_angle_is_30_degrees_l286_28663

/-- Ship represents a vessel with a position and velocity --/
structure Ship where
  position : ℝ × ℝ
  velocity : ℝ × ℝ

/-- The angle between two 2D vectors --/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- The optimal interception angle for Ship A to catch Ship B --/
noncomputable def optimal_interception_angle (a b : Ship) (initial_angle : ℝ) (speed_ratio : ℝ) : ℝ :=
  sorry

/-- Theorem stating the optimal interception angle for the given scenario --/
theorem optimal_angle_is_30_degrees 
  (a b : Ship) 
  (h1 : angle (b.position.1 - a.position.1, b.position.2 - a.position.2) (0, 1) = π / 3) 
  (h2 : ‖b.velocity‖ / ‖a.velocity‖ = 1 / Real.sqrt 3) 
  (h3 : b.velocity.1 = 0 ∧ b.velocity.2 > 0) :
  optimal_interception_angle a b (π / 3) (Real.sqrt 3) = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_angle_is_30_degrees_l286_28663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_sums_equals_forty_l286_28647

/-- Rounds a number to the nearest multiple of 5, rounding 2.5s up -/
def roundToNearestFive (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

/-- Calculates the sum of integers from 1 to n -/
def sumToN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Calculates the sum of rounded integers from 1 to n -/
def sumRoundedToN (n : ℕ) : ℕ :=
  (List.range n).map (fun i => roundToNearestFive (i + 1)) |>.sum

theorem difference_of_sums_equals_forty :
  sumToN 100 - sumRoundedToN 100 = 40 := by
  sorry

#eval sumToN 100 - sumRoundedToN 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_sums_equals_forty_l286_28647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_needs_twelve_days_l286_28623

/-- The number of days x needs to finish the remaining work --/
noncomputable def remaining_days (x_days y_days y_worked_days : ℝ) : ℝ :=
  (1 - y_worked_days / y_days) * x_days

/-- Theorem stating that x needs 12 days to finish the remaining work --/
theorem x_needs_twelve_days (x_days y_days y_worked_days : ℝ)
  (hx : x_days = 18)
  (hy : y_days = 15)
  (hw : y_worked_days = 5) :
  remaining_days x_days y_days y_worked_days = 12 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_needs_twelve_days_l286_28623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_archer_weekly_cost_is_810_l286_28636

/-- Represents the cost and recovery information for a type of arrow -/
structure ArrowType where
  cost : ℚ
  recovery_rate_1 : ℚ
  recovery_rate_2 : ℚ
  team_contribution_low : ℚ
  team_contribution_high : ℚ
  success_rate_threshold : ℚ

/-- Represents a shooting session -/
structure ShootingSession where
  arrow_type : ArrowType
  num_arrows : ℕ
  use_recovery_rate_1 : Bool
  success_rate : ℚ

def calculate_session_cost (session : ShootingSession) : ℚ :=
  let recovery_rate := if session.use_recovery_rate_1 then session.arrow_type.recovery_rate_1 else session.arrow_type.recovery_rate_2
  let arrows_to_pay := session.num_arrows - (↑session.num_arrows * recovery_rate).floor
  let team_contribution := if session.success_rate ≥ session.arrow_type.success_rate_threshold then session.arrow_type.team_contribution_high else session.arrow_type.team_contribution_low
  (1 - team_contribution) * (↑arrows_to_pay * session.arrow_type.cost)

def archer_weekly_cost (type_a : ArrowType) (type_b : ArrowType) : ℚ :=
  let monday := calculate_session_cost { arrow_type := type_a, num_arrows := 200, use_recovery_rate_1 := true, success_rate := 92/100 }
  let wednesday := calculate_session_cost { arrow_type := type_a, num_arrows := 200, use_recovery_rate_1 := false, success_rate := 92/100 }
  let tuesday := calculate_session_cost { arrow_type := type_b, num_arrows := 150, use_recovery_rate_1 := true, success_rate := 88/100 }
  let thursday := calculate_session_cost { arrow_type := type_b, num_arrows := 150, use_recovery_rate_1 := false, success_rate := 88/100 }
  monday + wednesday + tuesday + thursday

theorem archer_weekly_cost_is_810 :
  let type_a : ArrowType := {
    cost := 11/2,
    recovery_rate_1 := 15/100,
    recovery_rate_2 := 25/100,
    team_contribution_low := 60/100,
    team_contribution_high := 80/100,
    success_rate_threshold := 90/100
  }
  let type_b : ArrowType := {
    cost := 7,
    recovery_rate_1 := 20/100,
    recovery_rate_2 := 35/100,
    team_contribution_low := 40/100,
    team_contribution_high := 70/100,
    success_rate_threshold := 85/100
  }
  archer_weekly_cost type_a type_b = 810 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_archer_weekly_cost_is_810_l286_28636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_range_l286_28674

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x / (1 - x)) / Real.log 10

-- State the theorem
theorem ab_range (a b : ℝ) (ha : 0 < a) (hb : a < b) (hb1 : b < 1) 
  (hf : f a + f b = 0) : 0 < a * b ∧ a * b < 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_range_l286_28674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_tangent_lines_chord_length_l286_28656

-- Define the circle
def circle_center : ℝ × ℝ := (1, 2)
def circle_radius : ℝ := 3

-- Define points A, B, and M
def point_A : ℝ × ℝ := (1, -1)
def point_B : ℝ × ℝ := (4, 2)
def point_M : ℝ × ℝ := (-2, 1)

-- Define the line equation x - y + 1 = 0
def line_equation (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem for the standard equation of the circle
theorem circle_equation (x y : ℝ) : 
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 ↔
  (x - 1)^2 + (y - 2)^2 = 9 := by
  sorry

-- Theorem for the tangent lines
theorem tangent_lines (x y : ℝ) :
  ((x = -2) ∨ (4*x + 3*y + 5 = 0)) ↔
  ((x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 ∧
   (x = point_M.1 ∨ y = point_M.2)) := by
  sorry

-- Theorem for the chord length
theorem chord_length :
  let d := |circle_center.2 - 0|
  2 * Real.sqrt (circle_radius^2 - d^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_tangent_lines_chord_length_l286_28656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_count_l286_28654

theorem right_triangles_count :
  (∃! n : ℕ, n = (Finset.filter (λ p : ℕ × ℕ =>
    p.1 * p.1 + p.2 * p.2 = (p.2 + 3) * (p.2 + 3) ∧
    p.2 < 100)
    (Finset.product (Finset.range 100) (Finset.range 100))).card) ∧
  (∃ n : ℕ, n = (Finset.filter (λ p : ℕ × ℕ =>
    p.1 * p.1 + p.2 * p.2 = (p.2 + 3) * (p.2 + 3) ∧
    p.2 < 100)
    (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ n = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_count_l286_28654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l286_28650

/-- Given a train of length 140 m, traveling at 60 kmph, taking 23.998080153587715 seconds to pass a platform,
    the length of the platform is 259.9980801535877 meters. -/
theorem platform_length
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (time_to_pass : ℝ)
  (h1 : train_length = 140)
  (h2 : train_speed_kmph = 60)
  (h3 : time_to_pass = 23.998080153587715) :
  (train_speed_kmph * 1000 / 3600 * time_to_pass - train_length) = 259.9980801535877 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l286_28650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_18_10_l286_28613

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) : 
  Nat.choose 18 10 = 43758 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_18_10_l286_28613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_calculation_l286_28685

-- Define the constants
noncomputable def slant_height : ℝ := 15
noncomputable def cone_height : ℝ := 9

-- Define the volume function for a cone
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem cone_volume_calculation :
  ∃ (r : ℝ), r^2 + cone_height^2 = slant_height^2 ∧ 
  cone_volume r cone_height = 432 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_calculation_l286_28685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_l286_28642

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if sin C / sin A = 2 and b² - a² = (3/2)ac, then cos B = 1/4 -/
theorem triangle_cosine (a b c : ℝ) (A B C : Real) : 
  a > 0 → b > 0 → c > 0 →
  Real.sin A ≠ 0 →
  Real.sin C / Real.sin A = 2 →
  b^2 - a^2 = (3/2) * a * c →
  Real.cos B = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_l286_28642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_degree_l286_28662

/-- A polynomial P that satisfies P(sin x) + P(cos x) = 1 for all real x -/
def SpecialPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, P (Real.sin x) + P (Real.cos x) = 1

/-- The set of possible degrees for a SpecialPolynomial -/
def SpecialPolynomialDegrees : Set ℕ :=
  {0} ∪ {n : ℕ | ∃ k : ℕ, n = 2 + 4 * k}

/-- Theorem stating that the degree of a SpecialPolynomial is in SpecialPolynomialDegrees -/
theorem special_polynomial_degree (P : ℝ → ℝ) (hP : SpecialPolynomial P) 
    (hPoly : Polynomial ℝ) (hP_eq : (hPoly.eval : ℝ → ℝ) = P) :
  hPoly.natDegree ∈ SpecialPolynomialDegrees := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_degree_l286_28662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_delta_calculation_l286_28668

-- Define the nabla operation
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Define the delta operation
noncomputable def delta (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

-- Theorem statement
theorem nabla_delta_calculation :
  (nabla 3 4 = 7/13) ∧ (delta 3 4 = 1/11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_delta_calculation_l286_28668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_of_sets_condition_for_subset_l286_28660

-- Define set A
def A : Set ℝ := {x | 1 ≤ (2:ℝ)^x ∧ (2:ℝ)^x ≤ 4}

-- Define set B with parameter a
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Theorem for part 1
theorem intersection_and_union_of_sets :
  (A ∩ B 1 = Set.Ioc 1 2) ∧ ((Set.univ \ B 1) ∪ A = Set.Iic 2) :=
sorry

-- Theorem for part 2
theorem condition_for_subset :
  ∀ a : ℝ, A ∪ B a = B a → a < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_of_sets_condition_for_subset_l286_28660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_condition_l286_28664

theorem right_triangle_condition (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C ∧
  (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B) ∧
  A ≠ B →
  C = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_condition_l286_28664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_real_floor_product_72_l286_28632

theorem positive_real_floor_product_72 (x : ℝ) (h1 : x > 0) (h2 : x * ⌊x⌋ = 72) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_real_floor_product_72_l286_28632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l286_28699

theorem relationship_abc : 
  let a := (2 : ℝ)^(-2 : ℤ)
  let b := (3 : ℝ)^(1 : ℝ)
  let c := (-1 : ℝ)^(3 : ℕ)
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l286_28699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_toppings_l286_28610

def has_pepperoni : ℕ → Prop := sorry
def has_mushroom : ℕ → Prop := sorry
def has_olive : ℕ → Prop := sorry

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h_total : total_slices = 24)
  (h_pep : pepperoni_slices = 15)
  (h_mush : mushroom_slices = 14)
  (h_at_least_one : ∀ s, s < total_slices → (has_pepperoni s ∨ has_mushroom s ∨ has_olive s)) :
  ∃ both : ℕ, both = 5 ∧ 
    (∀ s, s < total_slices → (has_pepperoni s ∧ has_mushroom s) ↔ s < both) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_toppings_l286_28610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_l286_28644

/-- An arithmetic sequence with given first three terms -/
def arithmetic_sequence (a : ℝ) : ℕ → ℝ :=
  sorry  -- Implementation details are omitted for now

/-- The first term of the sequence -/
axiom first_term (a : ℝ) : arithmetic_sequence a 1 = a - 1

/-- The second term of the sequence -/
axiom second_term (a : ℝ) : arithmetic_sequence a 2 = a + 1

/-- The third term of the sequence -/
axiom third_term (a : ℝ) : arithmetic_sequence a 3 = 2*a + 3

/-- The sequence is arithmetic -/
axiom is_arithmetic (a : ℝ) (n : ℕ) : 
  arithmetic_sequence a (n+1) - arithmetic_sequence a n = 
  arithmetic_sequence a (n+2) - arithmetic_sequence a (n+1)

theorem general_term (n : ℕ) : 
  ∃ a : ℝ, arithmetic_sequence a n = 2*n - 3 := by
  sorry  -- Proof is omitted for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_l286_28644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_implies_a_values_l286_28607

/-- Parametric equation of line l -/
def line_l (t a : ℝ) : ℝ × ℝ := (-4 * t + a, 3 * t - 1)

/-- Polar equation of circle M -/
def circle_M (ρ θ : ℝ) : Prop := ρ^2 - 6 * ρ * Real.sin θ = -8

/-- Rectangular equation of circle M -/
def circle_M_rect (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 1

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

/-- Theorem: If the chord length of intersection between line l and circle M is √3,
    then the possible values of a are 9/2 or 37/6 -/
theorem intersection_chord_length_implies_a_values (a : ℝ) :
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
    circle_M_rect (line_l t₁ a).1 (line_l t₁ a).2 ∧
    circle_M_rect (line_l t₂ a).1 (line_l t₂ a).2 ∧
    ((line_l t₁ a).1 - (line_l t₂ a).1)^2 + ((line_l t₁ a).2 - (line_l t₂ a).2)^2 = 3) →
  (distance_point_to_line 0 3 3 4 (-3 * a + 4) = Real.sqrt (1 - (Real.sqrt 3 / 2)^2)) →
  (a = 9/2 ∨ a = 37/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_implies_a_values_l286_28607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_perpendicular_line_l286_28658

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a point in Cartesian coordinates -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Converts a polar point to a Cartesian point -/
noncomputable def polarToCartesian (p : PolarPoint) : CartesianPoint :=
  { x := p.ρ * Real.cos p.θ
    y := p.ρ * Real.sin p.θ }

/-- Defines the curve ρ cos θ = 0 in polar coordinates -/
def polarCurve (p : PolarPoint) : Prop :=
  p.ρ * Real.cos p.θ = 0

/-- Defines a line perpendicular to the polar axis (x = 0 in Cartesian coordinates) -/
def perpendicularLine (c : CartesianPoint) : Prop :=
  c.x = 0

/-- Theorem: The curve ρ cos θ = 0 in polar coordinates is equivalent to 
    a line perpendicular to the polar axis -/
theorem polar_curve_is_perpendicular_line :
  ∀ p : PolarPoint, polarCurve p ↔ perpendicularLine (polarToCartesian p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_perpendicular_line_l286_28658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_f_monotonicity_l286_28671

-- Define the function f
noncomputable def f (x θ : ℝ) : ℝ := x^2 + 2 * x * Real.tan θ - 1

-- Part 1
theorem f_extrema :
  ∀ x ∈ Set.Icc (-1 : ℝ) (Real.sqrt 3),
  f x (-Real.pi/4) ≥ -2 ∧
  f x (-Real.pi/4) ≤ 2 ∧
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) (Real.sqrt 3), f x₁ (-Real.pi/4) = -2) ∧
  (∃ x₂ ∈ Set.Icc (-1 : ℝ) (Real.sqrt 3), f x₂ (-Real.pi/4) = 2) :=
sorry

-- Part 2
theorem f_monotonicity :
  ∀ θ, θ ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) →
  ((∀ x y, x ∈ Set.Icc (-Real.sqrt 3) 1 → y ∈ Set.Icc (-Real.sqrt 3) 1 → x < y → f x θ < f y θ) ∨
  (∀ x y, x ∈ Set.Icc (-Real.sqrt 3) 1 → y ∈ Set.Icc (-Real.sqrt 3) 1 → x < y → f x θ > f y θ))
  ↔ 
  θ ∈ Set.Ioo (-Real.pi/2) (-Real.pi/4) ∪ Set.Ioo (Real.pi/3) (Real.pi/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_f_monotonicity_l286_28671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sasha_kolya_l286_28675

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race scenario -/
structure Race where
  sasha : Runner
  lyosha : Runner
  kolya : Runner
  raceLength : ℝ
  lyoshaBehindSasha : ℝ
  kolyaBehindLyosha : ℝ

/-- Theorem stating the distance between Sasha and Kolya when Sasha finishes -/
theorem distance_sasha_kolya (race : Race) 
  (h1 : race.raceLength = 100)
  (h2 : race.lyoshaBehindSasha = 10)
  (h3 : race.kolyaBehindLyosha = 10)
  (h4 : race.sasha.speed > 0)
  (h5 : race.lyosha.speed > 0)
  (h6 : race.kolya.speed > 0) :
  race.raceLength - (race.raceLength - race.lyoshaBehindSasha) * 
  (race.kolya.speed / race.lyosha.speed) = 19 := by
  sorry

#eval "Theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sasha_kolya_l286_28675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_implies_a_leq_3_f_has_min_max_on_interval_ln_plus_one_gt_exp_inequality_l286_28600

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x
def g (x : ℝ) : ℝ := -x^2 - 2

-- Theorem 1
theorem f_geq_g_implies_a_leq_3 (a : ℝ) : 
  (∀ x > 0, f a x ≥ g x) → a ≤ 3 := by sorry

-- Theorem 2
theorem f_has_min_max_on_interval (m : ℝ) (h : m > 0) :
  ∃ (min max : ℝ), ∀ x ∈ Set.Icc m (m + 3), 
    min ≤ f (-1) x ∧ f (-1) x ≤ max := by sorry

-- Theorem 3
theorem ln_plus_one_gt_exp_inequality :
  ∀ x > 0, Real.log x + 1 > 1 / Real.exp x - 2 / (Real.exp 1 * x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_implies_a_leq_3_f_has_min_max_on_interval_ln_plus_one_gt_exp_inequality_l286_28600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cost_is_two_l286_28618

/-- Represents an ice cream shop with a promotion where every sixth customer gets a free cone. -/
structure IceCreamShop where
  totalSales : ℚ
  freeConesGiven : ℕ

/-- Calculates the cost of each cone given the shop's sales data. -/
def coneCost (shop : IceCreamShop) : ℚ :=
  shop.totalSales / (5 * shop.freeConesGiven)

/-- Theorem stating that for the given conditions, each cone costs $2. -/
theorem cone_cost_is_two (shop : IceCreamShop)
  (h1 : shop.totalSales = 100)
  (h2 : shop.freeConesGiven = 10) :
  coneCost shop = 2 := by
  unfold coneCost
  rw [h1, h2]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cost_is_two_l286_28618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_catches_john_l286_28622

/-- The time it takes for Bob to catch up to John -/
noncomputable def catch_up_time (john_speed bob_speed initial_distance : ℝ) : ℝ :=
  initial_distance / (bob_speed - john_speed)

/-- Conversion from hours to minutes -/
def hours_to_minutes (hours : ℝ) : ℝ :=
  hours * 60

theorem bob_catches_john (john_speed bob_speed initial_distance : ℝ) 
  (h1 : john_speed = 2)
  (h2 : bob_speed = 6)
  (h3 : initial_distance = 2) :
  hours_to_minutes (catch_up_time john_speed bob_speed initial_distance) = 30 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval hours_to_minutes (catch_up_time 2 6 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_catches_john_l286_28622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l286_28649

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.log (x + 1) / Real.log (1/2)
  else Real.log (-x + 1) / Real.log (1/2)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧  -- f is even
  f 0 = 0 ∧
  f (-1) = -1 ∧
  (∀ a, f (a - 1) < f (3 - a) ↔ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l286_28649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l286_28645

theorem inequality_proof (a b x y : ℝ) :
  |a * x^2 + 2 * b * x * y - a * y^2| ≤ Real.sqrt (a^2 + b^2) * (x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l286_28645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_unattainable_postage_l286_28678

/-- The Frobenius number for two coprime positive integers -/
def frobenius_number (a b : ℕ) : ℕ := a * b - a - b

/-- Represents the ability to form a postage amount using given stamp denominations -/
def can_form_postage (a b n : ℕ) : Prop :=
  ∃ (x y : ℕ), n = a * x + b * y

/-- The main theorem about the largest unattainable postage amount -/
theorem largest_unattainable_postage :
  let a : ℕ := 8
  let b : ℕ := 15
  let n : ℕ := frobenius_number a b
  (∀ m : ℕ, m > n → can_form_postage a b m) ∧
  ¬(can_form_postage a b n) ∧
  n = 97 :=
by
  sorry

#eval frobenius_number 8 15  -- This will evaluate to 97

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_unattainable_postage_l286_28678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l286_28676

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := 1 - 1/(x^2)

-- Statement to prove
theorem tangent_slope_range :
  ∀ x : ℝ, x ≠ 0 → 
  (∃ k : ℝ, k = f' x ∧ k < 1) ∧ 
  (∀ L : ℝ, ∃ k : ℝ, k = f' x ∧ k < L) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l286_28676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Br_in_AlBr3_l286_28606

-- Define the molar masses
noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_Br : ℝ := 79.90

-- Define the chemical formula of aluminum bromide (AlBr3)
def num_Br_atoms : ℕ := 3

-- Define the mass percentage calculation function
noncomputable def mass_percentage (mass_element : ℝ) (total_mass : ℝ) : ℝ :=
  (mass_element / total_mass) * 100

-- Theorem statement
theorem mass_percentage_Br_in_AlBr3 :
  let total_mass_Br := molar_mass_Br * (num_Br_atoms : ℝ)
  let total_mass_AlBr3 := molar_mass_Al + total_mass_Br
  let percentage := mass_percentage total_mass_Br total_mass_AlBr3
  (percentage ≥ 89.88) ∧ (percentage ≤ 89.90) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Br_in_AlBr3_l286_28606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_of_30_60_90_triangles_l286_28641

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ

/-- Creates a 30-60-90 triangle with the given hypotenuse length -/
noncomputable def makeTriangle (h : ℝ) : Triangle30_60_90 :=
  { hypotenuse := h
  , shorterLeg := h / 2
  , longerLeg := h * (Real.sqrt 3) / 2 }

/-- Calculates the area of overlap between two congruent 30-60-90 triangles -/
noncomputable def overlapArea (t : Triangle30_60_90) (overlap : ℝ) : ℝ :=
  (overlap * t.longerLeg) / 4

theorem overlap_area_of_30_60_90_triangles :
  let t := makeTriangle 10
  let overlap := 5
  overlapArea t overlap = 25 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_of_30_60_90_triangles_l286_28641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_inequality_l286_28626

/-- A real-coefficient cubic polynomial with all non-negative roots -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  roots_nonneg : ∀ r : ℝ, r^3 + a*r^2 + b*r + c = 0 → r ≥ 0

/-- The polynomial function -/
def f (p : CubicPolynomial) (x : ℝ) : ℝ := x^3 + p.a*x^2 + p.b*x + p.c

/-- The theorem statement -/
theorem cubic_polynomial_inequality (p : CubicPolynomial) :
  ∃ lambda : ℝ, lambda = -1/27 ∧ 
  (∀ x : ℝ, x ≥ 0 → f p x ≥ lambda*(x - p.a)^3) ∧
  (∀ mu : ℝ, (∀ x : ℝ, x ≥ 0 → f p x ≥ mu*(x - p.a)^3) → mu ≤ lambda) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_inequality_l286_28626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_area_calculation_l286_28665

/-- Represents the total area of an island in hectares. -/
def island_area (A : ℝ) : Prop := A > 0

/-- Given an island where:
    - The total area is covered by forest, sand dunes, and farm land.
    - 2/5 of the area is forest.
    - 1/4 of the non-forest area is sand dunes.
    - 90 hectares are farm land.
    Then the total area of the island is 200 hectares. -/
theorem island_area_calculation (A : ℝ) :
  island_area A →
  2 / 5 * A + 1 / 4 * (3 / 5 * A) + 90 = A →
  A = 200 := by
  intros h1 h2
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_area_calculation_l286_28665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_dot_product_l286_28652

noncomputable section

-- Define the isosceles triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  (AB = AC) ∧ (BC = 2 * Real.sqrt 3)

-- Define the vertex angle
def VertexAngle (A B C : ℝ × ℝ) : ℝ :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  Real.arccos ((AB.1 * AC.1 + AB.2 * AC.2) / 
    (Real.sqrt (AB.1^2 + AB.2^2) * Real.sqrt (AC.1^2 + AC.2^2)))

-- Define the dot product
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem isosceles_triangle_dot_product 
  (A B C : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : VertexAngle A B C = 2 * Real.pi / 3) : 
  DotProduct (B.1 - A.1, B.2 - A.2) (C.1 - A.1, C.2 - A.2) = -2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_dot_product_l286_28652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_valid_triangle_perimeter_l286_28604

/-- Represents a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Generates the next triangle in the sequence. -/
noncomputable def nextTriangle (T : Triangle) : Triangle where
  a := (T.b + T.c - T.a) / 2
  b := (T.a + T.c - T.b) / 2
  c := (T.a + T.b - T.c) / 2

/-- Checks if the triangle inequality holds for a given triangle. -/
def triangleInequalityHolds (T : Triangle) : Prop :=
  T.a + T.b > T.c ∧ T.b + T.c > T.a ∧ T.c + T.a > T.b

/-- Calculates the perimeter of a triangle. -/
def perimeter (T : Triangle) : ℝ :=
  T.a + T.b + T.c

/-- The initial triangle in the sequence. -/
def T₁ : Triangle where
  a := 101
  b := 102
  c := 100

/-- The theorem stating the perimeter of the last valid triangle. -/
theorem last_valid_triangle_perimeter :
  ∃ n : ℕ, 
    let Tₙ := (Nat.iterate nextTriangle n) T₁
    triangleInequalityHolds Tₙ ∧
    ¬triangleInequalityHolds (nextTriangle Tₙ) ∧
    perimeter Tₙ = 151.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_valid_triangle_perimeter_l286_28604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l286_28697

-- Define the functions f and g
def f (x : ℝ) : ℝ := x

noncomputable def g (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l286_28697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_tetrahedron_edge_length_l286_28625

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a square pyramid -/
structure SquarePyramid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E : Point3D

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Defines a unit square pyramid with given coordinates -/
def unitSquarePyramid : SquarePyramid :=
  { A := ⟨0, 0, 0⟩
  , B := ⟨1, 0, 0⟩
  , C := ⟨1, 1, 0⟩
  , D := ⟨0, 1, 0⟩
  , E := ⟨0.5, 0.5, 1⟩ }

/-- Represents a tetrahedron inscribed in the pyramid -/
structure InscribedTetrahedron (p : SquarePyramid) where
  vertexOnAB : Point3D
  vertexOnBC : Point3D
  vertexOnCD : Point3D
  isOnEdgeAB : vertexOnAB.z = 0 ∧ 0 ≤ vertexOnAB.x ∧ vertexOnAB.x ≤ 1 ∧ vertexOnAB.y = 0
  isOnEdgeBC : vertexOnBC.x = 1 ∧ 0 ≤ vertexOnBC.y ∧ vertexOnBC.y ≤ 1 ∧ vertexOnBC.z = 0
  isOnEdgeCD : vertexOnCD.z = 0 ∧ 0 ≤ vertexOnCD.x ∧ vertexOnCD.x ≤ 1 ∧ vertexOnCD.y = 1
  isRegular : distance vertexOnAB vertexOnBC = distance vertexOnBC vertexOnCD ∧
              distance vertexOnBC vertexOnCD = distance vertexOnCD p.E ∧
              distance vertexOnCD p.E = distance p.E vertexOnAB ∧
              distance p.E vertexOnAB = distance vertexOnAB vertexOnBC

theorem inscribed_tetrahedron_edge_length 
  (t : InscribedTetrahedron unitSquarePyramid) : 
  distance t.vertexOnAB t.vertexOnBC = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_tetrahedron_edge_length_l286_28625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_volume_error_cylindrical_container_l286_28605

/-- Represents a cylindrical container with a given diameter and height. -/
structure Cylinder where
  diameter : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder. -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * (c.diameter / 2) ^ 2 * c.height

/-- Calculates the percent error between two values. -/
noncomputable def percentError (actual value : ℝ) : ℝ := |actual - value| / actual * 100

theorem largest_volume_error_cylindrical_container :
  let actualCylinder : Cylinder := { diameter := 30, height := 10 }
  let maxDiameter := actualCylinder.diameter * 1.1
  let maxVolume := cylinderVolume { diameter := maxDiameter, height := actualCylinder.height }
  percentError (cylinderVolume actualCylinder) maxVolume = 21 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_volume_error_cylindrical_container_l286_28605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_30_degrees_l286_28639

theorem cot_30_degrees : Real.tan (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_30_degrees_l286_28639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_formula_l286_28627

/-- Given a cubic equation x^3 + px + q = 0 with roots in the form x = ∛α + ∛β,
    α and β are given by the formula: α, β = -q/2 ± √(q^2/4 + p^3/27) -/
theorem cubic_root_formula (p q : ℝ) :
  let α₁ := -q/2 + Real.sqrt (q^2/4 + p^3/27)
  let α₂ := -q/2 - Real.sqrt (q^2/4 + p^3/27)
  let x := (α₁ ^ (1/3 : ℝ)) + (α₂ ^ (1/3 : ℝ))
  x^3 + p*x + q = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_formula_l286_28627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardboard_overlap_l286_28682

/-- Represents a shape in 2D space -/
structure Shape where
  contains : ℝ × ℝ → Prop

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  width : ℝ
  height : ℝ
  contains : ℝ × ℝ → Prop

/-- Two identical shapes can cover a rectangle -/
def can_cover (s : Shape) (r : Rectangle) : Prop :=
  ∀ p, r.contains p → ∃ q, s.contains p ∨ s.contains q

/-- The center point of a rectangle -/
noncomputable def center (r : Rectangle) : ℝ × ℝ :=
  (r.width / 2, r.height / 2)

/-- Main theorem -/
theorem cardboard_overlap (s : Shape) (r : Rectangle) 
  (h_cover : can_cover s r) :
  ∃ (p q : ℝ × ℝ), 
    r.contains (center r) ∧ 
    s.contains p ∧ 
    s.contains q ∧ 
    (s.contains (center r) ↔ p = (center r)) ∧
    (¬s.contains (center r) ↔ q = (center r)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardboard_overlap_l286_28682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bananas_purchased_correct_l286_28693

/-- The number of pounds of bananas purchased by the grocer -/
noncomputable def bananas_purchased : ℝ := 84

/-- The purchase price in dollars per pound -/
noncomputable def purchase_price : ℝ := 0.50 / 3

/-- The selling price in dollars per pound -/
noncomputable def selling_price : ℝ := 1.00 / 4

/-- The total profit in dollars -/
noncomputable def total_profit : ℝ := 7.00

/-- Theorem stating that the number of pounds of bananas purchased is correct -/
theorem bananas_purchased_correct : 
  bananas_purchased * (selling_price - purchase_price) = total_profit := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bananas_purchased_correct_l286_28693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_sum_l286_28659

/-- The ellipse C -/
def C (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

/-- The point Q -/
def Q (lambda mu : ℝ) : Prop := lambda^2 + 4*mu^2 = 1

/-- The fixed point E₁ -/
noncomputable def E₁ : ℝ × ℝ := (-Real.sqrt 3/2, 0)

/-- The fixed point E₂ -/
noncomputable def E₂ : ℝ × ℝ := (Real.sqrt 3/2, 0)

/-- The distance between two points -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem ellipse_distance_sum (lambda mu : ℝ) :
  Q lambda mu → distance (lambda, mu) E₁ + distance (lambda, mu) E₂ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_sum_l286_28659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_not_right_triangle_l286_28677

-- Define the sets of line segments
noncomputable def set_A : ℝ × ℝ × ℝ := (5, 12, 13)
noncomputable def set_B : ℝ × ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2, 2)
noncomputable def set_C : ℝ × ℝ × ℝ := (1, 2, Real.sqrt 5)
noncomputable def set_D : ℝ × ℝ × ℝ := (4, 5, 6)

-- Define a function to check if a set of line segments forms a right triangle
def is_right_triangle (s : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c) := s
  a^2 + b^2 = c^2

-- Theorem statement
theorem only_D_not_right_triangle :
  is_right_triangle set_A ∧
  is_right_triangle set_B ∧
  is_right_triangle set_C ∧
  ¬is_right_triangle set_D :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_not_right_triangle_l286_28677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zephyr_orbit_theorem_l286_28638

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perihelion : ℝ  -- Distance at perihelion
  aphelion : ℝ    -- Distance at aphelion

/-- The distance from the focus to a point on the minor axis of the ellipse -/
noncomputable def distance_to_minor_axis (orbit : EllipticalOrbit) : ℝ :=
  (orbit.perihelion + orbit.aphelion) / 2

/-- Theorem: In an elliptical orbit where the perihelion distance is 3 AU
    and the aphelion distance is 15 AU, the distance from the focus (star)
    to a point on the minor axis of the ellipse is 9 AU. -/
theorem zephyr_orbit_theorem (orbit : EllipticalOrbit)
    (h_perihelion : orbit.perihelion = 3)
    (h_aphelion : orbit.aphelion = 15) :
    distance_to_minor_axis orbit = 9 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zephyr_orbit_theorem_l286_28638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l286_28633

theorem remainder_problem (k : ℕ) (hk : k > 0) (h : 80 % k = 8) : 150 % (k^2) = 69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l286_28633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_increase_l286_28628

/-- Calculate the percent increase given initial and final prices -/
noncomputable def percentIncrease (initialPrice finalPrice : ℝ) : ℝ :=
  ((finalPrice - initialPrice) / initialPrice) * 100

/-- Theorem: The percent increase from $6 to $8 is approximately 33.33% -/
theorem stock_price_increase : 
  let initialPrice : ℝ := 6
  let finalPrice : ℝ := 8
  abs (percentIncrease initialPrice finalPrice - 33.33) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval percentIncrease 6 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_increase_l286_28628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_l286_28655

/-- The number of lines tangent to two circles -/
def tangent_lines_count (circle1 circle2 : ℝ → ℝ → Prop) : ℕ :=
  sorry

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 4*y + 7 = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 10*y + 13 = 0

/-- Theorem stating that there are exactly 3 lines tangent to both circles -/
theorem tangent_lines_theorem :
  tangent_lines_count circle1 circle2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_l286_28655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_f_max_value_l286_28616

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt (1 + Real.sin x) + Real.sqrt (1 - Real.sin x) + 
  Real.sqrt (2 + Real.sin x) + Real.sqrt (2 - Real.sin x) + 
  Real.sqrt (3 + Real.sin x) + Real.sqrt (3 - Real.sin x)

/-- The theorem stating that f(x) is bounded above by 2 + 2√2 + 2√3 for all real x -/
theorem f_upper_bound : ∀ x : ℝ, f x ≤ 2 + 2 * Real.sqrt 2 + 2 * Real.sqrt 3 := by
  sorry

/-- The theorem stating that the maximum value of f(x) is 2 + 2√2 + 2√3 -/
theorem f_max_value : ∃ x : ℝ, f x = 2 + 2 * Real.sqrt 2 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_f_max_value_l286_28616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_in_fourth_quadrant_l286_28669

theorem complex_in_fourth_quadrant (a b : ℝ) : 
  (Complex.mk (a^2 + 1) (-(b^2 + 1))).re > 0 ∧ 
  (Complex.mk (a^2 + 1) (-(b^2 + 1))).im < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_in_fourth_quadrant_l286_28669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l286_28603

-- Define the constants
noncomputable def a : ℝ := Real.rpow 0.7 (1/2)
noncomputable def b : ℝ := Real.rpow 0.2 (-2)
noncomputable def c : ℝ := Real.log 0.7 / Real.log 3

-- State the theorem
theorem relationship_abc : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l286_28603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_T_given_S₂₀₁₉_l286_28637

/-- An arithmetic sequence -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) : ℕ → ℝ := fun n ↦ a₁ + (n - 1) * d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := (n * (2 * a₁ + (n - 1) * d)) / 2

/-- Sum of first n sums of an arithmetic sequence -/
noncomputable def T (a₁ d : ℝ) (n : ℕ) : ℝ := (n * (n + 1) * (3 * a₁ + (n - 1) * d)) / 6

/-- The theorem stating that given S₂₀₁₉, T₃₀₂₈ is uniquely determined and 3028 is the smallest such n -/
theorem unique_T_given_S₂₀₁₉ (a₁ d : ℝ) :
  ∃! (t : ℝ), T a₁ d 3028 = t ∧
  ∀ (m : ℕ), m < 3028 → ¬(∃! (t' : ℝ), T a₁ d m = t') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_T_given_S₂₀₁₉_l286_28637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_undefined_values_l286_28686

-- Define the function representing the denominator
def f (x k : ℝ) : ℝ := (x^2 + 3*x - 4) * (x - k)

-- Define a function that counts the number of distinct real roots
noncomputable def count_distinct_roots (k : ℝ) : ℕ :=
  if k ≠ 1 ∧ k ≠ -4 then 3 else 2

-- State the theorem
theorem count_undefined_values (k : ℝ) :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f x k = 0) ∧ S.card = count_distinct_roots k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_undefined_values_l286_28686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_decomposition_l286_28651

theorem fraction_decomposition :
  ∃ (a b c : ℕ) (d e f : ℕ),
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ (f > 0) ∧
    (674 : ℚ) / 385 = a / d + b / e + c / f ∧
    a + b + c = (Nat.digits 10 d).sum + (Nat.digits 10 e).sum + (Nat.digits 10 f).sum :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_decomposition_l286_28651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_eq_pi_times_cross_section_area_l286_28602

/-- A cylinder with a square cross-section along its axis -/
structure SquareCrossSectionCylinder where
  /-- The area of the square cross-section -/
  cross_section_area : ℝ
  /-- Assumption that the cross-section area is positive -/
  cross_section_area_pos : 0 < cross_section_area

/-- The lateral surface area of a cylinder with a square cross-section -/
noncomputable def lateral_surface_area (c : SquareCrossSectionCylinder) : ℝ :=
  Real.pi * c.cross_section_area

/-- Theorem: The lateral surface area of a cylinder with a square cross-section
    of area S is equal to πS -/
theorem lateral_surface_area_eq_pi_times_cross_section_area 
  (c : SquareCrossSectionCylinder) : 
  lateral_surface_area c = Real.pi * c.cross_section_area := by
  -- Unfold the definition of lateral_surface_area
  unfold lateral_surface_area
  -- The equation now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_eq_pi_times_cross_section_area_l286_28602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_relationship_l286_28694

/-- A differentiable function f satisfying specific properties -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is differentiable -/
axiom f_differentiable : Differentiable ℝ f

/-- f satisfies f(x+2) - f(x) = 2f(1) for all x -/
axiom f_property (x : ℝ) : f (x + 2) - f x = 2 * f 1

/-- The graph of y = f(x+1) is symmetric about the line x = -1 -/
axiom f_symmetric (x : ℝ) : f (x + 1) = f (-x - 1)

/-- f(x) = x² + 2xf'(2) when x ∈ [2,4] -/
axiom f_form (x : ℝ) (h : x ∈ Set.Icc 2 4) :
  f x = x^2 + 2 * x * (deriv f 2)

/-- The main theorem to prove -/
theorem f_relationship : f (-1/2) < f (16/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_relationship_l286_28694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_stone_loss_l286_28634

/-- Represents the loss incurred when a precious stone is broken --/
noncomputable def stone_break_loss (original_weight : ℝ) (original_value : ℝ) (ratio1 ratio2 : ℝ) : ℝ :=
  let k := original_value / (original_weight ^ 2)
  let weight1 := original_weight * (ratio1 / (ratio1 + ratio2))
  let weight2 := original_weight * (ratio2 / (ratio1 + ratio2))
  let value1 := k * (weight1 ^ 2)
  let value2 := k * (weight2 ^ 2)
  original_value - (value1 + value2)

/-- Theorem stating the loss incurred when a specific stone is broken --/
theorem specific_stone_loss :
  stone_break_loss 35 12250 2 5 = 5000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_stone_loss_l286_28634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_independence_l286_28695

/-- A straight line in a plane -/
structure Line where

/-- A point in a plane -/
structure Point where

/-- The radius of the incircle of a triangle -/
noncomputable def incircle_radius (p q r : Point) : ℝ := sorry

/-- A sequence of points on a line -/
def point_sequence (ℓ : Line) : ℕ → Point := sorry

/-- Define membership for Point in Line -/
instance : Membership Point Line where
  mem := sorry

theorem incircle_radius_independence
  (ℓ : Line) (P : Point) (r₁ : ℝ) (h_not_on : P ∉ ℓ)
  (h_equal_radii : ∀ i : ℕ, incircle_radius P (point_sequence ℓ i) (point_sequence ℓ (i + 1)) = r₁) :
  ∀ k j : ℕ, 
    incircle_radius P (point_sequence ℓ j) (point_sequence ℓ (j + k)) = 
    1/2 * (1 - (1 - 2*r₁)^k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_independence_l286_28695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l286_28670

def sequenceA (n : ℕ) : ℤ :=
  match n with
  | 0 => -1
  | n + 1 => 2 * sequenceA n + 2

theorem sequence_formula (n : ℕ) : sequenceA n = 2^n - 2 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    calc sequenceA (n + 1)
      = 2 * sequenceA n + 2 := rfl
      _ = 2 * (2^n - 2) + 2 := by rw [ih]
      _ = 2^(n+1) - 4 + 2 := by ring
      _ = 2^(n+1) - 2 := by ring

#eval sequenceA 5  -- Should output 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l286_28670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_bisection_value_l286_28631

-- Define the function f(x) = lg x + x - 2
noncomputable def f (x : ℝ) := Real.log x / Real.log 10 + x - 2

-- Define the bisection method
noncomputable def bisection_step (a b : ℝ) : ℝ := (a + b) / 2

-- Theorem statement
theorem second_bisection_value :
  let a₀ := 1
  let b₀ := 2
  let x₁ := bisection_step a₀ b₀
  let a₁ := if f x₁ < 0 then x₁ else a₀
  let b₁ := if f x₁ < 0 then b₀ else x₁
  bisection_step a₁ b₁ = 1.75 := by
  sorry

#eval (1.5 + 2) / 2  -- This will print 1.75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_bisection_value_l286_28631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_bisects_triangle_eccentricity_l286_28667

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  h_pos : side > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

theorem ellipse_bisects_triangle_eccentricity (e : Ellipse) (t : EquilateralTriangle) :
  t.side = 2 * Real.sqrt (e.a^2 - e.b^2) →  -- distance between foci
  e.a = (Real.sqrt 3 * t.side) / 2 →        -- ellipse bisects triangle sides
  eccentricity e = Real.sqrt 3 - 1 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_bisects_triangle_eccentricity_l286_28667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_rise_l286_28688

/-- The rise in water level when a cube is immersed in a rectangular vessel -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length : ℝ) (vessel_width : ℝ) 
  (h_cube_edge : cube_edge = 15) 
  (h_vessel_length : vessel_length = 20) 
  (h_vessel_width : vessel_width = 15) :
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 11.25 := by
  sorry

#check water_level_rise

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_rise_l286_28688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_for_equal_division_l286_28691

/-- Represents a piece of cake as a rational number between 0 and 1 -/
def CakePiece : Type := { r : ℚ // 0 < r ∧ r ≤ 1 }

instance : DecidableEq CakePiece := by
  sorry -- We need to implement this instance

/-- A list of cake pieces representing a division of the cake -/
def CakeDivision := List CakePiece

/-- Checks if a cake division is valid (sums to 1) -/
def isValidDivision (division : CakeDivision) : Prop :=
  (division.map (λ p => p.val)).sum = 1

/-- Checks if a cake division can be equally distributed among a given number of children -/
def canDistributeEqually (division : CakeDivision) (children : ℕ) : Prop :=
  ∃ (childShares : List CakeDivision),
    childShares.length = children ∧
    (∀ share ∈ childShares, isValidDivision share) ∧
    (∀ share ∈ childShares, (share.map (λ p => p.val)).sum = 1 / children) ∧
    (division.toFinset = (childShares.join).toFinset)

/-- The main theorem stating that 8 is the minimum number of pieces needed -/
theorem min_pieces_for_equal_division :
  ∃ (division : CakeDivision),
    division.length = 8 ∧
    isValidDivision division ∧
    canDistributeEqually division 4 ∧
    canDistributeEqually division 5 ∧
    ∀ (smallerDivision : CakeDivision),
      smallerDivision.length < 8 →
      ¬(canDistributeEqually smallerDivision 4 ∧ canDistributeEqually smallerDivision 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_for_equal_division_l286_28691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_equation_solution_l286_28621

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -2 * x else x^2 - 1

theorem root_equation_solution (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ x : ℝ, f x + 2 * Real.sqrt (1 - x^2) + |f x - 2 * Real.sqrt (1 - x^2)| - 2 * a * x - 4 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    x₃ - x₂ = 2 * (x₂ - x₁)) →
  a = (-3 + Real.sqrt 17) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_equation_solution_l286_28621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_at_intersection_l286_28615

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ
  center : Point

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- A homogeneous plate composed of rectangles -/
structure HomogeneousPlate where
  rectangles : List Rectangle
  centroid : Point

/-- Function to calculate the center of a rectangle -/
noncomputable def rectangleCenter (r : Rectangle) : Point :=
  { x := r.width / 2, y := r.height / 2 }

/-- Function to create a line passing through two rectangle centers -/
def lineThroughCenters (r1 r2 : Rectangle) : Line :=
  { p1 := r1.center, p2 := r2.center }

/-- Function to find the intersection of two lines -/
noncomputable def lineIntersection (l1 l2 : Line) : Point :=
  sorry -- Actual implementation would go here

/-- Theorem stating that the centroid of a homogeneous plate is at the intersection of lines through rectangle centers -/
theorem centroid_at_intersection (plate : HomogeneousPlate) : 
  ∃ (l1 l2 : Line), 
    (∃ (r1 r2 r3 r4 : Rectangle), 
      r1 ∈ plate.rectangles ∧ 
      r2 ∈ plate.rectangles ∧ 
      r3 ∈ plate.rectangles ∧ 
      r4 ∈ plate.rectangles ∧ 
      l1 = lineThroughCenters r1 r2 ∧ 
      l2 = lineThroughCenters r3 r4) →
    lineIntersection l1 l2 = plate.centroid := by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_at_intersection_l286_28615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_m_eq_plus_minus_sqrt_two_l286_28679

/-- The system of equations has a unique solution if and only if m = ±√2 -/
theorem unique_solution_iff_m_eq_plus_minus_sqrt_two (m : ℝ) : 
  (∃! p : ℝ × ℝ, let (x, y) := p; x^2 + y^2 - 1 = 0 ∧ y - x - m = 0) ↔ m = Real.sqrt 2 ∨ m = -Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_m_eq_plus_minus_sqrt_two_l286_28679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_one_l286_28689

/-- The sum of the infinite series Σ(n=1 to ∞) of (n^4 + 5n^2 + 12n + 12) / (2^n * (n^4 + 16)) -/
noncomputable def infinite_series_sum : ℝ :=
  ∑' n, (n^4 + 5*n^2 + 12*n + 12) / (2^n * (n^4 + 16))

/-- The infinite series sum is equal to 1 -/
theorem infinite_series_sum_eq_one : infinite_series_sum = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_one_l286_28689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_bike_speed_ratio_l286_28683

/-- The average speed of a vehicle given distance and time -/
noncomputable def averageSpeed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem car_bike_speed_ratio :
  let tractorDistance : ℝ := 575
  let tractorTime : ℝ := 23
  let carDistance : ℝ := 360
  let carTime : ℝ := 4
  let tractorSpeed := averageSpeed tractorDistance tractorTime
  let bikeSpeed := 2 * tractorSpeed
  let carSpeed := averageSpeed carDistance carTime
  carSpeed / bikeSpeed = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_bike_speed_ratio_l286_28683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_equals_10_l286_28635

/-- A discrete uniform random variable on {1, 2, ..., n} -/
def X (n : ℕ) : Type := Fin n

/-- The probability of X being less than 4 -/
noncomputable def prob_X_less_than_4 (n : ℕ) : ℝ := (min 3 n : ℝ) / n

/-- The theorem stating that n = 10 given the conditions -/
theorem n_equals_10 :
  ∃ (n : ℕ), n > 0 ∧ prob_X_less_than_4 n = 0.3 ∧ n = 10 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_equals_10_l286_28635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_T_coordinates_l286_28687

/-- Triangle DEF with vertices D(-2,6), E(4,-2), and F(10,-2) -/
def triangle_DEF : Set (ℝ × ℝ) :=
  {⟨-2, 6⟩, ⟨4, -2⟩, ⟨10, -2⟩}

/-- Point T on line DF -/
noncomputable def T : ℝ × ℝ := ⟨4 * Real.sqrt 2, -8 * Real.sqrt 2 / 3 + 22 / 3⟩

/-- Point U on line EF -/
noncomputable def U : ℝ × ℝ := ⟨4 * Real.sqrt 2, -2⟩

/-- Triangle TUF -/
noncomputable def triangle_TUF : Set (ℝ × ℝ) :=
  {T, U, ⟨10, -2⟩}

/-- The area of triangle TUF is 16 -/
axiom area_TUF : MeasureTheory.MeasureSpace.volume triangle_TUF = 16

/-- The theorem to be proved -/
theorem difference_T_coordinates :
  |T.1 - T.2| = |20 * Real.sqrt 2 - 22| / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_T_coordinates_l286_28687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_l286_28624

noncomputable section

/-- The first function --/
def f (x : ℝ) : ℝ := 1 + x - x^2

/-- The second function --/
def g (x : ℝ) : ℝ := 0.5 * (x^2 + 3)

/-- First potential tangent line --/
def h₁ (x : ℝ) : ℝ := x + 1

/-- Second potential tangent line --/
def h₂ (x : ℝ) : ℝ := -1/3 * x + 13/9

/-- Theorem stating that h₁ and h₂ are common tangents to f and g --/
theorem common_tangents :
  (∃ x₁ : ℝ, f x₁ = h₁ x₁ ∧ (deriv f x₁ : ℝ) = deriv h₁ x₁) ∧
  (∃ x₂ : ℝ, f x₂ = h₂ x₂ ∧ (deriv f x₂ : ℝ) = deriv h₂ x₂) ∧
  (∃ x₃ : ℝ, g x₃ = h₁ x₃ ∧ (deriv g x₃ : ℝ) = deriv h₁ x₃) ∧
  (∃ x₄ : ℝ, g x₄ = h₂ x₄ ∧ (deriv g x₄ : ℝ) = deriv h₂ x₄) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_l286_28624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_equal_quadrilateral_area_l286_28696

/-- A convex quadrilateral with the property that line segments connecting
    midpoints of opposite sides are equal. -/
structure MidpointEqualQuadrilateral where
  -- The lengths of the diagonals
  diag1 : ℝ
  diag2 : ℝ
  -- The property that line segments connecting midpoints of opposite sides are equal
  midpoint_property : True

/-- The area of a MidpointEqualQuadrilateral -/
noncomputable def area (q : MidpointEqualQuadrilateral) : ℝ :=
  (1 / 2) * q.diag1 * q.diag2

/-- Theorem: The area of a MidpointEqualQuadrilateral with diagonals 8 and 12 is 48 -/
theorem midpoint_equal_quadrilateral_area :
  ∀ q : MidpointEqualQuadrilateral, q.diag1 = 8 ∧ q.diag2 = 12 → area q = 48 := by
  intro q ⟨h1, h2⟩
  unfold area
  rw [h1, h2]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_equal_quadrilateral_area_l286_28696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paving_rate_calculation_l286_28643

/-- Calculates the rate per square meter for paving a floor given the room dimensions and total cost. -/
theorem paving_rate_calculation (length width total_cost : ℚ) 
  (h1 : length = 5.5)
  (h2 : width = 4)
  (h3 : total_cost = 18700) : 
  total_cost / (length * width) = 850 := by
  sorry

#eval (18700 : ℚ) / (5.5 * 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paving_rate_calculation_l286_28643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l286_28684

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

-- State the theorem
theorem max_value_f_on_interval :
  ∃ (max : ℝ), max = π ∧
  ∀ x, x ∈ Set.Icc (-π) 0 → f x ≤ max := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l286_28684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_problem_l286_28672

noncomputable def coefficient_x_4 (p : Polynomial ℤ) : ℤ := p.coeff 4

theorem math_problem :
  (∃ x : ℕ, x ≠ 4 ∧ Nat.choose 28 x = Nat.choose 28 (3*x - 8)) ∧
  (coefficient_x_4 ((X - 1) * (X - 2) * (X - 3) * (X - 4) * (X - 5)) = -15) ∧
  (Nat.pow 3 8 % 5 = 1) ∧
  (Finset.card (Finset.powerset {1, 5, 10, 20, 50} \ {∅}) = 31) :=
by
  sorry

where
  X : Polynomial ℤ := Polynomial.X

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_problem_l286_28672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_3_equals_4_l286_28653

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2^x else x - 1

theorem f_f_3_equals_4 : f (f 3) = 4 := by
  -- Evaluate f(3)
  have h1 : f 3 = 2 := by
    simp [f]
    norm_num
  
  -- Evaluate f(f(3)), which is f(2)
  have h2 : f 2 = 4 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc
    f (f 3) = f 2 := by rw [h1]
    _       = 4   := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_3_equals_4_l286_28653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l286_28608

-- Define the constants
noncomputable def a : ℝ := Real.log 5 / Real.log 2
noncomputable def b : ℝ := Real.log 6 / Real.log 2
noncomputable def c : ℝ := (9 : ℝ) ^ (1/2)

-- State the theorem
theorem order_of_abc : c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l286_28608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l286_28680

-- Define the function f(x) = ln x
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the point of tangency
noncomputable def point_of_tangency : ℝ × ℝ := (Real.exp 1, f (Real.exp 1))

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ x y : ℝ,
    (x = point_of_tangency.fst ∧ y = point_of_tangency.snd) ∨
    (y = m * x + b) →
    x - Real.exp 1 * y = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l286_28680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_point_cycle_exists_l286_28661

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1/3 then 2*x + 1/3
  else if 1/3 ≤ x ∧ x ≤ 1 then 3/2*(1 - x)
  else 0

theorem five_point_cycle_exists : ∃ (x₀ x₁ x₂ x₃ x₄ : ℝ),
  0 ≤ x₀ ∧ x₀ ≤ 1 ∧
  0 ≤ x₁ ∧ x₁ ≤ 1 ∧
  0 ≤ x₂ ∧ x₂ ≤ 1 ∧
  0 ≤ x₃ ∧ x₃ ≤ 1 ∧
  0 ≤ x₄ ∧ x₄ ≤ 1 ∧
  x₀ ≠ x₁ ∧ x₀ ≠ x₂ ∧ x₀ ≠ x₃ ∧ x₀ ≠ x₄ ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧
  x₃ ≠ x₄ ∧
  f x₀ = x₁ ∧
  f x₁ = x₂ ∧
  f x₂ = x₃ ∧
  f x₃ = x₄ ∧
  f x₄ = x₀ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_point_cycle_exists_l286_28661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_strawberry_count_l286_28646

/-- The number of strawberry jelly beans Camilla originally had -/
def s : ℚ := sorry

/-- The number of blueberry jelly beans Camilla originally had -/
def b : ℚ := sorry

/-- The number of cherry jelly beans Camilla originally had -/
def c : ℚ := sorry

/-- Initial conditions -/
axiom initial_strawberry : s = 3 * c
axiom initial_blueberry : b = 2 * c

/-- Conditions after eating jelly beans -/
axiom after_eating_strawberry : s - 5 = 5 * (c - 10)
axiom after_eating_blueberry : b - 15 = 4 * (c - 10)

/-- Theorem stating that the original number of strawberry jelly beans is 67.5 -/
theorem original_strawberry_count : s = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_strawberry_count_l286_28646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_age_is_60_l286_28619

/-- Represents the present age of Neha -/
def neha_age : ℕ := sorry

/-- Represents the present age of Neha's mother -/
def mother_age : ℕ := sorry

/-- Neha's mother was 4 times her age 12 years ago -/
axiom past_condition : mother_age - 12 = 4 * (neha_age - 12)

/-- Neha's mother will be twice as old as Neha 12 years from now -/
axiom future_condition : mother_age + 12 = 2 * (neha_age + 12)

/-- The theorem states that given the conditions, Neha's mother's present age is 60 years -/
theorem mother_age_is_60 : mother_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_age_is_60_l286_28619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_inequality_l286_28673

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℂ)

-- Define the sides of the triangle
noncomputable def Triangle.a (t : Triangle) : ℝ := Complex.abs (t.B - t.C)
noncomputable def Triangle.b (t : Triangle) : ℝ := Complex.abs (t.C - t.A)
noncomputable def Triangle.c (t : Triangle) : ℝ := Complex.abs (t.A - t.B)

-- Define the distance between two points
noncomputable def distance (z w : ℂ) : ℝ := Complex.abs (z - w)

-- State the theorem
theorem erdos_inequality (t : Triangle) (P : ℂ) :
  t.a * (distance P t.A)^2 + t.b * (distance P t.B)^2 + t.c * (distance P t.C)^2 ≥ t.a * t.b * t.c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_inequality_l286_28673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l286_28617

/-- The speed of a train in km/hr given its length and time to cross a point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: A train 400 meters long crossing an electric pole in 9.99920006399488 seconds
    has a speed of approximately 144.03 km/hr -/
theorem train_speed_calculation :
  let length : ℝ := 400
  let time : ℝ := 9.99920006399488
  |train_speed length time - 144.03| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l286_28617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l286_28620

theorem book_price_change (P : ℝ) (h : P > 0) : 
  P * (1 - 0.3) * (1 + 0.2) = P * (1 - 0.16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l286_28620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c₁_value_c₁_half_valid_smallest_c₁_is_half_l286_28640

def sequence_c (c₁ : ℝ) : ℕ → ℝ
  | 0 => c₁
  | n + 1 => 13 * sequence_c c₁ n - 2 * (n + 1)

theorem smallest_c₁_value (c₁ : ℝ) :
  (∀ n : ℕ, sequence_c c₁ n > 0) →
  c₁ ≥ 1/2 :=
by sorry

theorem c₁_half_valid :
  ∀ n : ℕ, sequence_c (1/2) n > 0 :=
by sorry

theorem smallest_c₁_is_half :
  ∃ c₁ : ℝ, c₁ = 1/2 ∧
  (∀ n : ℕ, sequence_c c₁ n > 0) ∧
  ∀ c : ℝ, (∀ n : ℕ, sequence_c c n > 0) → c ≥ c₁ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c₁_value_c₁_half_valid_smallest_c₁_is_half_l286_28640
