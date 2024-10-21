import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_total_time_l1207_120758

/-- Represents the journey from home to the library -/
structure Journey where
  total_distance : ℚ
  walking_speed : ℚ
  jogging_speed : ℚ
  walking_time : ℚ
  jogging_time : ℚ

/-- The conditions of Sarah's journey -/
def sarah_journey : Journey where
  total_distance := 1  -- Normalized to 1 for simplicity
  walking_speed := 1/27  -- 1/3 distance in 9 minutes: 1/3 ÷ 9 = 1/27
  jogging_speed := 4/27  -- 4 times walking speed
  walking_time := 9
  jogging_time := 2/3 / (4/27)  -- 2/3 distance ÷ jogging speed

/-- The theorem to prove -/
theorem sarah_total_time :
  sarah_journey.walking_time + sarah_journey.jogging_time = 27/2 := by
  -- Expand the definition of sarah_journey
  unfold sarah_journey
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl

#eval (sarah_journey.walking_time + sarah_journey.jogging_time : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_total_time_l1207_120758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1207_120722

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + 2 * Real.cos x

/-- The interval [0, π/2] -/
def I : Set ℝ := Set.Icc 0 (Real.pi / 2)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1 + (3 * Real.sqrt 3) / 2 ∧ 
  (∀ x ∈ I, f x ≤ M) ∧
  (∃ x ∈ I, f x = M) := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1207_120722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1207_120748

/-- The ratio of the area of a perspective drawing to the original area --/
noncomputable def perspectiveRatio : ℝ := 1 / (2 * Real.sqrt 2)

/-- The side length of the square in the perspective drawing --/
def squareSideLength : ℝ := 3

/-- The area of the square in the perspective drawing --/
def perspectiveArea : ℝ := squareSideLength ^ 2

/-- The theorem stating that the area of the original parallelogram is 18√2 --/
theorem parallelogram_area : 
  perspectiveArea / perspectiveRatio = 18 * Real.sqrt 2 := by
  -- Expand the definitions
  unfold perspectiveArea perspectiveRatio squareSideLength
  -- Simplify the left-hand side
  simp [Real.sqrt_mul_self]
  -- The proof is completed with sorry as requested
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1207_120748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_inequality_l1207_120765

-- Define a circle
class Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral (c : Circle) where
  A : PointOnCircle c
  B : PointOnCircle c
  C : PointOnCircle c
  D : PointOnCircle c
  distinct : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A
  order : True  -- Representing the order A, B, C, D on the circle

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem cyclic_quadrilateral_inequality 
  (c : Circle) 
  (quad : CyclicQuadrilateral c) 
  (longest_side : ∀ (X Y : PointOnCircle c), 
    (X, Y) ≠ (quad.A, quad.B) → 
    distance quad.A.point quad.B.point ≥ distance X.point Y.point) :
  distance quad.A.point quad.B.point + distance quad.B.point quad.D.point >
  distance quad.A.point quad.C.point + distance quad.C.point quad.D.point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_inequality_l1207_120765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1207_120737

noncomputable def a (x : Real) : Real × Real := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))
noncomputable def b (x : Real) : Real × Real := (Real.cos (x / 2), -Real.sin (x / 2))

theorem vector_properties (x : Real) (h : x ∈ Set.Icc 0 (Real.pi / 2)) :
  (((a x).1 * (b x).1 + (a x).2 * (b x).2 = 0) ↔ (x = Real.pi / 4)) ∧
  ((Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2) ≥ 1 ↔ x ∈ Set.Icc 0 (Real.pi / 3)) ∧
  (∃ m : Real, (∀ y ∈ Set.Icc 0 (Real.pi / 2),
    ((a y).1 * (b y).1 + (a y).2 * (b y).2) - 2 * m * Real.sqrt ((a y).1 + (b y).1)^2 + ((a y).2 + (b y).2)^2 ≥ -2) ∧
    ((a x).1 * (b x).1 + (a x).2 * (b x).2) - 2 * m * Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2 = -2 ↔
    m = Real.sqrt 2 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1207_120737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_critical_point_implies_k_range_l1207_120712

open Set Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := exp x / x + k / 2 * x^2 - k * x

theorem unique_critical_point_implies_k_range :
  ∀ k : ℝ, (∀ x : ℝ, x > 0 → (∀ y : ℝ, y > 0 ∧ y ≠ 1 → deriv (f k) x ≠ 0) ∧ deriv (f k) 1 = 0) →
  k ∈ Ici (-exp 2 / 4) :=
by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_critical_point_implies_k_range_l1207_120712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_third_l1207_120755

/-- A rectangle in 2D space --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The specific rectangle from the problem --/
def problemRectangle : Rectangle where
  x_min := 0
  x_max := 4
  y_min := 0
  y_max := 3
  h_x := by norm_num
  h_y := by norm_num

/-- The area of a rectangle --/
def area (r : Rectangle) : ℝ := (r.x_max - r.x_min) * (r.y_max - r.y_min)

/-- The area satisfying the condition x < 2y --/
noncomputable def area_satisfying_condition (r : Rectangle) : ℝ := 
  let x_intersect := min r.x_max (2 * r.y_max)
  (x_intersect - r.x_min) * (x_intersect / 2 - r.y_min) / 2

/-- The probability of x < 2y for a point (x, y) randomly picked from the rectangle --/
noncomputable def probability (r : Rectangle) : ℝ :=
  (area_satisfying_condition r) / (area r)

theorem probability_is_one_third : 
  probability problemRectangle = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_third_l1207_120755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_purchase_price_l1207_120762

theorem machine_purchase_price (initial_price : ℝ) : 
  (∀ (year : ℕ), year ≤ 2 → 
    (initial_price * (1 - 0.1 * (year : ℝ)) = initial_price - initial_price * 0.1 * (year : ℝ))) →
  (initial_price * (1 - 0.1 * 2) = 6400) →
  initial_price = 8000 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_purchase_price_l1207_120762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_plus_four_cos_max_l1207_120700

theorem sin_squared_plus_four_cos_max (x : ℝ) : (Real.sin x) ^ 2 + 4 * (Real.cos x) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_plus_four_cos_max_l1207_120700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1207_120701

/-- An arithmetic sequence with common difference d ≠ 0 -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1 : ℚ) * seq.d) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h1 : seq.a 3 = S seq 5)
    (h2 : seq.a 2 * seq.a 4 = S seq 4) :
  (∀ n, seq.a n = 2 * n - 6) ∧
  (∀ n < 7, S seq n ≤ seq.a n) ∧
  S seq 7 > seq.a 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1207_120701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_tiling_l1207_120763

-- Define the internal angle of a regular polygon with n sides
noncomputable def internal_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

-- Define the possible regular polygons
inductive RegularPolygon
| triangle
| square
| pentagon
| hexagon

-- Function to get the number of sides for each regular polygon
def sides_of_polygon : RegularPolygon → ℕ
| RegularPolygon.triangle => 3
| RegularPolygon.square => 4
| RegularPolygon.pentagon => 5
| RegularPolygon.hexagon => 6

-- Theorem stating that only squares can be used with octagons for tiling
theorem octagon_tiling (p : RegularPolygon) :
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ 
    m * internal_angle 8 + n * internal_angle (sides_of_polygon p) = 360) ↔ 
  p = RegularPolygon.square :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_tiling_l1207_120763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1207_120744

-- Define the points as pairs of real numbers
variable (F M P N : ℝ × ℝ)

-- Define the conditions
def condition_F (F : ℝ × ℝ) : Prop := F = (1, 0)
def condition_M (M : ℝ × ℝ) : Prop := M.2 = 0
def condition_P (P : ℝ × ℝ) : Prop := P.1 = 0
def condition_MN_MP (M P N : ℝ × ℝ) : Prop := N - M = 2 • (P - M)
def condition_PM_perp_PF (F M P : ℝ × ℝ) : Prop := (P.1 - M.1) * (F.1 - P.1) + (P.2 - M.2) * (F.2 - P.2) = 0

-- Theorem statement
theorem trajectory_equation 
  (hF : condition_F F)
  (hM : condition_M M)
  (hP : condition_P P)
  (hMN_MP : condition_MN_MP M P N)
  (hPM_perp_PF : condition_PM_perp_PF F M P) :
  N.2^2 = 4 * N.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1207_120744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_four_fifths_l1207_120731

noncomputable def seriesSum (x : ℝ) : ℝ := 1 + (4 * x) / (1 - x)

def equation (x : ℝ) : Prop := seriesSum x = 85

theorem solution_is_four_fifths :
  ∃ x : ℝ, equation x ∧ -1 < x ∧ x < 1 ∧ x = 4/5 := by
  use 4/5
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_four_fifths_l1207_120731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_fixed_points_l1207_120795

/-- Ellipse C with equation x²/4 + y²/2 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

/-- Point A, left vertex of the ellipse -/
def point_A : ℝ × ℝ := (-2, 0)

/-- Point B, right vertex of the ellipse -/
def point_B : ℝ × ℝ := (2, 0)

/-- Line l: x = -3 -/
def line_l (x : ℝ) : Prop := x = -3

/-- Point M on ellipse C -/
def point_M (x₀ y₀ : ℝ) : Prop := ellipse_C x₀ y₀ ∧ x₀ ≠ 2 ∧ x₀ ≠ -2

/-- Point P on line l and line AM -/
noncomputable def point_P (x₀ y₀ : ℝ) : ℝ × ℝ := (-3, -y₀ / (x₀ + 2))

/-- Point Q on line l and line BM -/
noncomputable def point_Q (x₀ y₀ : ℝ) : ℝ × ℝ := (-3, -5 * y₀ / (x₀ - 2))

/-- Circle with diameter PQ -/
def circle_PQ (x y x₀ y₀ : ℝ) : Prop :=
  (x + 3)^2 + y^2 + ((y₀ / (x₀ + 2)) + (5 * y₀ / (x₀ - 2))) * y - 5/2 = 0

/-- Fixed points that the circle always passes through -/
noncomputable def fixed_points : Set (ℝ × ℝ) := {(-3 - Real.sqrt 10 / 2, 0), (-3 + Real.sqrt 10 / 2, 0)}

theorem circle_passes_through_fixed_points (x₀ y₀ : ℝ) (h : point_M x₀ y₀) :
  ∀ (p : ℝ × ℝ), p ∈ fixed_points → circle_PQ p.1 p.2 x₀ y₀ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_fixed_points_l1207_120795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_adjoining_squares_l1207_120734

/-- The area of the shaded region formed by two adjoining squares -/
theorem shaded_area_of_adjoining_squares : 
  ∀ (small_side large_side : ℝ),
  small_side = 4 →
  large_side = 10 →
  (small_side^2 - (1/2) * small_side * 
    ((large_side/2) / (large_side/2 + small_side)) * small_side) = 92/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_adjoining_squares_l1207_120734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alden_nephew_ratio_l1207_120779

/-- Represents the number of nephews Alden has now -/
def alden_nephews_now : ℕ := 100

/-- Represents the number of nephews Alden had 10 years ago -/
def alden_nephews_past : ℕ := 50

/-- Represents the number of nephews Vihaan has now -/
def vihaan_nephews : ℕ := alden_nephews_now + 60

/-- The total number of nephews Alden and Vihaan have -/
def total_nephews : ℕ := 260

/-- Theorem stating the ratio of Alden's past nephews to current nephews -/
theorem alden_nephew_ratio :
  alden_nephews_now = 100 ∧
  alden_nephews_past * 2 = alden_nephews_now :=
by
  constructor
  · rfl
  · rfl

#check alden_nephew_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alden_nephew_ratio_l1207_120779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roper_lawn_cuts_l1207_120750

/-- Represents the number of times Mr. Roper cuts his lawn per month from April to September -/
def cuts_apr_sep : ℕ := 15

/-- Represents the number of times Mr. Roper cuts his lawn per month from October to March -/
def cuts_oct_mar : ℕ := 3

/-- Represents the average number of times Mr. Roper cuts his lawn per month over the entire year -/
def avg_cuts_per_month : ℕ := 9

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the number of months from April to September -/
def months_apr_sep : ℕ := 6

/-- Represents the number of months from October to March -/
def months_oct_mar : ℕ := 6

theorem roper_lawn_cuts : cuts_apr_sep = 15 := by
  -- The proof goes here
  sorry

#check roper_lawn_cuts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roper_lawn_cuts_l1207_120750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_and_tangent_line_l1207_120784

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

noncomputable def f' (a x : ℝ) : ℝ := x^2 - 2*a*x + (a^2 - 1)

theorem extreme_point_and_tangent_line (a b : ℝ) :
  (f' a 1 = 0) ∧ 
  (f a b 1 = 2) ∧ 
  (f' a 1 = -1) →
  (a = 1 ∧ b = 8/3) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f 1 (8/3) x ≤ 8) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f 1 (8/3) x ≥ -4) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 4, f 1 (8/3) x = 8) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 4, f 1 (8/3) x = -4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_and_tangent_line_l1207_120784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_with_area_three_l1207_120727

noncomputable section

/-- The power function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The line l: x + y - 3 = 0 -/
def l (x y : ℝ) : Prop := x + y - 3 = 0

/-- Point A where line l intersects x-axis -/
def A : ℝ × ℝ := (3, 0)

/-- Point B where line l intersects y-axis -/
def B : ℝ × ℝ := (0, 3)

/-- The area of triangle ABP given point P(x, f(x)) -/
noncomputable def area_ABP (x : ℝ) : ℝ := 
  (3 * |x + f x - 3|) / 2

/-- The main theorem -/
theorem four_points_with_area_three : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, area_ABP x = 3) ∧ S.card = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_with_area_three_l1207_120727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1207_120776

noncomputable def vector_projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let magnitude_squared := w.1 * w.1 + w.2 * w.2
  (dot_product / magnitude_squared * w.1, dot_product / magnitude_squared * w.2)

theorem projection_theorem (u : ℝ × ℝ) :
  vector_projection (2, 6) u = (6/5, -2/5) →
  vector_projection (3, 2) u = (21/10, -7/10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1207_120776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_part_speed_approx_33_l1207_120715

/-- A driver's trip with two parts of different speeds -/
structure TripData where
  total_distance : ℝ
  first_part_distance : ℝ
  first_part_speed : ℝ
  average_speed : ℝ

/-- Calculate the speed of the second part of the trip -/
noncomputable def second_part_speed (trip : TripData) : ℝ :=
  let first_part_time := trip.first_part_distance / trip.first_part_speed
  let total_time := trip.total_distance / trip.average_speed
  let second_part_time := total_time - first_part_time
  let second_part_distance := trip.total_distance - trip.first_part_distance
  second_part_distance / second_part_time

/-- Theorem stating that the speed of the second part is approximately 33 km/h -/
theorem second_part_speed_approx_33 (trip : TripData) 
  (h1 : trip.total_distance = 50)
  (h2 : trip.first_part_distance = 25)
  (h3 : trip.first_part_speed = 66)
  (h4 : trip.average_speed = 44.00000000000001) :
  ∃ ε > 0, |second_part_speed trip - 33| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_part_speed_approx_33_l1207_120715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_shifted_tan_l1207_120703

/-- Defines the symmetry center of a shifted tangent function. -/
def SymmetryCenter (f : ℝ → ℝ) (center : ℝ × ℝ) : Prop :=
  ∀ x : ℝ, f (center.1 + x) = f (center.1 - x)

/-- Theorem stating the symmetry center of a shifted tangent function. -/
theorem symmetry_center_of_shifted_tan (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ Real.tan (x - π / 3)
  let center : ℝ × ℝ := (k * π / 2 + π / 3, 0)
  SymmetryCenter f center := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_shifted_tan_l1207_120703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_perfect_squares_l1207_120782

/-- The sequence y defined by the given recurrence relation -/
def y : ℕ → ℤ → ℤ
  | 0, _ => 1
  | 1, _ => 1
  | (n + 2), k => (4 * k - 5) * y (n + 1) k - y n k + 4 - 2 * k

/-- A predicate to check if a number is a perfect square -/
def isPerfectSquare (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m * m

/-- The main theorem stating the conditions for k -/
theorem sequence_perfect_squares :
  ∀ k : ℤ, (∀ n : ℕ, isPerfectSquare (y n k)) ↔ (k = 1 ∨ k = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_perfect_squares_l1207_120782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_y_minus_xy_squared_x_squared_minus_xy_plus_y_squared_l1207_120794

-- Define x and y as noncomputable
noncomputable def x : ℝ := 1 / (3 - 2 * Real.sqrt 2)
noncomputable def y : ℝ := 1 / (3 + 2 * Real.sqrt 2)

-- Theorem for the first question
theorem x_squared_y_minus_xy_squared : x^2 * y - x * y^2 = 4 * Real.sqrt 2 := by
  sorry

-- Theorem for the second question
theorem x_squared_minus_xy_plus_y_squared : x^2 - x * y + y^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_y_minus_xy_squared_x_squared_minus_xy_plus_y_squared_l1207_120794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_10_l1207_120749

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  M : Point
  N : Point
  P : Point
  Q : Point

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Checks if a line is perpendicular to another line -/
def is_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Area of rectangle MNPQ is 10 -/
theorem rectangle_area_is_10 
  (MNPQ : Rectangle) 
  (R S : Line) 
  (h1 : distance MNPQ.P MNPQ.Q = 5)
  (h2 : are_parallel R S)
  (h3 : is_perpendicular R (Line.mk (MNPQ.Q.x - MNPQ.P.x) (MNPQ.Q.y - MNPQ.P.y) 0))
  (h4 : is_perpendicular S (Line.mk (MNPQ.Q.x - MNPQ.P.x) (MNPQ.Q.y - MNPQ.P.y) 0))
  (h5 : ∃ X Y : Point, distance MNPQ.P X = 1 ∧ distance X Y = 2 ∧ distance Y MNPQ.Q = 2) :
  (distance MNPQ.M MNPQ.N) * (distance MNPQ.N MNPQ.P) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_10_l1207_120749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l1207_120745

/-- Roots of the cubic equation x^3 - 3x - b = 0 -/
noncomputable def cubicRoots (b : ℝ) : Set ℝ :=
  if b = 1 then
    {2 * Real.cos (Real.pi / 9), 2 * Real.cos (7 * Real.pi / 9), 2 * Real.cos (13 * Real.pi / 9)}
  else if b = Real.sqrt 3 then
    {2 * Real.cos (Real.pi / 18), 2 * Real.cos (13 * Real.pi / 18), 2 * Real.cos (25 * Real.pi / 18)}
  else
    ∅

/-- Theorem stating that the roots of the cubic equation x^3 - 3x - b = 0 are correct -/
theorem cubic_equation_roots (b : ℝ) (x : ℝ) :
  x ∈ cubicRoots b ↔ x^3 - 3*x - b = 0 := by
  sorry

#check cubic_equation_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l1207_120745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_binary_numbers_l1207_120774

/-- Represents a positive integer in base-2 -/
def Base2Repr := List Bool

/-- Converts a natural number to its base-2 representation -/
def toBase2 (n : ℕ) : Base2Repr :=
  sorry

/-- Counts the number of 1's in a base-2 representation -/
def countOnes (repr : Base2Repr) : ℕ :=
  sorry

/-- Counts the number of 0's in a base-2 representation -/
def countZeros (repr : Base2Repr) : ℕ :=
  sorry

/-- Checks if a base-2 representation has more 1's than 0's -/
def hasMoreOnesThanZeros (repr : Base2Repr) : Bool :=
  countOnes repr > countZeros repr

/-- Checks if a base-2 representation has an equal number of 1's and 0's -/
def hasEqualOnesAndZeros (repr : Base2Repr) : Bool :=
  countOnes repr = countZeros repr

/-- The set of positive integers less than or equal to 1000 -/
def numbersUpTo1000 : Finset ℕ :=
  Finset.range 1000

theorem sum_of_special_binary_numbers :
  let M := (numbersUpTo1000.filter (fun n => hasMoreOnesThanZeros (toBase2 (n + 1)))).card
  let E := (numbersUpTo1000.filter (fun n => hasEqualOnesAndZeros (toBase2 (n + 1)))).card
  M + E = 627 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_binary_numbers_l1207_120774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_approx_l1207_120716

/-- The volume of a right circular cone -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The height of a right circular cone given its volume and radius -/
noncomputable def cone_height (v r : ℝ) : ℝ := (3 * v) / (Real.pi * r^2)

theorem funnel_height_approx : 
  let r : ℝ := 4
  let v : ℝ := 150
  let h : ℝ := cone_height v r
  ⌊h⌋₊ = 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_approx_l1207_120716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_b_value_l1207_120710

/-- Given a geometric sequence with first term 210, second term b (where b > 0), 
    and third term 35/36, prove that b is equal to the square root of (35 * 210 / 36). -/
theorem geometric_sequence_b_value (b : ℝ) (h1 : b > 0) 
  (h2 : ∃ r : ℝ, r ≠ 0 ∧ b = 210 * r ∧ 35 / 36 = b * r) :
  b = Real.sqrt ((35 * 210) / 36) := by
  sorry

#check geometric_sequence_b_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_b_value_l1207_120710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l1207_120789

theorem max_value_of_function :
  ∃ (M : ℝ), M = 4 ∧ 
  ∀ x : ℝ, Real.sqrt (2*x - 3) + Real.sqrt (2*x) + Real.sqrt (7 - 3*x) ≤ M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l1207_120789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_and_median_extension_l1207_120738

-- Define the points
def A : ℝ × ℝ := (-2, 1)
def B : ℝ × ℝ := (2, 5)
def C : ℝ × ℝ := (4, -1)
def D : ℝ × ℝ := (8, 3)

-- Define the vector between two points
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define a parallelogram
def is_parallelogram (a b c d : ℝ × ℝ) : Prop :=
  vector a b = vector d c ∧ vector a d = vector b c

-- Define a point on a line
def point_on_line (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))

-- Theorem statement
theorem parallelogram_and_median_extension :
  let M : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  is_parallelogram A B D C ∧ point_on_line A M D :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_and_median_extension_l1207_120738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1207_120788

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance from the right focus to the line x/a + y/b = 1 -/
noncomputable def Ellipse.focusToLine (e : Ellipse) : ℝ :=
  (e.b * Real.sqrt (e.a^2 - e.b^2) - e.a * e.b) / Real.sqrt (e.a^2 + e.b^2)

/-- The set of points on the ellipse -/
def Ellipse.points (e : Ellipse) : Set (ℝ × ℝ) :=
  {p | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1}

/-- Theorem about the properties of a specific ellipse -/
theorem ellipse_properties (e : Ellipse) 
    (h_ecc : e.eccentricity = 1/2)
    (h_dist : e.focusToLine = Real.sqrt 21 / 7) :
  ∃ (A B : ℝ × ℝ),
    -- 1. The equation of the ellipse
    (∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1) ↔ (x^2 / e.a^2 + y^2 / e.b^2 = 1)) ∧
    -- 2. Distance from origin to line AB is constant
    (A ∈ e.points ∧ B ∈ e.points ∧ (A.1 * B.1 + A.2 * B.2 = 0) →
      ∃ (k m : ℝ), (∀ (x : ℝ), m / Real.sqrt (k^2 + 1) = 2 * Real.sqrt 21 / 7)) ∧
    -- 3. Minimum length of chord AB
    (∀ (C D : ℝ × ℝ), C ∈ e.points ∧ D ∈ e.points →
      Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) ≥ 4 * Real.sqrt 21 / 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1207_120788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_numbers_sum_l1207_120732

theorem square_numbers_sum (a b c d : ℕ) :
  (a > 0) → (b > 0) → (c > 0) → (d > 0) →
  a * b + b * c + c * d + d * a = 40 →
  (a + b + c + d : ℕ) ∈ ({13, 14, 22, 41} : Finset ℕ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_numbers_sum_l1207_120732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrogen_percent_yield_is_33_33_l1207_120793

/-- Represents the chemical reaction between Iron and Sulfuric acid -/
structure ChemicalReaction where
  iron_moles : ℚ
  sulfuric_acid_moles : ℚ
  actual_hydrogen_yield : ℚ

/-- Calculates the percent yield of Hydrogen gas in the chemical reaction -/
def percent_yield (reaction : ChemicalReaction) : ℚ :=
  (reaction.actual_hydrogen_yield / reaction.iron_moles) * 100

/-- Theorem stating that the percent yield of Hydrogen gas is approximately 33.33% -/
theorem hydrogen_percent_yield_is_33_33 (reaction : ChemicalReaction) 
  (h1 : reaction.iron_moles = 3)
  (h2 : reaction.sulfuric_acid_moles = 4)
  (h3 : reaction.actual_hydrogen_yield = 1) :
  ∃ ε > 0, |percent_yield reaction - 33333 / 1000| < ε := by
  sorry

#eval percent_yield { iron_moles := 3, sulfuric_acid_moles := 4, actual_hydrogen_yield := 1 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrogen_percent_yield_is_33_33_l1207_120793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_iff_a_in_range_l1207_120751

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / Real.exp x + a * Real.log x

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := -1 / Real.exp x + a / x

-- Theorem statement
theorem f_not_monotonic_iff_a_in_range (a : ℝ) :
  (∃ x y, x > 0 ∧ y > 0 ∧ x < y ∧ f a x > f a y) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x < y ∧ f a x < f a y) ↔
  0 < a ∧ a < 1 / Real.exp 1 := by
  sorry

#check f_not_monotonic_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_iff_a_in_range_l1207_120751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_opposite_2023_l1207_120760

theorem reciprocal_of_opposite_2023 : (-(1 / 2023 : ℚ)) = -1/2023 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_opposite_2023_l1207_120760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_fewest_cookies_l1207_120792

/-- Represents a baker in the cookie competition -/
structure Baker where
  name : String
  cookieThickness : ℝ
  cookieVolume : ℝ

/-- Calculates the number of cookies a baker can make given a fixed amount of dough -/
noncomputable def numCookies (baker : Baker) (doughAmount : ℝ) : ℝ :=
  doughAmount / baker.cookieVolume

theorem david_fewest_cookies 
  (amy bob claire david : Baker)
  (doughAmount : ℝ)
  (h_amy : amy.name = "Amy" ∧ amy.cookieThickness = 0.5 ∧ amy.cookieVolume = 2 * Real.pi)
  (h_bob : bob.name = "Bob" ∧ bob.cookieThickness = 0.4 ∧ bob.cookieVolume = 3.6)
  (h_claire : claire.name = "Claire" ∧ claire.cookieThickness = 0.3 ∧ claire.cookieVolume = 3.6)
  (h_david : david.name = "David" ∧ david.cookieThickness = 0.6 ∧ david.cookieVolume = 3.75 * Real.sqrt 3)
  (h_dough : doughAmount > 0)
  (h_amy_cookies : numCookies amy doughAmount = 15) :
  numCookies david doughAmount ≤ min (numCookies amy doughAmount) (min (numCookies bob doughAmount) (numCookies claire doughAmount)) := by
  sorry

#check david_fewest_cookies

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_fewest_cookies_l1207_120792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1207_120769

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the vectors m and n
noncomputable def m (t : Triangle) : ℝ × ℝ := (-Real.cos (t.A / 2), Real.sin (t.A / 2))
noncomputable def n (t : Triangle) : ℝ × ℝ := (Real.cos (t.A / 2), Real.sin (t.A / 2))

-- Define dot product for pairs
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : dot_product (m t) (n t) = 1/2)
  (h2 : Real.sqrt 2 * t.a = Real.sqrt 3 * t.b)
  (h3 : t.a = 2 * Real.sqrt 3)
  (h4 : 1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3) :
  t.B = π/4 ∧ t.a + t.b + t.c = 4 + 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1207_120769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_b_values_l1207_120799

-- Define the function g
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

-- Define the inverse function of g
noncomputable def g_inv (b : ℝ) (y : ℝ) : ℝ := 
  (b / y + 4) / 3

-- State the theorem
theorem product_of_b_values (b : ℝ) : 
  (g b 3 = g_inv b (b + 2)) → 
  (∃ b₁ b₂ : ℝ, b = b₁ ∨ b = b₂) ∧ 
  (∀ b₁ b₂ : ℝ, (g b₁ 3 = g_inv b₁ (b₁ + 2)) ∧ (g b₂ 3 = g_inv b₂ (b₂ + 2)) → b₁ * b₂ = -40/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_b_values_l1207_120799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_plus_cos_2theta_l1207_120718

theorem sin_2theta_plus_cos_2theta (θ : ℝ) 
  (h : (Real.cos θ, Real.sin θ) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1}) : 
  Real.sin (2 * θ) + Real.cos (2 * θ) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_plus_cos_2theta_l1207_120718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_linear_function_value_l1207_120702

/-- A right triangle with side lengths a, b, and hypotenuse c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The Pythagorean linear function for a given right triangle -/
noncomputable def pythagorean_linear_function (t : RightTriangle) (x : ℝ) : ℝ :=
  (t.a / t.c) * x + (t.b / t.c)

theorem pythagorean_linear_function_value (t : RightTriangle) :
  pythagorean_linear_function t (-1) = Real.sqrt 3 / 3 →
  2 = (1/2) * t.a * t.b →
  t.c = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_linear_function_value_l1207_120702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1207_120705

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola a b) : ℝ × ℝ :=
  (Real.sqrt (a^2 + b^2), 0)

/-- A point on the asymptote of the hyperbola in the first quadrant -/
noncomputable def asymptote_point (h : Hyperbola a b) (x : ℝ) : ℝ × ℝ :=
  (x, (b / a) * x)

/-- The intersection point of a line with slope -1 passing through the right focus
    and the asymptote of the hyperbola in the first quadrant -/
noncomputable def intersection_point (h : Hyperbola a b) : ℝ × ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  ((a * c) / (a + b), (b * c) / (a + b))

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

/-- Theorem stating the eccentricity of the hyperbola under given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) :
  let p := intersection_point h
  let f := right_focus h
  let o := (0, 0)
  (1 / 2) * f.1 * p.2 = (a^2 + b^2) / 8 →
  eccentricity h = Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1207_120705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_right_triangle_l1207_120773

theorem circumradius_right_triangle (a b c : ℝ) (h1 : a = 8) (h2 : b = 6) (h3 : c = 10) :
  let r := c / 2
  r = 5 := by
  rw [h3]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_right_triangle_l1207_120773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1207_120733

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt (1 - Real.sin x ^ 2)) / Real.cos x +
  (Real.sqrt (1 - Real.cos x ^ 2)) / Real.sin x

theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ π / 2 + π * Int.floor (x / π) ∧ f x = y) ↔ y ∈ ({-2, 0, 2} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1207_120733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_consecutive_red_balls_expectation_l1207_120720

-- Define the color type
inductive Color
| Red | Yellow | Blue | Green

-- Define the probability of drawing a red ball
noncomputable def prob_red : ℝ := 1 / 4

-- Define the probability of not drawing a red ball
noncomputable def prob_not_red : ℝ := 1 - prob_red

-- Define the expected number of draws
def expected_draws : ℝ := 20

-- State the theorem
theorem two_consecutive_red_balls_expectation :
  let E₀ := expected_draws -- Expected draws starting from no red balls
  let E₁ := E₀ - 4 -- Expected draws starting from one red ball
  E₀ = prob_red * (1 + E₁) + prob_not_red * (1 + E₀) ∧
  E₁ = prob_red * 1 + prob_not_red * (1 + E₀) ∧
  E₀ = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_consecutive_red_balls_expectation_l1207_120720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_cubic_roots_l1207_120704

theorem triangle_area_cubic_roots (r s t : ℝ) : 
  (r^3 - 4*r^2 + 5*r - 19/10 = 0) →
  (s^3 - 4*s^2 + 5*s - 19/10 = 0) →
  (t^3 - 4*t^2 + 5*t - 19/10 = 0) →
  r ≠ s → s ≠ t → r ≠ t →
  let p := (r + s + t) / 2
  let area := Real.sqrt (p * (p - r) * (p - s) * (p - t))
  area = Real.sqrt 5 / 5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_cubic_roots_l1207_120704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_equation_solutions_l1207_120724

open Real Set

theorem tan_cot_equation_solutions :
  ∃ (S : Set ℝ), 
    S ⊆ Ioo 0 (2 * π) ∧ 
    (∀ θ ∈ S, tan (7 * π * cos θ) = 1 / tan (7 * π * sin θ)) ∧
    (∃ (f : S → Fin 36), Function.Bijective f) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_equation_solutions_l1207_120724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_top_layer_lights_l1207_120739

/-- Represents the number of lights on each layer of a tower -/
def tower_lights : ℕ → ℕ := sorry

/-- The total number of layers in the tower -/
def total_layers : ℕ := 7

/-- The property that each layer has twice the lights of the layer above -/
axiom double_lights (n : ℕ) (h : n < total_layers) : 
  tower_lights (n + 1) = 2 * tower_lights n

/-- The total number of lights on all layers -/
def total_lights : ℕ := 381

/-- The sum of lights on all layers equals the total lights -/
axiom sum_equals_total : 
  (Finset.range total_layers).sum tower_lights = total_lights

theorem top_layer_lights : tower_lights 0 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_top_layer_lights_l1207_120739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_l1207_120743

/-- If the terminal side of angle θ passes through point P(3, -4), then tan(θ + π) = 4/3 -/
theorem tan_theta_plus_pi (θ : ℝ) (h : ∃ (k : ℝ), k * 3 = Real.cos θ ∧ k * (-4) = Real.sin θ) :
  Real.tan (θ + π) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_l1207_120743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_204_303_l1207_120707

/-- The product of terms from 2 to 100 -/
def series_product : ℕ → ℚ
  | 0 => 1  -- Base case for 0
  | 1 => 1  -- Base case for 1
  | n + 2 => (n + 2) * (n + 4) / ((n + 3) * (n + 3)) * series_product (n + 1)

theorem series_sum_equals_204_303 : series_product 100 = 204 / 303 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_204_303_l1207_120707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l1207_120713

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := 3*x - 4*y = 24

/-- Distance from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |3*x - 4*y - 24| / 5

/-- Theorem stating the maximum and minimum distances -/
theorem distance_bounds :
  ∃ (max_dist min_dist : ℝ),
    (∀ x y : ℝ, ellipse x y → distance_to_line x y ≤ max_dist) ∧
    (∃ x y : ℝ, ellipse x y ∧ distance_to_line x y = max_dist) ∧
    (∀ x y : ℝ, ellipse x y → distance_to_line x y ≥ min_dist) ∧
    (∃ x y : ℝ, ellipse x y ∧ distance_to_line x y = min_dist) ∧
    max_dist = 12/5 * (2 + Real.sqrt 2) ∧
    min_dist = 12/5 * (2 - Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l1207_120713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_special_integers_l1207_120711

theorem product_of_special_integers (E F G H : ℕ+) 
  (sum_eq : E + F + G + H = 100)
  (relation : E^2 = F - 4 ∧ E^2 = G + 6 ∧ E^2 * 2 = H) :
  E * F * G * H = 2106 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_special_integers_l1207_120711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1207_120747

-- Define the sequences a_n and b_n
def a : ℕ → ℚ := sorry

def b : ℕ → ℚ := sorry

-- Define the sum of the first n terms of b_n
def s : ℕ → ℚ := sorry

-- State the theorem
theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 3) ∧ 
  (b 1 = 1) ∧ 
  (b 2 = 1/3) ∧ 
  (∀ n : ℕ, n ≥ 1 → a n * b (n + 1) + b (n + 1) = n * b n) →
  (∀ n : ℕ, n ≥ 1 → a n = 3 * n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → b n = (1/3)^(n-1)) ∧
  (∀ n : ℕ, n ≥ 1 → s n = 3/2 - 1/(2 * 3^(n-1))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1207_120747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_as_610_degrees_l1207_120719

theorem same_terminal_side_as_610_degrees :
  ∃ (k : ℤ), ∀ (θ : ℝ), 
    (∃ (n : ℤ), θ = 360 * n + 610) ↔ (∃ (m : ℤ), θ = 360 * m + 250) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_as_610_degrees_l1207_120719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_early_arrival_speed_l1207_120736

/-- Represents the speed at which Mrs. Early needs to drive to arrive exactly on time -/
noncomputable def exact_speed : ℝ := 59

/-- The speed at which Mrs. Early would be 5 minutes late -/
noncomputable def late_speed : ℝ := 50

/-- The speed at which Mrs. Early would be 7 minutes early -/
noncomputable def early_speed : ℝ := 80

/-- The time in hours that Mrs. Early would be late if driving at late_speed -/
noncomputable def late_time : ℝ := 5 / 60

/-- The time in hours that Mrs. Early would be early if driving at early_speed -/
noncomputable def early_time : ℝ := 7 / 60

theorem early_arrival_speed :
  ∃ (d t : ℝ),
    d > 0 ∧ t > 0 ∧
    d = late_speed * (t + late_time) ∧
    d = early_speed * (t - early_time) ∧
    d = exact_speed * t := by
  sorry

#check early_arrival_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_early_arrival_speed_l1207_120736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_drink_volume_l1207_120766

/-- Represents the composition of a fruit drink -/
structure FruitDrink where
  orange_percent : ℚ
  watermelon_percent : ℚ
  grape_ounces : ℚ

/-- Calculates the total volume of a fruit drink given its composition -/
def total_volume (drink : FruitDrink) : ℚ :=
  drink.grape_ounces / (1 - drink.orange_percent - drink.watermelon_percent)

/-- Theorem stating the total volume of a specific fruit drink composition -/
theorem fruit_drink_volume (drink : FruitDrink) 
  (h1 : drink.orange_percent = 35/100)
  (h2 : drink.watermelon_percent = 35/100)
  (h3 : drink.grape_ounces = 45) :
  total_volume drink = 150 := by
  sorry

#eval total_volume { orange_percent := 35/100, watermelon_percent := 35/100, grape_ounces := 45 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_drink_volume_l1207_120766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1207_120777

/-- Ellipse C with given properties -/
structure EllipseC where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Point on the ellipse -/
structure PointOnEllipse (e : EllipseC) where
  x : ℝ
  y : ℝ
  h : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The equation of the ellipse and the fixed intersection point -/
theorem ellipse_properties (e : EllipseC) 
  (h1 : e.a^2 - e.b^2 = 3)  -- Foci at (-√3, 0) and (√3, 0)
  (h2 : e.b * Real.sqrt 3 / 2 = Real.sqrt 3 / 2)  -- Area of triangle
  (P : ℝ × ℝ) (hP : P = (0, 4))
  (M N : PointOnEllipse e) (hMN : M.x = -N.x ∧ M.y = N.y)
  (E : PointOnEllipse e) (hE : ∃ k, E.y = k * E.x + 4 ∧ N.y = k * N.x + 4) :
  (e.a = 2 ∧ e.b = 1) ∧  -- Equation of ellipse C
  (∃ t, t = 1/4 ∧ ∀ x, (M.y - t) * (E.x - M.x) = (E.y - M.y) * (x - M.x) → x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1207_120777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l1207_120781

theorem quadratic_function_proof (a b c y₁ y₂ y₃ : ℝ) : 
  a > 0 → 
  b > 0 → 
  y₁^2 = 1 → 
  y₂^2 = 1 → 
  y₃^2 = 1 → 
  c = y₁ → 
  a + b + c = y₂ → 
  a - b + c = y₃ → 
  ∃ (f : ℝ → ℝ), (∀ x, f x = x^2 + x - 1) ∧ (∀ x, a * x^2 + b * x + c = f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_proof_l1207_120781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_in_square_l1207_120798

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A square with side length 1 -/
def UnitSquare : Set Point := {p | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

theorem pigeonhole_in_square (points : Finset Point) (h : points.card = 5) 
    (h_in_square : ∀ p ∈ points, p ∈ UnitSquare) :
    ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ distance p q ≤ 1 / Real.sqrt 2 := by
  sorry

#check pigeonhole_in_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_in_square_l1207_120798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1207_120757

def M : Set ℤ := {1, 2}

def N : Set ℤ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1207_120757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sum_zero_l1207_120742

/-- A quadratic function -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_sum_zero (a b c : ℝ) :
  quadratic_function a b c 1 = 0 ∧
  quadratic_function a b c 5 = 0 ∧
  (∃ x, quadratic_function a b c x = 36 ∧ ∀ y, quadratic_function a b c y ≥ 36) →
  a + b + c = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sum_zero_l1207_120742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_filled_probability_l1207_120725

/-- Represents the total number of dumplings -/
def total_dumplings : ℕ := 5

/-- Represents the number of meat-filled dumplings -/
def meat_dumplings : ℕ := 2

/-- Represents the number of red bean paste-filled dumplings -/
def bean_dumplings : ℕ := 3

/-- Represents the number of dumplings selected -/
def selected_dumplings : ℕ := 2

/-- Calculates the probability of selecting two meat-filled dumplings given that the selected dumplings have the same filling -/
theorem meat_filled_probability : 
  (Nat.choose meat_dumplings selected_dumplings : ℚ) / (Nat.choose total_dumplings selected_dumplings) /
  ((Nat.choose meat_dumplings selected_dumplings + Nat.choose bean_dumplings selected_dumplings : ℚ) / 
   (Nat.choose total_dumplings selected_dumplings)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_filled_probability_l1207_120725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1207_120729

-- Define the function f(x) = x^(1/2)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem range_of_a (a : ℝ) : f (3 - a) > f a → 0 ≤ a ∧ a < 3/2 := by
  intro h
  have h1 : 0 ≤ a := by
    -- Proof that a is non-negative
    sorry
  have h2 : a < 3/2 := by
    -- Proof that a is less than 3/2
    sorry
  exact ⟨h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1207_120729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1207_120723

noncomputable def f (x : ℝ) : ℝ := 2 / (x^3)

theorem range_of_f :
  Set.range f = {y : ℝ | y < 0 ∨ y > 0} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1207_120723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_theorem_circle_l1207_120790

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define parallel chords
def ParallelChords (c : Circle) (a b d e : ℝ × ℝ) : Prop :=
  PointOnCircle c a ∧ PointOnCircle c b ∧ PointOnCircle c d ∧ PointOnCircle c e ∧
  (b.2 - a.2) * (e.1 - d.1) = (b.1 - a.1) * (e.2 - d.2)

-- Define a line
structure Line where
  a : ℝ × ℝ
  b : ℝ × ℝ

-- Define intersection of two lines
noncomputable def Intersect (l1 l2 : Line) : ℝ × ℝ :=
  sorry

-- Define if a point is on a line
def PointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  (p.2 - l.a.2) * (l.b.1 - l.a.1) = (p.1 - l.a.1) * (l.b.2 - l.a.2)

theorem pascal_theorem_circle (o : Circle) (a b c d e : ℝ × ℝ) :
  PointOnCircle o a ∧ PointOnCircle o b ∧ PointOnCircle o c ∧ 
  PointOnCircle o d ∧ PointOnCircle o e ∧
  ParallelChords o a b d e →
  let x := Intersect (Line.mk a c) (Line.mk b e)
  let y := Intersect (Line.mk c e) (Line.mk d a)
  let z := Intersect (Line.mk b x) (Line.mk d y)
  PointOnCircle o z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_theorem_circle_l1207_120790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_cuboid_width_l1207_120797

/-- Proves that the width of a larger cuboid is 15m given specific conditions -/
theorem larger_cuboid_width :
  ∃ large_width : ℝ,
    let small_length : ℝ := 6
    let small_width : ℝ := 4
    let small_height : ℝ := 3
    let large_length : ℝ := 18
    let large_height : ℝ := 2
    let num_small_cuboids : ℝ := 7.5
    large_width = 15 ∧
    large_length * large_width * large_height =
      num_small_cuboids * small_length * small_width * small_height :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_cuboid_width_l1207_120797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_a_inv_zero_l1207_120775

/-- The coefficient of a^(-1) in the expansion of (a - 1/a^2)^6 is 0 -/
theorem coeff_a_inv_zero (a : ℝ) (a_nonzero : a ≠ 0) :
  (Finset.range 7).sum (λ k ↦ (Finset.range 7).card.choose k * (-1)^k * a^(6 - 3*k)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_a_inv_zero_l1207_120775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_intersection_points_l1207_120741

/-- A type representing a line in a plane -/
structure Line where
  id : ℕ

/-- A type representing an intersection point of two lines -/
structure IntersectionPoint where
  line1 : Line
  line2 : Line

/-- The set of all lines in the plane -/
axiom lines : Finset Line

/-- The property that all lines are distinct -/
axiom lines_distinct : ∀ l1 l2, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → l1.id ≠ l2.id

/-- The number of lines in the set -/
axiom num_lines : Finset.card lines = 5

/-- The set of all intersection points -/
axiom intersection_points : Finset IntersectionPoint

/-- No three lines meet at a single point -/
axiom no_triple_intersection :
  ∀ p1 p2 p3, p1 ∈ intersection_points → p2 ∈ intersection_points → p3 ∈ intersection_points →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    (p1.line1 = p2.line1 ∧ p1.line1 = p3.line1) ∨
    (p1.line1 = p2.line2 ∧ p1.line1 = p3.line2) ∨
    (p1.line2 = p2.line1 ∧ p1.line2 = p3.line1) ∨
    (p1.line2 = p2.line2 ∧ p1.line2 = p3.line2) →
    False

/-- The theorem to be proved -/
theorem sum_of_intersection_points :
  Finset.card intersection_points = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_intersection_points_l1207_120741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_percentage_is_24_percent_l1207_120753

/-- Represents a circular design with five concentric circles -/
structure CircularDesign where
  -- The radius of the smallest circle
  smallest_radius : ℝ
  -- The increment in radius for each successive circle
  radius_increment : ℝ

/-- Calculates the area of a circle given its radius -/
noncomputable def circle_area (radius : ℝ) : ℝ := Real.pi * radius^2

/-- Calculates the total area of the design (area of the largest circle) -/
noncomputable def total_area (design : CircularDesign) : ℝ :=
  circle_area (design.smallest_radius + 4 * design.radius_increment)

/-- Calculates the area of the black regions in the design -/
noncomputable def black_area (design : CircularDesign) : ℝ :=
  circle_area design.smallest_radius +
  (circle_area (design.smallest_radius + 2 * design.radius_increment) -
   circle_area (design.smallest_radius + design.radius_increment))

/-- Calculates the percentage of the design that is black -/
noncomputable def black_percentage (design : CircularDesign) : ℝ :=
  (black_area design / total_area design) * 100

/-- The main theorem stating that the black percentage is 24% for the given design -/
theorem black_percentage_is_24_percent (design : CircularDesign)
  (h1 : design.smallest_radius = 3)
  (h2 : design.radius_increment = 3) :
  black_percentage design = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_percentage_is_24_percent_l1207_120753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_l1207_120714

theorem trigonometric_values (α : Real) 
  (h1 : Real.cos α = Real.sqrt 2 / 2) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.tan α = 1 ∧ Real.sin (α + Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_l1207_120714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1207_120721

-- Define the quadratic function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m - 1

-- Theorem for part (1)
theorem part_one (m : ℝ) : f m 0 = f m 2 → m = 2 := by sorry

-- Define the minimum value function
noncomputable def f_min (m : ℝ) : ℝ :=
  if m ≤ -4 then 3*m + 3
  else if m < 4 then -m^2/4 + m - 1
  else 3 - m

-- Theorem for part (2)
theorem part_two (m : ℝ) : 
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f m x ≥ f_min m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1207_120721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_with_center_M_and_radius_C_l1207_120708

noncomputable def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def circle_C (x y m : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + m = 0

noncomputable def tangent_circles (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_O x y ∧ circle_C x y m

noncomputable def tangent_point (m : ℝ) : ℝ × ℝ :=
  if m = 9 then (3/5, 4/5) else (-3/5, -4/5)

theorem circle_with_center_M_and_radius_C (m : ℝ) :
  tangent_circles m →
  let (mx, my) := tangent_point m
  (∀ x y, (x - mx)^2 + (y - my)^2 = 16) ∨
  (∀ x y, (x + mx)^2 + (y + my)^2 = 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_with_center_M_and_radius_C_l1207_120708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cm_to_m_conversion_hectare_to_sqm_conversion_kg_g_to_kg_m_cm_to_m_problem1_problem2_problem3_problem4_l1207_120728

-- Define conversion rates
noncomputable def cm_to_m : ℚ := 1 / 100
noncomputable def hectare_to_sqm : ℚ := 10000
noncomputable def g_to_kg : ℚ := 1 / 1000

-- Define the theorems to be proved
theorem cm_to_m_conversion (x : ℚ) : x * cm_to_m = x / 100 := by sorry

theorem hectare_to_sqm_conversion (x : ℚ) : x * hectare_to_sqm = x * 10000 := by sorry

theorem kg_g_to_kg (kg g : ℚ) : kg + g * g_to_kg = kg + g / 1000 := by sorry

theorem m_cm_to_m (m cm : ℚ) : m + cm * cm_to_m = m + cm / 100 := by sorry

-- Prove the specific conversions
theorem problem1 : 120 * cm_to_m = 1.2 := by sorry

theorem problem2 : 0.3 * hectare_to_sqm = 3000 := by sorry

theorem problem3 : 10 + 10 * g_to_kg = 10.01 := by sorry

theorem problem4 : 1 + 3 * cm_to_m = 1.03 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cm_to_m_conversion_hectare_to_sqm_conversion_kg_g_to_kg_m_cm_to_m_problem1_problem2_problem3_problem4_l1207_120728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1207_120785

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2 * x - 3)}
def B : Set ℝ := {x | (2 : ℝ)^(x - 2) < 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ici (3/2) ∩ Set.Iio 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1207_120785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_conditions_l1207_120706

theorem perfect_square_conditions (n : ℕ) :
  (∃ k : ℕ, 2^n + 3 = k^2) = (n = 0) ∧
  (∃ m : ℕ+, 2^n + 1 = m^2) = (n = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_conditions_l1207_120706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_outlay_is_10000_profit_calculation_correct_l1207_120730

/-- Represents the initial outlay for manufacturing horseshoes -/
def initial_outlay : ℝ := 10000

/-- Represents the cost per set of horseshoes -/
def cost_per_set : ℝ := 20

/-- Represents the selling price per set of horseshoes -/
def price_per_set : ℝ := 50

/-- Represents the number of sets produced and sold -/
def num_sets : ℕ := 500

/-- Represents the total profit -/
def total_profit : ℝ := 5000

/-- Calculates the manufacturing cost for a given number of sets -/
def manufacturing_cost (outlay : ℝ) (sets : ℕ) : ℝ :=
  outlay + cost_per_set * (sets : ℝ)

/-- Calculates the revenue for a given number of sets -/
def revenue (sets : ℕ) : ℝ :=
  price_per_set * (sets : ℝ)

/-- Theorem stating that the initial outlay is $10,000 -/
theorem initial_outlay_is_10000 :
  initial_outlay = 10000 :=
by
  -- The proof is trivial since we defined initial_outlay as 10000
  rfl

/-- Theorem verifying the profit calculation -/
theorem profit_calculation_correct :
  total_profit = revenue num_sets - manufacturing_cost initial_outlay num_sets :=
by
  -- Expand definitions and perform calculations
  simp [total_profit, revenue, manufacturing_cost, initial_outlay, cost_per_set, price_per_set, num_sets]
  -- The rest of the proof would involve arithmetic, which we'll skip for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_outlay_is_10000_profit_calculation_correct_l1207_120730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equality_l1207_120796

-- Define the function f as noncomputable
noncomputable def f (y : ℝ) : ℝ := Real.rpow (30 * y + Real.rpow (30 * y + Real.rpow (30 * y + 14) (1/3)) (1/3)) (1/3)

-- State the theorem
theorem solution_equality : f 91 = 14 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equality_l1207_120796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_start_c_l1207_120764

-- Define the speeds of A, B, and C
variable (Va Vb Vc : ℝ)

-- Define the conditions
variable (h1 : Va * 930 = Vb * 1000)
variable (h2 : Va * 800 = Vc * 1000)

-- Define the start distance B can give C
noncomputable def start_distance (Va Vb Vc : ℝ) : ℝ := 1000 - (1000 * Vc / Vb)

-- Theorem statement
theorem b_start_c (h1 : Va * 930 = Vb * 1000) (h2 : Va * 800 = Vc * 1000) :
  ∃ ε > 0, |start_distance Va Vb Vc - 139.78| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_start_c_l1207_120764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_internally_l1207_120717

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 12 = 0
def circle_O₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 6*y + 16 = 0

-- Define the centers and radii of the circles
def center_O₁ : ℝ × ℝ := (2, 3)
def center_O₂ : ℝ × ℝ := (4, 3)
def radius_O₁ : ℝ := 1
def radius_O₂ : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((center_O₂.1 - center_O₁.1)^2 + (center_O₂.2 - center_O₁.2)^2)

-- Theorem: The circles are tangent internally
theorem circles_tangent_internally :
  distance_between_centers = radius_O₂ - radius_O₁ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_internally_l1207_120717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1207_120778

theorem power_equality (x : ℝ) (h : (2 : ℝ)^(3*x) = 5) : (8 : ℝ)^(x+1) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1207_120778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_four_frequency_l1207_120761

theorem group_four_frequency (total_sample : ℕ) (num_groups : ℕ) 
  (group_one : ℕ) (group_two : ℕ) (group_three : ℕ) (group_five : ℕ) :
  total_sample = 50 →
  num_groups = 5 →
  group_one = 2 →
  group_two = 8 →
  group_three = 15 →
  group_five = 5 →
  (total_sample - (group_one + group_two + group_three + group_five)) / total_sample = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_four_frequency_l1207_120761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1207_120726

theorem polynomial_divisibility 
  (n : ℕ+) 
  (m : ℕ) 
  (f : Polynomial ℤ) 
  (h_m : m > 1)
  (h_coeff : ∀ (i : ℕ) (p : ℕ), 2 ≤ i → i ≤ m → Nat.Prime p → p ∣ n → p ∣ (f.coeff i).natAbs)
  (h_gcd : Nat.gcd (f.coeff 1).natAbs n = 1) :
  ∀ (k : ℕ+), ∃ (c : ℕ+), (n^(k : ℕ) : ℤ) ∣ f.eval c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1207_120726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_505_count_l1207_120756

theorem divisible_by_505_count : 
  (Finset.filter (fun k : ℕ => k ≤ 353500 ∧ (505 ∣ k^2 + k)) (Finset.range 353501)).card = 2800 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_505_count_l1207_120756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1207_120770

noncomputable def f (x : ℝ) := 1 + Real.sin x * Real.cos x

theorem f_properties :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi ∧
  (∀ x, f x ≥ 1/2) ∧ (∃ x, f x = 1/2) ∧
  (∀ x, 0 < x ∧ x < Real.pi / 2 → Real.tan x = 3/4 → f (Real.pi / 4 - x / 2) = 7/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1207_120770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_covering_l1207_120791

/-- A function that checks if a rectangle with dimensions m × n can be repeatably covered with the same overlap in each cell. -/
def canBeCovered (m n : ℕ) : Prop :=
  (m, n) ∉ ({(1, n) | n : ℕ} ∪ {(m, 1) | m : ℕ} ∪ {(2*m+1, 3) | m : ℕ} ∪ 
   {(3, 2*n+1) | n : ℕ} ∪ {(5, 5), (5, 7), (7, 5)})

/-- Theorem stating the necessary and sufficient condition for an m × n rectangle to be repeatably covered with the same overlap in each cell. -/
theorem rectangle_covering (m n : ℕ) :
  (∃ (k : ℕ), ∀ (i j : ℕ), i < m ∧ j < n → (∃ (c : ℕ), c = k)) ↔ canBeCovered m n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_covering_l1207_120791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_smallest_integer_l1207_120740

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) := a * Real.exp (x - 2)
noncomputable def g (x : ℝ) := (x + 1/x + 2) * Real.log x

theorem inequality_and_smallest_integer (a : ℕ) :
  (∀ x > 1, Real.log x < x/2 - 1/(2*x)) ∧
  (a = 4 ∧ ∀ x > 0, f a x > g x ∧ ∀ b < 4, ∃ y > 0, f b y ≤ g y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_smallest_integer_l1207_120740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_l1207_120780

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x / (x - 1)

-- State the theorem
theorem f_decreasing : 
  ∀ x₁ x₂ : ℝ, x₁ > 1 → x₂ > 1 → x₁ > x₂ → f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_l1207_120780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_farm_harvest_interval_l1207_120786

/-- Represents a coconut farm -/
structure CoconutFarm where
  size : ℕ  -- farm size in square meters
  treesPerSqm : ℕ  -- number of trees per square meter
  coconutsPerTree : ℕ  -- number of coconuts per tree
  pricePerCoconut : ℚ  -- price per coconut in dollars
  earningsAfter6Months : ℚ  -- earnings after 6 months in dollars

/-- Calculates the months between harvests for a given coconut farm -/
noncomputable def monthsBetweenHarvests (farm : CoconutFarm) : ℚ :=
  6 / ((farm.earningsAfter6Months / farm.pricePerCoconut) / (farm.size * farm.treesPerSqm * farm.coconutsPerTree))

/-- Theorem stating that for the given farm conditions, the months between harvests is 3 -/
theorem rohan_farm_harvest_interval :
  let farm := CoconutFarm.mk 20 2 6 (1/2) 240
  monthsBetweenHarvests farm = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_farm_harvest_interval_l1207_120786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1207_120752

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the system of equations
def system_eq (x y : ℝ) : Prop :=
  (2 : ℝ)^(lg x) + (3 : ℝ)^(lg y) = 5 ∧ (2 : ℝ)^(lg x) * (3 : ℝ)^(lg y) = 4

-- State the theorem
theorem system_solutions :
  ∃! (s : Set (ℝ × ℝ)), s = {(100, 1), (1, (10 : ℝ)^(Real.log 4 / Real.log 3))} ∧
  ∀ (x y : ℝ), (x, y) ∈ s ↔ system_eq x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l1207_120752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_square_area_l1207_120759

theorem midpoint_square_area (s : ℝ) (h : s^2 = 100) : 
  (s * Real.sqrt 2 / 2)^2 = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_square_area_l1207_120759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_ratio_triangle_l1207_120772

theorem smallest_angle_in_ratio_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  (a : ℝ) / 3 = (b : ℝ) / 4 ∧ (b : ℝ) / 4 = (c : ℝ) / 5 →
  min a (min b c) = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_ratio_triangle_l1207_120772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_as_difference_of_nth_powers_l1207_120787

/-- Given positive integers a, b, c, n such that a, b, c, a+b+c are pairwise coprime
    and (a+b)(b+c)(c+a)(a+b+c)(ab+bc+ca) is a perfect n-th power,
    prove that there exist integers k and m such that abc = k^n - m^n. -/
theorem abc_as_difference_of_nth_powers
  (a b c n : ℕ+)
  (h_coprime : Nat.Coprime a b ∧ Nat.Coprime a c ∧ Nat.Coprime b c ∧
               Nat.Coprime a (a + b + c) ∧ Nat.Coprime b (a + b + c) ∧ Nat.Coprime c (a + b + c))
  (h_perfect_power : ∃ x : ℕ+, (a + b) * (b + c) * (c + a) * (a + b + c) * (a * b + b * c + c * a) = x ^ n.val) :
  ∃ k m : ℤ, (a * b * c : ℤ) = k ^ n.val - m ^ n.val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_as_difference_of_nth_powers_l1207_120787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_division_ratio_l1207_120754

/-- Represents a right circular cone -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a frustum of a cone -/
structure Frustum where
  topRadius : ℝ
  bottomRadius : ℝ
  height : ℝ

noncomputable def Cone.volume (c : Cone) : ℝ := (1/3) * Real.pi * c.baseRadius^2 * c.height

noncomputable def Cone.surfaceArea (c : Cone) : ℝ := 
  Real.pi * c.baseRadius^2 + Real.pi * c.baseRadius * Real.sqrt (c.baseRadius^2 + c.height^2)

noncomputable def Frustum.volume (f : Frustum) : ℝ := 
  (1/3) * Real.pi * f.height * (f.bottomRadius^2 + f.topRadius^2 + f.bottomRadius * f.topRadius)

noncomputable def Frustum.surfaceArea (f : Frustum) : ℝ := 
  Real.pi * (f.bottomRadius + f.topRadius) * Real.sqrt ((f.bottomRadius - f.topRadius)^2 + f.height^2) +
  Real.pi * (f.bottomRadius^2 + f.topRadius^2)

theorem cone_division_ratio (c : Cone) (h : c.height = 6 ∧ c.baseRadius = 5) :
  let smallCone : Cone := { height := 2, baseRadius := 5/3 }
  let remainingFrustum : Frustum := { topRadius := 5/3, bottomRadius := 5, height := 4 }
  (smallCone.surfaceArea / remainingFrustum.surfaceArea) = 
  (smallCone.volume / remainingFrustum.volume) ∧
  (smallCone.volume / remainingFrustum.volume) = 1/24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_division_ratio_l1207_120754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1207_120709

-- Define the triangle ABC
structure Triangle where
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C
  A : ℝ  -- angle A in radians
  B : ℝ  -- angle B in radians
  C : ℝ  -- angle C in radians

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the arithmetic sequence property
def isArithmeticSequence (t : Triangle) : Prop :=
  t.b = t.a + 2 ∧ t.c = t.a + 4

-- Define the angle C as 120°
def hasAngleC120 (t : Triangle) : Prop :=
  t.C = 2 * Real.pi / 3

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : isArithmeticSequence t) 
  (h3 : hasAngleC120 t) : 
  t.a = 3 ∧ 
  let CD := t.b * Real.sin t.C / t.c
  CD = 15 * Real.sqrt 3 / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1207_120709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_concyclic_points_l1207_120746

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if two circles are tangent -/
def are_tangent (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a point lies on a circle -/
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

/-- Predicate to check if a list of points is concyclic -/
def are_concyclic (points : List (ℝ × ℝ)) : Prop := sorry

theorem tangent_circles_concyclic_points
  (R₁ R₂ : Circle)
  (S : List Circle)
  (A : List (ℝ × ℝ))
  (n : ℕ)
  (h1 : S.length = n)
  (h2 : A.length = n - 1)
  (h3 : ∀ i, i < n → are_tangent (S.get ⟨i, by sorry⟩) R₁ ∧ are_tangent (S.get ⟨i, by sorry⟩) R₂)
  (h4 : ∀ i, i < n - 1 → are_tangent (S.get ⟨i, by sorry⟩) (S.get ⟨i+1, by sorry⟩) ∧ 
             point_on_circle (A.get ⟨i, by sorry⟩) (S.get ⟨i, by sorry⟩) ∧ 
             point_on_circle (A.get ⟨i, by sorry⟩) (S.get ⟨i+1, by sorry⟩)) :
  are_concyclic A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_concyclic_points_l1207_120746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_sqrt_three_l1207_120783

/-- Helper function to calculate the area of a triangle given its side lengths -/
noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Given an acute triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that its area is √3 under the given conditions. -/
theorem triangle_area_sqrt_three (a b c : ℝ) (A B C : ℝ) : 
  b = 2 →
  B = π / 3 →
  c * Real.sin A = Real.sqrt 3 * a * Real.cos C →
  area_triangle a b c = Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_sqrt_three_l1207_120783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_monotonicity_l1207_120735

def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem f_extrema_and_monotonicity :
  (∀ x, x ∈ Set.Icc (-5 : ℝ) 5 → f (-1) x ≥ 1 ∧ f (-1) x ≤ 37) ∧
  (∃ x₁ x₂, x₁ ∈ Set.Icc (-5 : ℝ) 5 ∧ x₂ ∈ Set.Icc (-5 : ℝ) 5 ∧ f (-1) x₁ = 1 ∧ f (-1) x₂ = 37) ∧
  (∀ a : ℝ, (∀ x y, x ∈ Set.Icc (-5 : ℝ) 5 → y ∈ Set.Icc (-5 : ℝ) 5 → x < y → (f a x < f a y ∨ f a x > f a y)) ↔ 
    (a ≤ -5 ∨ a ≥ 5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_monotonicity_l1207_120735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_l1207_120768

/-- The function f(x) = 1/x + a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/x + a * Real.log x

/-- The theorem stating that if the minimum value of f(x) on [1/2, 1] is 0, then a = 2/ln(2) -/
theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, f a x ≥ 0) ∧ 
  (∃ x ∈ Set.Icc (1/2 : ℝ) 1, f a x = 0) →
  a = 2 / Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_l1207_120768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1207_120767

-- Define the functions f and g
def f (m x : ℝ) : ℝ := m * (x - m) * (x + m + 3)
noncomputable def g (x : ℝ) : ℝ := (2 : ℝ)^x - 4

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ,
  (∀ x : ℝ, f m x < 0 ∨ g x < 0) ↔
  -5 < m ∧ m < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1207_120767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l1207_120771

noncomputable def a : ℝ × ℝ := (-5, 1)
noncomputable def b : ℝ × ℝ := (2, 3)
noncomputable def p : ℝ × ℝ := (-34/53, 119/53)

theorem projection_equality (v : ℝ × ℝ) : v ≠ (0, 0) → 
  ((p.1 * v.1 + p.2 * v.2) / (v.1^2 + v.2^2)) • v = p ∧
  ((a.1 * v.1 + a.2 * v.2) / (v.1^2 + v.2^2)) • v = p ∧
  ((b.1 * v.1 + b.2 * v.2) / (v.1^2 + v.2^2)) • v = p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l1207_120771
