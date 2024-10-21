import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_athlete_last_l692_69253

/-- Represents an athlete in the race -/
structure Athlete where
  initial_speed : ℝ
  final_speed : ℝ

/-- The race setup -/
noncomputable def race_setup (v₁ v₂ v₃ : ℝ) : Fin 3 → Athlete :=
  fun i => match i with
  | 0 => { initial_speed := v₁, final_speed := v₂ }
  | 1 => { initial_speed := v₂, final_speed := v₃ }
  | 2 => { initial_speed := v₃, final_speed := v₁ }

/-- Calculate the total time for an athlete -/
noncomputable def total_time (a : Athlete) : ℝ := 1 / a.initial_speed + 2 / a.final_speed

/-- Theorem: The second athlete finishes last -/
theorem second_athlete_last (v₁ v₂ v₃ : ℝ) 
  (h1 : v₁ > v₂) (h2 : v₂ > v₃) (h3 : v₃ > 0) : 
  let athletes := race_setup v₁ v₂ v₃
  total_time (athletes 1) > max (total_time (athletes 0)) (total_time (athletes 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_athlete_last_l692_69253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equals_one_exp_diff_equals_one_l692_69262

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem 1
theorem log_sum_equals_one : lg 2 + lg 5 = 1 := by sorry

-- Theorem 2
theorem exp_diff_equals_one : (2 : ℝ)^(Real.log 3 / Real.log 2) - (8 : ℝ)^(1/3) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equals_one_exp_diff_equals_one_l692_69262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l692_69215

/-- A function f(x) = sin(ωx + φ) with specific properties -/
noncomputable def f (ω : ℝ) (φ : ℝ) : ℝ → ℝ := fun x ↦ Real.sin (ω * x + φ)

/-- The theorem stating that under given conditions, ω = 2 -/
theorem omega_value (ω : ℝ) (φ : ℝ) : 
  ω > 0 ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < π / 3 → f ω φ x < f ω φ y) ∧
  f ω φ (π / 6) + f ω φ (π / 3) = 0 ∧
  f ω φ 0 = -1 →
  ω = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l692_69215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_points_on_ellipse_sum_of_inverse_squares_l692_69282

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the distance from origin to a point
noncomputable def distance_from_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

-- Theorem statement
theorem perpendicular_points_on_ellipse_sum_of_inverse_squares (A B : ℝ × ℝ) :
  ellipse A.1 A.2 →
  ellipse B.1 B.2 →
  A.1 * B.1 + A.2 * B.2 = 0 →  -- Perpendicularity condition
  1 / (distance_from_origin A.1 A.2)^2 + 1 / (distance_from_origin B.1 B.2)^2 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_points_on_ellipse_sum_of_inverse_squares_l692_69282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_solution_l692_69212

/-- Two lines in a Cartesian coordinate system -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The point of intersection of two lines -/
noncomputable def intersection (l1 l2 : Line) : ℝ × ℝ :=
  let x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.intercept
  (x, y)

/-- The system of equations represented by two lines -/
def system_solution (l1 l2 : Line) : Prop :=
  let (x, y) := intersection l1 l2
  x = 1 ∧ y = 2

theorem intersection_solution (l1 l2 : Line) :
  l1 = Line.mk 1 1 →
  l2 = Line.mk (-1) 3 →
  intersection l1 l2 = (1, 2) →
  system_solution l1 l2 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_solution_l692_69212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_iff_k_range_l692_69273

/-- The circle equation -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + k - 2 = 0

/-- The point (1,2) -/
def point : ℝ × ℝ := (1, 2)

/-- The center of the circle -/
def center : ℝ → ℝ × ℝ := λ k ↦ (-1, 2)

/-- The radius of the circle -/
noncomputable def radius (k : ℝ) : ℝ := Real.sqrt (7 - k)

/-- The distance between the point and the center -/
def distance : ℝ := 2

/-- Theorem: The point (1,2) can be the intersection of two tangents
    of the circle if and only if k is in the range (3,7) -/
theorem tangent_intersection_iff_k_range :
  ∀ k : ℝ, (distance > radius k ∧ 7 - k > 0) ↔ (3 < k ∧ k < 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_iff_k_range_l692_69273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_properties_max_k_value_l692_69277

noncomputable section

variable (x x1 x2 : ℝ)

def f (a : ℝ) (x : ℝ) := Real.exp x + 2 * Real.exp (-x) + a
def g (x : ℝ) := x^2 - x + 1

def common_tangent (m : ℝ) (x : ℝ) := m * x + 1

theorem tangent_line_properties :
  ∃ (m a : ℝ),
    (∀ x, common_tangent m x = g x → (deriv (f a)) x = m) ∧
    m = -1 ∧
    a = -2 := by sorry

theorem max_k_value :
  ∃ k : ℝ,
    (Real.exp x1 + Real.exp x2 = 3 →
     f (-2) x1 * f (-2) x2 ≥ 3 * (x1 + x2 + k)) ∧
    k = 25 / 108 - 2 * Real.log (3 / 2) ∧
    ∀ k' : ℝ, (∀ x1 x2 : ℝ, Real.exp x1 + Real.exp x2 = 3 →
               f (-2) x1 * f (-2) x2 ≥ 3 * (x1 + x2 + k')) →
              k' ≤ k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_properties_max_k_value_l692_69277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_triangle_l692_69232

/-- A triangle with integer coordinates -/
structure IntTriangle where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ

/-- Calculate the area of a triangle given its vertices -/
def triangleArea (t : IntTriangle) : ℚ :=
  let (x1, y1) := t.a
  let (x2, y2) := t.b
  let (x3, y3) := t.c
  (1 / 2) * ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)).natAbs : ℚ)

/-- Check if a line is parallel to an axis -/
def isParallelToAxis (p1 p2 : ℤ × ℤ) : Bool :=
  (p1.1 = p2.1) ∨ (p1.2 = p2.2)

/-- Calculate the perimeter of a triangle -/
noncomputable def trianglePerimeter (t : IntTriangle) : ℝ :=
  let d (p1 p2 : ℤ × ℤ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 : ℝ)
  d t.a t.b + d t.b t.c + d t.c t.a

/-- The main theorem -/
theorem smallest_perimeter_triangle :
  ∃ (t : IntTriangle),
    triangleArea t = 1/2 ∧
    ¬isParallelToAxis t.a t.b ∧
    ¬isParallelToAxis t.b t.c ∧
    ¬isParallelToAxis t.c t.a ∧
    ∀ (t' : IntTriangle),
      triangleArea t' = 1/2 →
      ¬isParallelToAxis t'.a t'.b →
      ¬isParallelToAxis t'.b t'.c →
      ¬isParallelToAxis t'.c t'.a →
      trianglePerimeter t ≤ trianglePerimeter t' ∧
      trianglePerimeter t = Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 13 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_triangle_l692_69232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_boiling_time_correct_specific_problem_solution_l692_69263

/-- Represents a heater with a specific boiling time -/
structure Heater where
  boiling_time : ℝ
  boiling_time_pos : boiling_time > 0

/-- Calculates the parallel boiling time for two heaters -/
noncomputable def parallel_boiling_time (h1 h2 : Heater) : ℝ :=
  (h1.boiling_time * h2.boiling_time) / (h1.boiling_time + h2.boiling_time)

/-- Theorem: The parallel boiling time is correct for any two heaters -/
theorem parallel_boiling_time_correct (h1 h2 : Heater) :
  parallel_boiling_time h1 h2 < min h1.boiling_time h2.boiling_time :=
by sorry

/-- Theorem: The parallel boiling time for the specific problem is 72 seconds -/
theorem specific_problem_solution :
  let h1 : Heater := ⟨120, by norm_num⟩
  let h2 : Heater := ⟨180, by norm_num⟩
  parallel_boiling_time h1 h2 = 72 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_boiling_time_correct_specific_problem_solution_l692_69263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_for_positive_f_log_l692_69211

/-- Given a constant a between 0 and 1, and a function f satisfying certain properties,
    this theorem states the range of x for which f(log_a x) > 0. -/
theorem range_of_x_for_positive_f_log (a : ℝ) (f : ℝ → ℝ) 
    (ha : 0 < a ∧ a < 1)
    (hf_odd : ∀ x, f (-x) = -f x)
    (hf_decreasing : ∀ x y, 0 < x ∧ x < y → f y < f x)
    (hf_half : f (1/2) = 0)
    : {x : ℝ | f (Real.log x / Real.log a) > 0} = {x : ℝ | x > 1/Real.sqrt a ∨ Real.sqrt a < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_for_positive_f_log_l692_69211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_length_last_four_of_5_2011_l692_69245

def last_four_digits (n : ℕ) : ℕ := n % 10000

def last_four_pattern : List ℕ := [3125, 5625, 8125, 0625]

theorem pattern_length : List.length last_four_pattern = 4 := by rfl

axiom pattern_correct : ∀ k : ℕ, last_four_digits (5^(k + 5)) = last_four_pattern[k % 4]'(by {
  rw [pattern_length]
  exact Nat.mod_lt k (by norm_num)
})

theorem last_four_of_5_2011 : last_four_digits (5^2011) = 8125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_length_last_four_of_5_2011_l692_69245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_theorem_min_weighings_optimal_l692_69234

/-- Represents the minimum number of weighings needed to identify a counterfeit coin -/
def min_weighings (n : ℕ) : ℕ :=
  if n = 3 then 2
  else if n = 4 then 2
  else if n = 9 then 3
  else 0  -- undefined for other cases

/-- Represents the weight of a coin -/
noncomputable def weight (i : ℕ) : ℝ := sorry

/-- The properties of the coin identification problem -/
axiom coin_properties (n : ℕ) :
  n > 0 →  -- there is at least one coin
  ∃ (i : ℕ), i > 0 ∧ i ≤ n ∧  -- there exists a counterfeit coin
  ∀ (j : ℕ), j ≠ i → j ≤ n →  -- all other coins are genuine
  (∀ (k : ℕ), k ≠ i → k ≤ n → k ≠ j → 
    (weight j = weight k))  -- all genuine coins have the same weight
  ∧ (weight i ≠ weight j)  -- the counterfeit coin has a different weight

/-- Theorem stating the minimum number of weighings for 3, 4, and 9 coins -/
theorem min_weighings_theorem :
  (min_weighings 3 = 2) ∧
  (min_weighings 4 = 2) ∧
  (min_weighings 9 = 3) := by
  sorry

/-- Predicate to check if the counterfeit coin can be identified with m weighings -/
def can_identify_counterfeit (arrangement : List ℕ) (m : ℕ) : Prop :=
  sorry  -- Definition would involve complex logic about weighing strategies

/-- Proof that the minimum number of weighings is optimal -/
theorem min_weighings_optimal (n : ℕ) :
  (n = 3 ∨ n = 4 ∨ n = 9) →
  ∀ (m : ℕ), m < min_weighings n →
    ∃ (arrangement : List ℕ), 
      arrangement.length = n ∧
      ¬ (can_identify_counterfeit arrangement m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_theorem_min_weighings_optimal_l692_69234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_ratio_range_l692_69280

theorem geometric_triangle_ratio_range :
  ∀ a q : ℝ, a > 0 → q > 0 →
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y > z ∧ x + z > y ∧ y + z > x ∧
    y = a ∧ x = a / q ∧ z = a * q) →
  (((Real.sqrt 5 - 1) / 2 < q) ∧ (q < (Real.sqrt 5 + 1) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_ratio_range_l692_69280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_length_l692_69240

noncomputable section

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Define the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 1

-- Define the asymptotic line
def asymptotic_line (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x ∨ y = -(b / a) * x

-- Theorem statement
theorem hyperbola_intersection_length (a b : ℝ) (A B : ℝ × ℝ) :
  hyperbola a b A.1 A.2 →
  hyperbola a b B.1 B.2 →
  eccentricity a b = Real.sqrt 5 →
  circle_equation A.1 A.2 →
  circle_equation B.1 B.2 →
  asymptotic_line a b A.1 A.2 →
  asymptotic_line a b B.1 B.2 →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (4 * Real.sqrt 5) / 5 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_length_l692_69240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_divisibility_l692_69204

/-- Lucas sequence -/
def L : ℕ → ℤ
  | 0 => 2
  | 1 => 1
  | (n + 2) => L (n + 1) + L n

/-- α and β definitions -/
noncomputable def α : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def β : ℝ := (1 - Real.sqrt 5) / 2

/-- Lucas sequence identity -/
axiom L_identity (n : ℕ) : (L n : ℝ) = α ^ n + β ^ n

/-- Main theorem -/
theorem lucas_divisibility (j : ℕ) : 
  (1 + L (2 ^ j)) % (2 ^ (j + 1)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_divisibility_l692_69204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_circle_area_sum_l692_69252

/-- Represents a square grid with circles placed on it -/
structure GridWithCircles where
  gridSize : Nat
  squareSize : ℝ
  numCircles : Nat

/-- Calculates the total area of the grid -/
noncomputable def totalGridArea (g : GridWithCircles) : ℝ :=
  (g.gridSize * g.squareSize) ^ 2

/-- Calculates the total area of the circles -/
noncomputable def totalCircleArea (g : GridWithCircles) : ℝ :=
  g.numCircles * Real.pi * (g.squareSize / 2) ^ 2

/-- The main theorem to be proved -/
theorem grid_circle_area_sum (g : GridWithCircles) 
  (h1 : g.gridSize = 4)
  (h2 : g.squareSize = 3)
  (h3 : g.numCircles = 4) : 
  totalGridArea g + (totalCircleArea g / Real.pi) = 153 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_circle_area_sum_l692_69252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_x_intercepts_count_l692_69281

theorem distinct_x_intercepts_count : ∃ (roots : Set ℝ), 
  (∀ x ∈ roots, (x - 2) * (x^2 + 6*x + 9) = 0) ∧ 
  (∀ x : ℝ, (x - 2) * (x^2 + 6*x + 9) = 0 → x ∈ roots) ∧ 
  (∃ (l : List ℝ), roots = l.toFinset ∧ l.length = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_x_intercepts_count_l692_69281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_percentage_theorem_l692_69274

/-- A pentagon formed by an equilateral triangle on top of a square -/
structure TriangleSquarePentagon where
  s : ℝ  -- side length of the equilateral triangle
  h : ℝ  -- height of the square
  triangle_side_eq_square_height : s = h

/-- The area of the equilateral triangle in the pentagon -/
noncomputable def triangle_area (p : TriangleSquarePentagon) : ℝ :=
  (p.s^2 * Real.sqrt 3) / 4

/-- The area of the square in the pentagon -/
def square_area (p : TriangleSquarePentagon) : ℝ :=
  p.h^2

/-- The total area of the pentagon -/
noncomputable def pentagon_area (p : TriangleSquarePentagon) : ℝ :=
  triangle_area p + square_area p

/-- The percentage of the pentagon's area that is the triangle -/
noncomputable def triangle_percentage (p : TriangleSquarePentagon) : ℝ :=
  (triangle_area p / pentagon_area p) * 100

theorem triangle_percentage_theorem (p : TriangleSquarePentagon) :
  triangle_percentage p = (Real.sqrt 3 / (3 + Real.sqrt 3)) * 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_percentage_theorem_l692_69274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l692_69244

/-- A line in a 2D coordinate system -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Checks if a point (x, y) is on the line -/
noncomputable def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.y_intercept

/-- The x-intercept of a line -/
noncomputable def Line.x_intercept (l : Line) : ℝ :=
  -l.y_intercept / l.slope

/-- Theorem stating that if a line passes through (-8, -6) and has x-intercept 4,
    then it also passes through (10, 3) -/
theorem line_through_points (l : Line) :
  l.contains (-8) (-6) ∧ l.x_intercept = 4 →
  l.contains 10 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l692_69244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operations_count_l692_69203

/-- Represents a polynomial of degree 5 -/
structure Polynomial5 where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Counts the number of operations (additions and multiplications) in Horner's method -/
def hornerOperations (p : Polynomial5) : ℕ :=
  5 + 5  -- 5 multiplications and 5 additions

/-- The theorem stating that Horner's method for the given polynomial requires 10 operations -/
theorem horner_operations_count :
  let p : Polynomial5 := { a := 5, b := 4, c := 3, d := 2, e := 1, f := 0.3 }
  hornerOperations p = 10 := by
  -- Unfold the definition of hornerOperations
  unfold hornerOperations
  -- The result is immediate from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operations_count_l692_69203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l692_69292

theorem triangle_abc_properties (A B C : Real) (BC AC : Real) :
  -- Given conditions
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  2 * (Real.sin (B + C))^2 = Real.sqrt 3 * Real.sin (2 * A) ∧
  BC = 7 ∧
  AC = 5 →
  -- Conclusions to prove
  A = π / 3 ∧
  let AB := Real.sqrt (BC^2 + AC^2 - 2 * BC * AC * Real.cos A);
  let S := 1 / 2 * AB * AC * Real.sin A;
  S = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l692_69292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l692_69294

/-- Curve C₁ in polar coordinates -/
def C₁ (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 4

/-- Point on line segment OM satisfying condition -/
noncomputable def P_on_OM (O M P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ P = (t * M.1, t * M.2) ∧ 
  (((M.1 - O.1)^2 + (M.2 - O.2)^2) * ((P.1 - O.1)^2 + (P.2 - O.2)^2))^(1/2 : ℝ) = 16

/-- Curve C₂ in Cartesian coordinates -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4 ∧ x ≠ 0

/-- Point A in Cartesian coordinates -/
noncomputable def A : ℝ × ℝ := (1, Real.sqrt 3)

/-- Area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  1/2 * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

/-- The main theorem -/
theorem max_triangle_area :
  ∃ B : ℝ × ℝ, C₂ B.1 B.2 ∧
  ∀ P : ℝ × ℝ, C₂ P.1 P.2 → triangle_area (0, 0) A P ≤ triangle_area (0, 0) A B ∧
  triangle_area (0, 0) A B = 1 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l692_69294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_non_negative_f_range_l692_69271

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := (x - 1) * Real.log x - m * (x + 1)

-- Part 1: Tangent line equation
theorem tangent_line_at_one :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x y : ℝ, (f x 1 = y) → (a * x + b * y + c = 0)) ∧
  (a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

-- Part 2: Range of m for non-negative f(x)
theorem non_negative_f_range :
  ∀ m : ℝ, (∀ x : ℝ, x > 0 → f x m ≥ 0) ↔ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_non_negative_f_range_l692_69271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l692_69284

theorem problem_solution (x y z : ℤ) 
  (h1 : x > 0 ∧ y > 0 ∧ z > 0)
  (h2 : x ≥ y ∧ y ≥ z)
  (h3 : x^2 - y^2 - z^2 + x*y = 3007)
  (h4 : x^2 + 3*y^2 + 3*z^2 - 2*x*y - 3*x*z - 3*y*z = -2013) :
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l692_69284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tripleSum_eq_one_over_fiftyfour_l692_69226

/-- The triple sum defined in the problem -/
noncomputable def tripleSum : ℝ :=
  ∑' (a : ℕ+), ∑' (b : ℕ+), ∑' (c : ℕ+),
    (a.val * b.val * (3 * a.val + c.val : ℝ)) / 
    (4^(a.val + b.val + c.val) * (a.val + b.val) * (b.val + c.val) * (c.val + a.val))

/-- The theorem stating that the triple sum equals 1/54 -/
theorem tripleSum_eq_one_over_fiftyfour : tripleSum = 1/54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tripleSum_eq_one_over_fiftyfour_l692_69226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_partition_l692_69296

/-- A graph with n vertices -/
structure Graph (n : ℕ) where
  edges : Fin n → Fin n → Prop

/-- A clique in a graph is a set of vertices where every pair is connected -/
def is_clique {n : ℕ} (G : Graph n) (S : Set (Fin n)) : Prop :=
  ∀ i j, i ∈ S → j ∈ S → i ≠ j → G.edges i j

/-- An independent set in a graph is a set of vertices where no pair is connected -/
def is_independent_set {n : ℕ} (G : Graph n) (S : Set (Fin n)) : Prop :=
  ∀ i j, i ∈ S → j ∈ S → i ≠ j → ¬G.edges i j

/-- The property that every subgraph of 4 vertices contains either a triangle or an independent set of size 3 -/
def has_triangle_or_independent_triple {n : ℕ} (G : Graph n) : Prop :=
  ∀ (a b c d : Fin n), 
    (is_clique G {a, b, c}) ∨ 
    (is_clique G {a, b, d}) ∨ 
    (is_clique G {a, c, d}) ∨ 
    (is_clique G {b, c, d}) ∨
    (is_independent_set G {a, b, c}) ∨ 
    (is_independent_set G {a, b, d}) ∨ 
    (is_independent_set G {a, c, d}) ∨ 
    (is_independent_set G {b, c, d})

/-- The main theorem -/
theorem ramsey_partition {n : ℕ} (G : Graph n) 
  (h : has_triangle_or_independent_triple G) : 
  ∃ (A B : Set (Fin n)), (A ∪ B : Set (Fin n)) = Set.univ ∧ A ∩ B = ∅ ∧ 
    is_clique G A ∧ is_independent_set G B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_partition_l692_69296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sequence_angle_l692_69297

theorem triangle_arithmetic_sequence_angle (A B C : ℝ) (a b c : ℝ) : 
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Arithmetic sequence condition
  ∃ (d : ℝ), b * Real.cos B = a * Real.cos C + d ∧ 
              c * Real.cos A = b * Real.cos B + d →
  -- Conclusion
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sequence_angle_l692_69297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_musical_assignment_count_l692_69227

/-- The number of ways to assign roles in a musical with specific constraints. -/
def musical_role_assignments (num_men : ℕ) (num_women : ℕ) 
  (male_roles : ℕ) (female_roles : ℕ) (either_roles : ℕ) : ℕ :=
  (Nat.factorial num_men / Nat.factorial (num_men - male_roles)) *
  (Nat.factorial num_women / Nat.factorial (num_women - female_roles)) *
  (Nat.factorial (num_men + num_women - male_roles - female_roles) / 
   Nat.factorial (num_men + num_women - male_roles - female_roles - either_roles))

/-- Theorem stating the number of ways to assign roles in the given musical scenario. -/
theorem musical_assignment_count : 
  musical_role_assignments 7 8 3 3 2 = 5080320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_musical_assignment_count_l692_69227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_monotone_function_with_uncountable_solutions_no_cont_diff_function_with_uncountable_solutions_l692_69224

-- Part (a)
theorem no_monotone_function_with_uncountable_solutions :
  ¬ ∃ (f : Set.Icc 0 1 → Set.Icc 0 1),
    Monotone f ∧
    ∀ y ∈ Set.Icc 0 1, ¬(Set.Countable {x ∈ Set.Icc 0 1 | f x = y}) := by
  sorry

-- Part (b)
theorem no_cont_diff_function_with_uncountable_solutions :
  ¬ ∃ (f : ℝ → ℝ),
    ContinuousOn f (Set.Icc 0 1) ∧
    DifferentiableOn ℝ f (Set.Icc 0 1) ∧
    ∀ y ∈ Set.Icc 0 1, ¬(Set.Countable {x ∈ Set.Icc 0 1 | f x = y}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_monotone_function_with_uncountable_solutions_no_cont_diff_function_with_uncountable_solutions_l692_69224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_quadrilateral_inscribed_circle_radius_l692_69289

/-- The radius of the largest inscribed circle in a quadrilateral with given side lengths -/
noncomputable def largest_inscribed_circle_radius (a b c d : ℝ) : ℝ :=
  let s := (a + b + c + d) / 2
  let area := Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d))
  area / s

/-- Theorem stating the radius of the largest inscribed circle in a specific quadrilateral -/
theorem specific_quadrilateral_inscribed_circle_radius :
  largest_inscribed_circle_radius 10 11 13 12 = 2 * Real.sqrt 2145 / 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_quadrilateral_inscribed_circle_radius_l692_69289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_extension_constant_string_extension_value_l692_69254

/-- The extension of a string wrapped around a sphere and then extended 1 meter from its surface -/
noncomputable def string_extension (sphere_radius : ℝ) : ℝ :=
  2 * Real.pi * (sphere_radius + 1) - 2 * Real.pi * sphere_radius

theorem string_extension_constant (r₁ r₂ : ℝ) (hr₁ : r₁ > 0) (hr₂ : r₂ > 0) :
  string_extension r₁ = string_extension r₂ := by
  sorry

theorem string_extension_value (r : ℝ) (hr : r > 0) :
  string_extension r = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_extension_constant_string_extension_value_l692_69254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_of_roots_l692_69255

theorem max_value_of_sum_of_roots (x y z : ℝ) 
  (eq_constraint : 2*x + 3*y + z = 3)
  (x_constraint : x ≥ -2)
  (y_constraint : y ≥ -2/3)
  (z_constraint : z ≥ -4) :
  (Real.sqrt (6*x + 4) + Real.sqrt (9*y + 2) + Real.sqrt (3*z + 12) ≤ Real.sqrt 168) ∧ 
  (∃ x₀ y₀ z₀ : ℝ, 2*x₀ + 3*y₀ + z₀ = 3 ∧ 
   x₀ ≥ -2 ∧ y₀ ≥ -2/3 ∧ z₀ ≥ -4 ∧
   Real.sqrt (6*x₀ + 4) + Real.sqrt (9*y₀ + 2) + Real.sqrt (3*z₀ + 12) = Real.sqrt 168) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_of_roots_l692_69255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_sequence_formula_diagonal_sequence_7th_term_l692_69260

def diagonal_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => diagonal_sequence n + 2 * (n + 1)

theorem diagonal_sequence_formula (n : ℕ) :
  diagonal_sequence n = n^2 + n + 1 := by sorry

theorem diagonal_sequence_7th_term :
  diagonal_sequence 6 = 43 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_sequence_formula_diagonal_sequence_7th_term_l692_69260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_sale_price_l692_69229

/-- Calculates the sale price of an item given its original price and discount percentage. -/
noncomputable def salePrice (originalPrice : ℝ) (discountPercentage : ℝ) : ℝ :=
  originalPrice - (discountPercentage / 100 * originalPrice)

/-- Theorem stating that a $100 trouser with a 50% discount has a sale price of $50. -/
theorem trouser_sale_price :
  let originalPrice : ℝ := 100
  let discountPercentage : ℝ := 50
  salePrice originalPrice discountPercentage = 50 := by
  -- Unfold the definition of salePrice
  unfold salePrice
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_sale_price_l692_69229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_range_l692_69214

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 else x^2 - 2*a*x + 2*a

def has_two_symmetric_pairs (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
  f a x₁ = -(f a (-x₁)) ∧ f a x₂ = -(f a (-x₂)) ∧
  ∀ x : ℝ, x > 0 → x ≠ x₁ → x ≠ x₂ → f a x ≠ -(f a (-x))

theorem f_symmetric_range (a : ℝ) :
  has_two_symmetric_pairs a ↔ a > 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_range_l692_69214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l692_69298

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) / x^2

noncomputable def g (t : ℝ) (x : ℝ) : ℝ := t * f x + 1

theorem problem_solution :
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  (∀ k : ℝ, (∀ x ∈ Set.Icc 1 3, k ≤ x * f x + 1/x) → k ≤ 1) ∧
  (∀ m n t : ℝ, m > 0 → n > 0 → t ≥ 0 →
    (∀ x ∈ Set.Icc (1/m) (1/n), g t x ∈ Set.Icc (2 - 3*m) (2 - 3*n)) →
    0 < t ∧ t < 1) :=
by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l692_69298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l692_69295

-- Define the points A and B
noncomputable def A : ℝ × ℝ := (0, 14)
noncomputable def B : ℝ × ℝ := (0, 4)

-- Define the hyperbola
noncomputable def hyperbola (x : ℝ) : ℝ := 1 / x

-- Define the parallel lines
noncomputable def line_A (k : ℝ) (x : ℝ) : ℝ := k * x + A.snd
noncomputable def line_B (k : ℝ) (x : ℝ) : ℝ := k * x + B.snd

-- Define the intersection points
noncomputable def K (k : ℝ) : ℝ × ℝ := sorry
noncomputable def L (k : ℝ) : ℝ × ℝ := sorry
noncomputable def M (k : ℝ) : ℝ × ℝ := sorry
noncomputable def N (k : ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem intersection_ratio :
  ∀ k : ℝ, k ≠ 0 →
  (A.snd - (L k).snd) - (A.snd - (K k).snd) = 3.5 * ((B.snd - (N k).snd) - (B.snd - (M k).snd)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l692_69295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_fourth_l692_69276

theorem tan_alpha_minus_pi_fourth (α : ℝ) 
  (h1 : α > -π/2 ∧ α < 0) 
  (h2 : Real.cos α = Real.sqrt 5 / 5) : 
  Real.tan (α - π/4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_fourth_l692_69276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_sin_2theta_value_l692_69249

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sqrt 3 * Real.sin x * Real.cos x + 1

theorem f_monotone_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi + Real.pi / 3) (k * Real.pi + 5 * Real.pi / 6)) :=
sorry

theorem sin_2theta_value (θ : ℝ) (h1 : f θ = 5/6) (h2 : θ ∈ Set.Ioo (Real.pi / 3) (2 * Real.pi / 3)) :
  Real.sin (2 * θ) = (2 * Real.sqrt 3 - Real.sqrt 5) / 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_sin_2theta_value_l692_69249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l692_69265

noncomputable def f (x : ℝ) : ℝ := 1 / (2 * x - 18)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 9} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l692_69265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sum_l692_69210

noncomputable def arithmetic_seq (n : ℕ) : ℝ := 2 * n

noncomputable def S (n : ℕ) : ℝ := n * (n + 1)

noncomputable def b (n : ℕ) : ℝ := 1 / S n

noncomputable def T (n : ℕ) : ℝ := n / (n + 1)

theorem arithmetic_geometric_sum (n : ℕ) :
  (arithmetic_seq 1) * (arithmetic_seq 2) = (arithmetic_seq 1)^2 * (arithmetic_seq 4) →
  T n = n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sum_l692_69210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l692_69205

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - x - 6 < 0}

-- Define set B
def B : Set ℝ := {x | ∃ y ∈ A, |x| = y + 2}

-- Define the complement of a set
def complement (S : Set ℝ) : Set ℝ := U \ S

-- Theorem statement
theorem set_operations :
  (complement B = Set.Iic (-5) ∪ {0} ∪ Set.Ici 5) ∧
  (A ∩ B = Set.Ioo (-2) 0 ∪ Set.Ioo 0 3) ∧
  (A ∪ B = Set.Ioo (-5) 5) ∧
  (A ∪ (complement B) = Set.Iic 5 ∪ Set.Ioo (-2) 3 ∪ Set.Ici 5) ∧
  (A ∩ (complement B) = {0}) ∧
  (complement (A ∪ B) = (complement A) ∩ (complement B)) ∧
  ((complement A) ∩ (complement B) = Set.Iic (-5) ∪ Set.Ici 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l692_69205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_scaled_functions_sum_of_coordinates_l692_69219

-- Define the functions h and j
def h : ℝ → ℝ := sorry
def j : ℝ → ℝ := sorry

-- Define the intersection points of h and j
axiom intersection1 : h 3 = j 3 ∧ h 3 = 3
axiom intersection2 : h 6 = j 6 ∧ h 6 = 9
axiom intersection3 : h 9 = j 9 ∧ h 9 = 18
axiom intersection4 : h 12 = j 12 ∧ h 12 = 18

-- State the theorem
theorem intersection_of_scaled_functions :
  h (3 * 4) = 3 * j 4 ∧ h (3 * 4) = 18 := by
  sorry

-- Prove that the sum of coordinates is 22
theorem sum_of_coordinates : 
  ∃ (x y : ℝ), h (3 * x) = 3 * j x ∧ h (3 * x) = y ∧ x + y = 22 := by
  use 4, 18
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_scaled_functions_sum_of_coordinates_l692_69219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_width_of_wall_A_correct_l692_69267

/-- The width of Wall A in a building with three walls -/
noncomputable def width_of_wall_A (x y z w V : ℝ) : ℝ :=
  (V / (252 + y * x^2 + 21 * w * x * z))^(1/3)

theorem width_of_wall_A_correct (x y z w V : ℝ) :
  let wall_A_width := width_of_wall_A x y z w V
  let wall_A_height := 6 * wall_A_width
  let wall_A_length := 7 * wall_A_height
  let wall_B_width := wall_A_width
  let wall_B_height := x * wall_A_width
  let wall_B_length := y * wall_B_height
  let wall_C_width := z * wall_A_width
  let wall_C_height := (1/2) * wall_B_height
  let wall_C_length := w * wall_A_length
  let volume_A := wall_A_width * wall_A_height * wall_A_length
  let volume_B := wall_B_width * wall_B_height * wall_B_length
  let volume_C := wall_C_width * wall_C_height * wall_C_length
  volume_A + volume_B + volume_C = V := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_width_of_wall_A_correct_l692_69267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l692_69233

/-- The length of two trains passing each other on parallel tracks --/
theorem train_length_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) 
  (h1 : faster_speed = 46) 
  (h2 : slower_speed = 36) 
  (h3 : passing_time = 54) : 
  (((faster_speed - slower_speed) * (1000 / 3600) * passing_time) / 2) = 75 := by
  -- Convert given speeds from km/h to m/s
  have faster_speed_ms : ℝ := faster_speed * (1000 / 3600)
  have slower_speed_ms : ℝ := slower_speed * (1000 / 3600)
  
  -- Calculate relative speed in m/s
  let relative_speed : ℝ := faster_speed_ms - slower_speed_ms
  
  -- Calculate train length
  let train_length : ℝ := (relative_speed * passing_time) / 2
  
  -- Prove that train_length equals 75 meters
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l692_69233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_run_time_l692_69235

/-- The time it takes Jill to run up and down a hill -/
noncomputable def total_run_time (hill_height : ℝ) (speed_up : ℝ) (speed_down : ℝ) : ℝ :=
  hill_height / speed_up + hill_height / speed_down

/-- Theorem: Jill's total time to run up and down a 900 foot hill is 175 seconds -/
theorem jills_run_time :
  total_run_time 900 9 12 = 175 := by
  -- Unfold the definition of total_run_time
  unfold total_run_time
  -- Simplify the arithmetic
  simp [div_add_div]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_run_time_l692_69235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_no_intersection_l692_69272

-- Define the hyperbola
noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

-- Define the line y = 2x
def line (x y : ℝ) : Prop :=
  y = 2 * x

-- Theorem statement
theorem hyperbola_line_no_intersection (a b : ℝ) :
  (∀ x y : ℝ, hyperbola a b x y → ¬ line x y) ↔
  1 < eccentricity a b ∧ eccentricity a b ≤ Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_no_intersection_l692_69272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l692_69247

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y^2 = 4x -/
def focus : Point := ⟨1, 0⟩

/-- The given point Q -/
def Q : Point := ⟨2, 1⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Sum of distances from a point to Q and to the focus -/
noncomputable def sumDistances (p : Point) : ℝ :=
  distance p Q + distance p focus

/-- Theorem stating the minimum sum of distances is 3 -/
theorem min_sum_distances :
  ∃ (minDist : ℝ), minDist = 3 ∧
    ∀ (p : Point), p ∈ Parabola → sumDistances p ≥ minDist := by
  sorry

#check min_sum_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l692_69247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_km_nonnegative_for_odd_m_k7_m2k1_ge_mk4_l692_69287

/-- Definition of Km for positive integer m and non-negative real numbers x, y, z -/
def Km (m : ℕ) (x y z : ℝ) : ℝ :=
  x * (x - y)^m * (x - z)^m +
  y * (y - x)^m * (y - z)^m +
  z * (z - x)^m * (z - y)^m

/-- Definition of M for non-negative real numbers x, y, z -/
def M (x y z : ℝ) : ℝ :=
  (x - y)^2 * (y - z)^2 * (z - x)^2

theorem km_nonnegative_for_odd_m (m : ℕ) (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
    (hm : Odd m) :
    Km m x y z ≥ 0 := by
  sorry

theorem k7_m2k1_ge_mk4 (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
    Km 7 x y z + (M x y z)^2 * Km 1 x y z ≥ M x y z * Km 4 x y z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_km_nonnegative_for_odd_m_k7_m2k1_ge_mk4_l692_69287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l692_69259

noncomputable def triangle_problem (a b c : ℝ) (A : ℝ) : Prop :=
  a = Real.sqrt 7 ∧ 
  c = 3 ∧ 
  A = Real.pi / 3 ∧ 
  (b = 1 ∨ b = 2) ∧
  let S := (1/2) * b * c * Real.sin A
  S = (3 * Real.sqrt 3) / 4 ∨ S = (3 * Real.sqrt 3) / 2

theorem triangle_theorem : 
  ∃ (a b c A : ℝ), triangle_problem a b c A := by
  sorry

#check triangle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l692_69259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l692_69236

/-- The focus of a parabola y = ax^2 + bx + c -/
noncomputable def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  (h, k + 1 / (4 * a))

/-- Theorem: The focus of the parabola y = 4x^2 + 8x - 1 is (-1, -79/16) -/
theorem focus_of_specific_parabola :
  parabola_focus 4 8 (-1) = (-1, -79/16) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l692_69236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_50_factorial_l692_69268

theorem prime_divisors_of_50_factorial (n : ℕ) : 
  n = 50 → (Finset.filter Nat.Prime (Nat.divisors n.factorial)).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_50_factorial_l692_69268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_passes_through_fixed_point_minimum_a_for_inequality_l692_69261

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + 1

-- Define the tangent line at x = 1
noncomputable def tangent_line (a : ℝ) (x : ℝ) : ℝ := -a * (x - 1/2) + x

theorem tangent_passes_through_fixed_point :
  ∀ a : ℝ, tangent_line a (1/2) = 1/2 := by sorry

theorem minimum_a_for_inequality :
  (∀ a : ℕ, (∀ x : ℝ, x > 0 → f a x ≤ (a - 1) * x) → a ≥ 2) ∧
  (∀ x : ℝ, x > 0 → f 2 x ≤ x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_passes_through_fixed_point_minimum_a_for_inequality_l692_69261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l692_69201

noncomputable def z : ℂ := (Complex.abs (1 - Complex.I)) / (3 + Complex.I)

theorem modulus_of_z : Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l692_69201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cover_condition_l692_69251

/-- A hook shape is a configuration of 6 unit squares that can be rotated or reflected. -/
def HookShape : Type := Unit

/-- Predicate to check if a rectangle can be covered by hook shapes -/
def can_cover_with_hooks (m n : ℕ) : Prop :=
  (3 ∣ m ∧ 4 ∣ n) ∨ (3 ∣ n ∧ 4 ∣ m) ∨ 
  (12 ∣ m ∧ n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 5) ∨ 
  (12 ∣ n ∧ m ≠ 1 ∧ m ≠ 2 ∧ m ≠ 5)

/-- Represents an arrangement of hook shapes covering a rectangle -/
def covers_rectangle (m n : ℕ) (arrangement : Set HookShape) : Prop :=
  sorry

/-- Theorem stating the condition for covering a rectangle with hook shapes -/
theorem rectangle_cover_condition (m n : ℕ) :
  (∃ (arrangement : Set HookShape), covers_rectangle m n arrangement) ↔ 
  can_cover_with_hooks m n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cover_condition_l692_69251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_neg_reals_f_min_on_interval_f_max_on_interval_l692_69279

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

-- Theorem for monotonicity
theorem f_increasing_on_neg_reals : 
  StrictMonoOn f (Set.Iio 0) := by
  sorry

-- Theorem for minimum value
theorem f_min_on_interval : 
  ∃ (x : ℝ), x ∈ Set.Icc (-3) (-1) ∧ 
  f x = 1/10 ∧ 
  ∀ y ∈ Set.Icc (-3) (-1), f y ≥ f x := by
  sorry

-- Theorem for maximum value
theorem f_max_on_interval : 
  ∃ (x : ℝ), x ∈ Set.Icc (-3) (-1) ∧ 
  f x = 1/2 ∧ 
  ∀ y ∈ Set.Icc (-3) (-1), f y ≤ f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_neg_reals_f_min_on_interval_f_max_on_interval_l692_69279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_volume_l692_69208

theorem initial_mixture_volume 
  (initial_ratio : ℚ)
  (added_water : ℚ)
  (final_ratio : ℚ)
  (h1 : initial_ratio = 4 / 1)
  (h2 : added_water = 18)
  (h3 : final_ratio = 4 / 3) :
  ∃ (volume : ℚ), volume = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_volume_l692_69208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_center_square_is_correct_l692_69250

/-- Regular octagon with a center square -/
structure RegularOctagonWithCenterSquare where
  /-- Side length of the octagon -/
  side_length : ℝ
  /-- Assumption that side_length is positive -/
  side_length_pos : 0 < side_length

/-- The probability of a dart landing in the center square of a regular octagon -/
noncomputable def probability_center_square (octagon : RegularOctagonWithCenterSquare) : ℝ :=
  (Real.sqrt 2 - 1) / 2

/-- Theorem stating that the probability of a dart landing in the center square is (√2 - 1) / 2 -/
theorem probability_center_square_is_correct (octagon : RegularOctagonWithCenterSquare) :
    probability_center_square octagon = (Real.sqrt 2 - 1) / 2 := by
  sorry

#check probability_center_square_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_center_square_is_correct_l692_69250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrant_for_positive_sin_and_tan_quadrant_for_negative_tan_sin_product_quadrant_for_opposite_sin_cos_signs_quadrant_for_same_cos_tan_signs_l692_69285

-- Part 1
theorem quadrant_for_positive_sin_and_tan (α : ℝ) 
  (h1 : Real.sin α > 0) (h2 : Real.tan α > 0) : 
  0 < α ∧ α < Real.pi / 2 := by sorry

-- Part 2
theorem quadrant_for_negative_tan_sin_product (α : ℝ) 
  (h : Real.tan α * Real.sin α < 0) : 
  Real.pi / 2 < α ∧ α < 3 * Real.pi / 2 := by sorry

-- Part 3
theorem quadrant_for_opposite_sin_cos_signs (α : ℝ) 
  (h : Real.sin α * Real.cos α < 0) : 
  (Real.pi / 2 < α ∧ α < Real.pi) ∨ (3 * Real.pi / 2 < α ∧ α < 2 * Real.pi) := by sorry

-- Part 4
theorem quadrant_for_same_cos_tan_signs (α : ℝ) 
  (h : Real.cos α * Real.tan α > 0) : 
  (0 < α ∧ α < Real.pi / 2) ∨ (Real.pi / 2 < α ∧ α < Real.pi) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrant_for_positive_sin_and_tan_quadrant_for_negative_tan_sin_product_quadrant_for_opposite_sin_cos_signs_quadrant_for_same_cos_tan_signs_l692_69285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_QZY_is_30_l692_69257

-- Define the points
variable (P Q R S X Y Z T : EuclideanSpace ℝ (Fin 2))

-- Define the angles
def angle_YXS : ℝ := 20
def angle_ZYX : ℝ := 50

-- Define necessary geometric concepts
def Parallel (a b c d : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def OnLine (p a b : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def BetweenLines (p a b c d : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def Angle (a b c : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- State the theorem
theorem angle_QZY_is_30
  (hpar : Parallel P Q R S)
  (hZPQ : OnLine Z P Q)
  (hXRS : OnLine X R S)
  (hY : BetweenLines Y P Q R S)
  (hYXS : Angle Y X S = angle_YXS)
  (hZYX : Angle Z Y X = angle_ZYX) :
  Angle Q Z Y = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_QZY_is_30_l692_69257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_cube_edge_l692_69206

theorem new_cube_edge (e1 e2 e3 : ℝ) (h1 : e1 = 3) (h2 : e2 = 4) (h3 : e3 = 5) :
  Real.rpow (e1^3 + e2^3 + e3^3) (1/3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_cube_edge_l692_69206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l692_69223

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem sum_of_f_values : 
  f 1 + f 2 + f (1/2) + f 3 + f (1/3) + f 4 + f (1/4) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l692_69223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l692_69200

theorem inequality_solution_set (x : ℝ) :
  (4 + x^2 + 2*x*(Real.sqrt (2 - x^2)) < 8*(Real.sqrt (2 - x^2)) + 5*x) ↔ 
  (-1 < x ∧ x ≤ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l692_69200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_improper_fraction_count_l692_69216

theorem improper_fraction_count : 
  (Finset.filter (fun N : ℕ => Nat.gcd (N^2 + 9) (N + 5) > 1) 
    (Finset.range 2000)).card = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_improper_fraction_count_l692_69216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l692_69228

-- Define the equation
def equation (x a : ℝ) : Prop :=
  2 * Real.sin (Real.pi - Real.pi * x^2 / 12) * Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2)) + 1 =
  a + 2 * Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2)) * Real.cos (Real.pi * x^2 / 12)

-- Define the theorem
theorem smallest_a_value :
  (∃ (a : ℝ), ∃ (x : ℝ), equation x a) ∧
  (∀ (a' : ℝ), a' < 2 → ∀ (x : ℝ), ¬ equation x a') :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l692_69228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l692_69242

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side length opposite to A
  b : ℝ  -- Side length opposite to B
  c : ℝ  -- Side length opposite to C

-- Define vectors m and n
def m (t : Triangle) : ℝ × ℝ := (t.a + t.c, t.b)
def n (t : Triangle) : ℝ × ℝ := (t.a - t.c, t.b - t.a)

-- Define perpendicularity of vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h_perp : perpendicular (m t) (n t))
  (h_angles : t.A + t.B + t.C = Real.pi)
  (h_positive : t.A > 0 ∧ t.B > 0 ∧ t.C > 0) : 
  t.C = Real.pi / 3 ∧ Real.sqrt 3 / 2 < Real.sin t.A + Real.sin t.B ∧ Real.sin t.A + Real.sin t.B ≤ Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l692_69242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l692_69218

open Real

noncomputable def g (θ : ℝ) (x : ℝ) : ℝ := 1 / (x * cos θ) + log x

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x - (m - 1 + 2 * Real.exp 1) / x - log x

theorem problem_solution :
  (∀ x ≥ 1, Monotone (g θ)) →
  θ ∈ Set.Ioo (-π/2) (π/2) →
  (θ = 0) ∧
  (∃ x₀ > 0, IsLocalMax (f 0) x₀ ∧ x₀ = 2 * Real.exp 1 - 1) ∧
  ((∃ x₀ ∈ Set.Icc 1 (Real.exp 1), f m x₀ > g 0 x₀) → m > 4 * Real.exp 1 / ((Real.exp 1)^2 - 1))
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l692_69218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_section_a_l692_69220

theorem average_weight_section_a (students_a students_b : ℕ) 
  (avg_weight_b avg_weight_total : ℝ) : 
  students_a = 50 ∧ 
  students_b = 50 ∧ 
  avg_weight_b = 80 ∧ 
  avg_weight_total = 70 → 
  ∃ avg_weight_a : ℝ, 
    avg_weight_a * (students_a : ℝ) + avg_weight_b * (students_b : ℝ) = 
      avg_weight_total * ((students_a + students_b) : ℝ) ∧
    avg_weight_a = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_section_a_l692_69220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_speed_l692_69221

/-- Calculates the speed given total distance and total time -/
noncomputable def calculate_speed (total_distance : ℝ) (total_time : ℝ) : ℝ :=
  total_distance / total_time

/-- Proves that given a total distance of 275 miles and a total driving time of 5 hours, the speed is 55 mph -/
theorem johns_speed : calculate_speed 275 5 = 55 := by
  -- Unfold the definition of calculate_speed
  unfold calculate_speed
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_speed_l692_69221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_inequality_l692_69275

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x
def g (x : ℝ) : ℝ := x^3 - x + 6

-- State the theorem
theorem range_of_a_for_inequality :
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → 2 * (f a x) ≤ (deriv g x) + 2) ↔ a ≥ -2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_inequality_l692_69275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_2_prop_3_l692_69256

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∀ x y, (f x)^2 - (f y)^2 = f (x + y) * f (x - y)}

-- Proposition 2
theorem prop_2 : (λ x : ℝ => 2 * x) ∈ M := by
  intro x y
  simp
  ring

-- Proposition 3
theorem prop_3 (f : ℝ → ℝ) (h : f ∈ M) : ∀ x, f (-x) = -f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_2_prop_3_l692_69256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_indexed_sum_of_arithmetic_sequence_l692_69207

def arithmetic_sequence (a₁ d : ℤ) : ℕ → ℤ
  | 0     => a₁
  | (k+1) => a₁ + k * d

def sequence_sum (n : ℕ) (a₁ d : ℤ) : ℤ :=
  Finset.sum (Finset.range n) (λ k => arithmetic_sequence a₁ d k)

def even_indexed_sum (n : ℕ) (a₁ d : ℤ) : ℤ :=
  Finset.sum (Finset.range (n / 2)) (λ k => arithmetic_sequence a₁ d (2 * k + 1))

theorem even_indexed_sum_of_arithmetic_sequence 
  (n : ℕ) (a₁ d : ℤ) 
  (h₁ : n = 2020) 
  (h₂ : d = 2) 
  (h₃ : sequence_sum n a₁ d = 7040) : 
  even_indexed_sum n a₁ d = 4530 := by
  sorry

#eval even_indexed_sum 2020 1 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_indexed_sum_of_arithmetic_sequence_l692_69207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_equal_perimeter_area_triangles_l692_69217

/-- A right triangle with side lengths a, b, and c (where c is the hypotenuse) -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The perimeter of a right triangle -/
noncomputable def perimeter (t : RightTriangle) : ℝ := t.a + t.b + t.c

/-- The area of a right triangle -/
noncomputable def area (t : RightTriangle) : ℝ := t.a * t.b / 2

/-- Two right triangles are similar if their corresponding angles are equal -/
def similar (t1 t2 : RightTriangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.a / t2.a = t1.c / t2.c

theorem infinitely_many_equal_perimeter_area_triangles :
  ∃ (f : ℝ → RightTriangle), 
    (∀ k : ℝ, 0 < k → perimeter (f k) = area (f k)) ∧ 
    (∀ k1 k2 : ℝ, 0 < k1 → 0 < k2 → k1 ≠ k2 → ¬ similar (f k1) (f k2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_equal_perimeter_area_triangles_l692_69217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_negative_four_l692_69222

noncomputable def f (x : ℝ) : ℝ := (2*x^2 + 6*x - 8)/(x + 4)

theorem limit_f_at_negative_four :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ -4 → 0 < |x + 4| ∧ |x + 4| < δ → |f x + 10| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_negative_four_l692_69222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_position_of_2020_2187_l692_69239

/-- Represents a term in the sequence as a pair of natural numbers (numerator, denominator) -/
def SequenceTerm := ℕ × ℕ

/-- The sequence of fractions as described in the problem -/
def ourSequence : List SequenceTerm := sorry

/-- Function to check if a given fraction is in the sequence -/
def isInSequence (term : SequenceTerm) : Prop := sorry

/-- Function to find the position of a term in the sequence -/
def positionInSequence (term : SequenceTerm) : ℕ := sorry

theorem position_of_2020_2187 :
  isInSequence (2020, 2187) ∧ positionInSequence (2020, 2187) = 1553 := by
  sorry

#check position_of_2020_2187

end NUMINAMATH_CALUDE_ERRORFEEDBACK_position_of_2020_2187_l692_69239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_population_upper_limit_l692_69291

theorem room_population_upper_limit :
  ∀ (total : ℕ),
  (3 : ℚ) / 8 * (total : ℚ) = 36 →
  ∃ (n : ℕ), (5 : ℚ) / 12 * (total : ℚ) = n →
  total > 50 →
  ∀ m : ℕ,
  ((3 : ℚ) / 8 * (m : ℚ) = 36 ∧
   ∃ (k : ℕ), (5 : ℚ) / 12 * (m : ℚ) = k ∧
   m > 50) →
  m ≤ total →
  total ≤ 96 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_population_upper_limit_l692_69291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_1_calculation_2_calculation_3_calculation_4_l692_69278

theorem calculation_1 : 6 - (-2) + (-4) - 3 = 1 := by sorry

theorem calculation_2 : 8 / (-2) * (1 / 3) * (-3 / 2) = 2 := by sorry

theorem calculation_3 : (13 + (2 / 7 - 1 / 14) * 56) / (-1 / 4) = -100 := by sorry

theorem calculation_4 : |(- 5 / 6)| / ((-16 / 5) / 16 - 1) = -25 / 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_1_calculation_2_calculation_3_calculation_4_l692_69278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disc_coverage_l692_69270

/-- Represents a square on the checkerboard -/
structure Square where
  x : ℕ
  y : ℕ

/-- Represents the circular disc -/
structure Disc where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- Function to check if a square is completely covered by the disc -/
def is_square_covered (s : Square) (d : Disc) : Prop :=
  ∀ (corner_x corner_y : ℝ),
    (corner_x = s.x * d.radius * 2 ∨ corner_x = (s.x + 1) * d.radius * 2) →
    (corner_y = s.y * d.radius * 2 ∨ corner_y = (s.y + 1) * d.radius * 2) →
    (corner_x - d.center_x) ^ 2 + (corner_y - d.center_y) ^ 2 ≤ d.radius ^ 2

/-- The main theorem -/
theorem disc_coverage : 
  ∃ (covered_squares : Finset Square),
    let d : Disc := { center_x := 10, center_y := 8, radius := 1 }
    (∀ s ∈ covered_squares, is_square_covered s d) ∧
    (∀ s : Square, s.x < 10 ∧ s.y < 10 → (is_square_covered s d ↔ s ∈ covered_squares)) ∧
    covered_squares.card = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disc_coverage_l692_69270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_theorem_l692_69202

/-- The distance traveled by the center of a ball rolling along a track with two semicircular arcs and a straight segment -/
noncomputable def ball_travel_distance (ball_diameter : ℝ) (arc1_radius : ℝ) (arc2_radius : ℝ) (straight_segment : ℝ) : ℝ :=
  let ball_radius := ball_diameter / 2
  let arc1_center_path := arc1_radius - ball_radius
  let arc2_center_path := arc2_radius - ball_radius
  (arc1_center_path + arc2_center_path) * Real.pi + straight_segment

/-- The theorem stating that the ball's center travels 204π + 50 inches given the specified conditions -/
theorem ball_travel_distance_theorem :
  ball_travel_distance 6 120 90 50 = 204 * Real.pi + 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_theorem_l692_69202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_nonempty_domain_and_single_point_l692_69238

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => λ x => 0  -- Add a case for 0 to cover all natural numbers
| 1 => λ x => Real.sqrt (1 - x)
| (n + 1) => λ x => f n (Real.sqrt ((n + 1)^2 - x))

-- Define the domain of a function
def Domain (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | ∃ y : ℝ, f x = y}

-- State the theorem
theorem largest_nonempty_domain_and_single_point :
  (∃ N : ℕ, N = 5 ∧
    (∀ n : ℕ, n > N → Domain (f n) = ∅) ∧
    (Domain (f N) = {-231})) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_nonempty_domain_and_single_point_l692_69238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amc8_participants_is_33_l692_69258

/-- The number of distinct students taking the AMC 8 contest at Euclid Middle School -/
def amc8_participants : ℕ := 
  -- Define the number of students in each teacher's class
  let germain_students : ℕ := 15
  let newton_students : ℕ := 12
  let young_students : ℕ := 9
  
  -- Define the number of overlapping students
  let overlap_students : ℕ := 3
  
  -- Calculate the total number of distinct students
  germain_students + newton_students + young_students - overlap_students

theorem amc8_participants_is_33 : amc8_participants = 33 := by
  -- Unfold the definition of amc8_participants
  unfold amc8_participants
  
  -- Perform the calculation
  norm_num

#eval amc8_participants

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amc8_participants_is_33_l692_69258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hmmt_price_relation_l692_69288

/-- The price of HMMT stock as a function of x and y -/
noncomputable def price (k : ℝ) (x y : ℝ) : ℝ := k * x / y

/-- Theorem stating the relationship between initial and final prices -/
theorem hmmt_price_relation (k : ℝ) :
  price k 8 4 = 12 → price k 4 8 = 3 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check hmmt_price_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hmmt_price_relation_l692_69288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_AB_AC_l692_69293

def A : Fin 3 → ℝ := ![(-3), (-7), (-5)]
def B : Fin 3 → ℝ := ![0, (-1), (-2)]
def C : Fin 3 → ℝ := ![2, 3, 0]

def AB : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]
def AC : Fin 3 → ℝ := ![C 0 - A 0, C 1 - A 1, C 2 - A 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ := Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

theorem cosine_of_angle_between_AB_AC :
  dot_product AB AC / (magnitude AB * magnitude AC) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_AB_AC_l692_69293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_person_circle_arrangements_l692_69269

/-- The number of ways to arrange n people in a circle -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem eight_person_circle_arrangements :
  circularArrangements 8 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_person_circle_arrangements_l692_69269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l692_69246

noncomputable section

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  (∀ α β, Real.sin (2 * α) + Real.sin (2 * β) = 2 * Real.sin (α + β) * Real.cos (α - β)) →
  A + B + C = π →
  Real.sin (2 * A) + Real.sin (A - B + C) = Real.sin (C - A - B) + 1 / 2 →
  2 ≤ S ∧ S ≤ 3 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  S = 1 / 2 * a * b * Real.sin C →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  (Real.sin A * Real.sin B * Real.sin C = 1 / 8) ∧
  (16 * Real.sqrt 2 ≤ a * b * c ∧ a * b * c ≤ 24 * Real.sqrt 3) ∧
  (b * c * (b + c) > 16 * Real.sqrt 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l692_69246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_roll_is_six_l692_69237

/-- Represents the outcome of rolling a six-sided die multiple times -/
structure DieRolls where
  rolls : List Nat
  sum : Nat
  lastRoll : Nat

/-- Checks if the die rolls are valid according to the problem conditions -/
def isValidDieRolls (d : DieRolls) : Prop :=
  d.rolls.length = 12 ∧
  d.sum = 47 ∧
  d.rolls.all (λ x => 1 ≤ x ∧ x ≤ 6) ∧
  ∃ n, d.rolls.count n = 3 ∧ 
       ∀ m, m ≠ n → d.rolls.count m ≤ 2 ∧
  d.lastRoll = d.rolls.getLast!

theorem final_roll_is_six (d : DieRolls) (h : isValidDieRolls d) : 
  d.lastRoll = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_roll_is_six_l692_69237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_calculation_l692_69286

/-- Represents the compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Theorem stating the relationship between the initial investment and final amount --/
theorem investment_calculation (final_amount : ℝ) (rate : ℝ) (time : ℕ) 
  (h1 : final_amount = 7372.46)
  (h2 : rate = 0.065)
  (h3 : time = 2) :
  ∃ initial_investment : ℝ, 
    (compound_interest initial_investment rate time = final_amount) ∧ 
    (abs (initial_investment - 6510.00) < 0.01) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_calculation_l692_69286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_problem_l692_69266

def distribute_rolls (n : ℕ) (k : ℕ) : ℕ := 
  -- Number of ways to distribute n indistinguishable objects into k distinct boxes,
  -- with each box containing at least one object
  sorry -- Placeholder for the actual implementation

theorem bakery_problem : distribute_rolls 10 4 = 44 := by
  sorry -- Placeholder for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_problem_l692_69266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_beta_value_l692_69248

theorem angle_beta_value (α β : ℝ) :
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α = Real.sqrt 5 / 5 →
  Real.sin (α - β) = -(Real.sqrt 10) / 10 →
  β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_beta_value_l692_69248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_logarithm_l692_69241

theorem min_max_logarithm (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  max (Real.log (x / y + z)) (max (Real.log (y * z + 1 / x)) (Real.log (1 / (x * z) + y))) ≥ Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_logarithm_l692_69241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_f_right_pi_third_l692_69283

-- Define the determinant of a 2x2 matrix
def det2x2 (a₁ a₂ a₃ a₄ : ℝ) : ℝ := a₁ * a₄ - a₂ * a₃

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := det2x2 (Real.sin (Real.pi - x)) (Real.sqrt 3) (Real.cos (Real.pi + x)) 1

-- Theorem statement
theorem shift_f_right_pi_third (x : ℝ) : f (x - Real.pi/3) = 2 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_f_right_pi_third_l692_69283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_volume_correct_l692_69264

open Real

/-- The volume of the solid formed by rotating the region bounded by y = ln x, x = 2, and y = 0 around the y-axis -/
noncomputable def rotationVolume : ℝ := π * (4 * log 2 - 3 / 2)

/-- Theorem stating that the calculated volume is correct -/
theorem rotation_volume_correct :
  let f (x : ℝ) := log x
  let a : ℝ := 1  -- lower bound of x (when y = 0)
  let b : ℝ := 2  -- upper bound of x
  let V := π * ((b ^ 2 * (f b - f a)) - ∫ x in a..b, (f x) ^ 2)
  V = rotationVolume := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_volume_correct_l692_69264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_surface_area_l692_69243

/-- Area of an equilateral triangle with side length a. -/
noncomputable def area_equilateral_triangle (a : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * a^2

/-- Surface area of a regular tetrahedron with side length a. -/
noncomputable def surface_area_regular_tetrahedron (a : ℝ) : ℝ :=
  4 * area_equilateral_triangle a

/-- The surface area of a regular tetrahedron with side length a is √3 * a². -/
theorem regular_tetrahedron_surface_area (a : ℝ) (h : a > 0) :
  surface_area_regular_tetrahedron a = Real.sqrt 3 * a^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_surface_area_l692_69243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l692_69230

-- Define the basic objects
structure Point where
  x : ℝ
  y : ℝ

-- Define the circles
structure Circle where
  center : Point
  radius : ℝ

-- Define the line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of being on a circle
def OnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define the property of being inside a circle
def InsideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

-- Define the property of two circles intersecting
def CirclesIntersect (c1 c2 : Circle) : Prop :=
  ∃ p : Point, OnCircle p c1 ∧ OnCircle p c2

-- Define the property of a line intersecting a circle
def LineIntersectsCircle (l : Line) (c : Circle) (p : Point) : Prop :=
  OnCircle p c ∧ l.a * p.x + l.b * p.y + l.c = 0

-- Define the property of a line being tangent to a circle
def LineTangentToCircle (l : Line) (c : Circle) (p : Point) : Prop :=
  OnCircle p c ∧ l.a * p.x + l.b * p.y + l.c = 0 ∧
  ∀ q : Point, OnCircle q c → (q = p ∨ l.a * q.x + l.b * q.y + l.c ≠ 0)

-- Define the property of a line being parallel to another line
def LinesParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- Define the property of a circle being tangent to another circle
def CirclesTangent (c1 c2 : Circle) : Prop :=
  ∃ p : Point, OnCircle p c1 ∧ OnCircle p c2 ∧
  ∀ q : Point, (OnCircle q c1 ∧ OnCircle q c2) → q = p

-- State the theorem
theorem circle_tangency 
  (O₁ O₂ Q P X A B O : Point)
  (c₁ : Circle)
  (c₂ : Circle)
  (c : Circle)
  (l : Line)
  (pq : Line)
  (ab : Line)
  (h1 : CirclesIntersect c₁ c₂)
  (h2 : OnCircle Q c₁)
  (h3 : OnCircle Q c₂)
  (h4 : OnCircle P c₂)
  (h5 : InsideCircle P c₁)
  (h6 : LineIntersectsCircle pq c₁ X)
  (h7 : X ≠ Q)
  (h8 : LineTangentToCircle ab c₁ X)
  (h9 : OnCircle A c₂)
  (h10 : OnCircle B c₂)
  (h11 : LinesParallel l ab)
  (h12 : OnCircle A c)
  (h13 : OnCircle B c)
  (h14 : LineTangentToCircle l c P) :
  CirclesTangent c c₁ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l692_69230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_implies_k_values_l692_69231

-- Define the lines that form the quadrilateral
def line1 : ℝ := 1
def line2 : ℝ := -1
def line3 : ℝ := 3
def line4 (k : ℝ) (x : ℝ) : ℝ := k * x - 3

-- Define the area of the quadrilateral
noncomputable def quadrilateralArea (k : ℝ) : ℝ := 
  if k < 0 then
    2 * ((1 - 2/k) + (1 - 6/k))
  else
    2 * ((2/k - 1) + (6/k - 1))

-- Theorem statement
theorem quadrilateral_area_implies_k_values (k : ℝ) :
  quadrilateralArea k = 12 → k = -2 ∨ k = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_implies_k_values_l692_69231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_plate_number_puzzle_l692_69213

theorem car_plate_number_puzzle : ∃ (a b : ℕ), 
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (1 ≤ b ∧ b ≤ 9) ∧ 
  (41 ≤ b * 10 + b ∧ b * 10 + b ≤ 49) ∧
  (∀ i : ℕ, i ∈ ({1, 2, 3, 4, 5, 7, 8, 9} : Finset ℕ) → (a * 1000 + a * 100 + b * 10 + b) % i = 0) ∧
  (a * 1000 + a * 100 + b * 10 + b) % 6 ≠ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_plate_number_puzzle_l692_69213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_value_l692_69299

noncomputable section

-- Define the two curves
def curve1 (a : ℝ) (x : ℝ) : ℝ := a / x
def curve2 (x : ℝ) : ℝ := x^2

-- Define the slopes of the tangents at the intersection point
def slope1 (a : ℝ) (x : ℝ) : ℝ := -a / x^2
def slope2 (x : ℝ) : ℝ := 2 * x

-- Define the condition for perpendicular tangents
def perpendicular_tangents (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ curve1 a x = curve2 x ∧ slope1 a x * slope2 x = -1

-- State the theorem
theorem perpendicular_tangents_value :
  ∃ a : ℝ, perpendicular_tangents a ∧ a^2 = 1/8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_value_l692_69299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l692_69225

/-- The profit function for the company -/
noncomputable def profit (x : ℝ) : ℝ := 19 - 24 / (x + 2) - (3/2) * x

/-- The theorem stating the maximum profit for the company -/
theorem max_profit (a : ℝ) (ha : a > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ a → profit x ≤ (if a ≥ 2 then 10 else 19 - 24 / (a + 2) - (3/2) * a)) ∧
  (if a ≥ 2 
   then profit 2 = 10
   else profit a = 19 - 24 / (a + 2) - (3/2) * a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l692_69225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l692_69290

-- Define the ellipse
def ellipse_equation (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 6

-- Define the focal length
noncomputable def focal_length (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

-- Theorem statement
theorem ellipse_focal_length :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
  (∀ x y : ℝ, ellipse_equation x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  focal_length a b = 2 := by
  -- Proof goes here
  sorry

#check ellipse_focal_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l692_69290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_dozen_oranges_cost_l692_69209

/-- The cost of oranges in dollars given the number of dozens -/
noncomputable def orange_cost (dozens : ℝ) : ℝ :=
  (21.90 / 3) * dozens

theorem five_dozen_oranges_cost :
  orange_cost 5 = 36.50 :=
by
  -- Unfold the definition of orange_cost
  unfold orange_cost
  -- Simplify the expression
  simp [mul_div_assoc]
  -- Perform the numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_dozen_oranges_cost_l692_69209
