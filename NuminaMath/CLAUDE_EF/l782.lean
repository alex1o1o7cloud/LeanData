import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_non_invertible_l782_78236

def matrix (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3 * y + 2, 4],
    ![-7 * y + 6, 5]]

theorem matrix_non_invertible :
  ¬(IsUnit (Matrix.det (matrix (14 / 43)))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_non_invertible_l782_78236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anne_distance_from_others_l782_78284

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool

/-- Distance between two runners at time t -/
noncomputable def distance (trackLength : ℝ) (runner1 runner2 : Runner) (t : ℝ) : ℝ :=
  let relativeDistance := (runner1.speed * (if runner1.direction then 1 else -1) - 
                           runner2.speed * (if runner2.direction then 1 else -1)) * t
  min (relativeDistance % trackLength) (trackLength - (relativeDistance % trackLength))

/-- The main theorem statement -/
theorem anne_distance_from_others (trackLength : ℝ) (anne beth carmen : Runner) 
  (h1 : trackLength = 300)
  (h2 : anne.speed ≠ beth.speed)
  (h3 : anne.speed ≠ carmen.speed) :
  ∃ (t : ℝ), t ≥ 0 ∧ 
    distance trackLength anne beth t ≥ 100 ∧ 
    distance trackLength anne carmen t ≥ 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_anne_distance_from_others_l782_78284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_parallel_to_all_planes_l782_78257

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two lines are parallel -/
def parallel_lines (a b : Line3D) : Prop :=
  ∃ (k : ℝ), a.direction = (k * b.direction.1, k * b.direction.2.1, k * b.direction.2.2)

/-- A line is parallel to a plane -/
def line_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  l.direction.1 * p.normal.1 + l.direction.2.1 * p.normal.2.1 + l.direction.2.2 * p.normal.2.2 = 0

/-- A line lies in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  ((l.point.1 - p.point.1) * p.normal.1 + 
   (l.point.2.1 - p.point.2.1) * p.normal.2.1 + 
   (l.point.2.2 - p.point.2.2) * p.normal.2.2 = 0) ∧ 
  line_parallel_to_plane l p

/-- The main theorem -/
theorem not_always_parallel_to_all_planes (a b : Line3D) :
  parallel_lines a b →
  ¬(∀ (p : Plane3D), b.direction.1 * p.normal.1 + b.direction.2.1 * p.normal.2.1 + b.direction.2.2 * p.normal.2.2 = 0 → 
    line_parallel_to_plane a p) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_parallel_to_all_planes_l782_78257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_value_in_quadratic_sequence_l782_78219

def quadratic_sequence (a b c : ℝ) (x : ℕ → ℝ) : ℕ → ℝ := 
  λ n ↦ a * (x n)^2 + b * (x n) + c

def is_equally_spaced (x : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, x (n + 1) - x n = d

def sequence_values : List ℝ := [3844, 3989, 4144, 4311, 4496, 4689, 4892, 5105]

theorem incorrect_value_in_quadratic_sequence 
  (a b c : ℝ) 
  (x : ℕ → ℝ) 
  (h1 : is_equally_spaced x) 
  (h2 : ∀ n : Fin 8, quadratic_sequence a b c x n.val = sequence_values.get n) :
  ∃ n : Fin 8, sequence_values.get n ≠ 4496 ∧ 
    ∀ m : Fin 8, m ≠ n → sequence_values.get m = quadratic_sequence a b c x m.val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_value_in_quadratic_sequence_l782_78219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_pool_volume_12_5_l782_78223

/-- The volume of a cone-shaped pool -/
noncomputable def cone_pool_volume (base_diameter : ℝ) (depth : ℝ) : ℝ :=
  (1 / 3) * Real.pi * (base_diameter / 2)^2 * depth

/-- Theorem: The volume of a cone-shaped pool with base diameter 12 feet and depth 5 feet is 60π cubic feet -/
theorem cone_pool_volume_12_5 :
  cone_pool_volume 12 5 = 60 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_pool_volume_12_5_l782_78223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_one_zero_to_line_l782_78298

/-- The distance from a point (ρ₀, θ₀) to a line ρ = 2 / (cos θ + sin θ) in polar coordinates -/
noncomputable def distance_to_line (ρ₀ θ₀ : ℝ) : ℝ :=
  |ρ₀ - 2 / (Real.cos θ₀ + Real.sin θ₀)|

/-- The theorem stating that the distance from (1,0) to the line ρ(cos θ + sin θ) = 2 is 1 -/
theorem distance_from_one_zero_to_line : distance_to_line 1 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_one_zero_to_line_l782_78298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_class_strength_l782_78230

theorem original_class_strength : ℕ := by
  -- Define the initial average age
  let initial_avg : ℕ := 40

  -- Define the number of new students
  let new_students : ℕ := 17

  -- Define the average age of new students
  let new_students_avg : ℕ := 32

  -- Define the new average age after new students join
  let new_avg : ℕ := 36

  -- Define the original strength
  let original_strength : ℕ := 17

  -- Proof that the original strength is correct
  have h : (original_strength * initial_avg) +
           (new_students * new_students_avg) =
           ((original_strength + new_students) * new_avg) := by
    -- Actual computation
    norm_num

  -- Conclude the theorem
  exact original_strength

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_class_strength_l782_78230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_two_intersections_l782_78280

/-- A hyperbola in a 2D plane -/
structure Hyperbola where
  -- Add necessary fields to define a hyperbola
  a : ℝ
  b : ℝ

/-- A line in a 2D plane -/
structure Line where
  -- Add necessary fields to define a line
  m : ℝ
  c : ℝ

/-- Represents the number of intersection points between a line and a hyperbola -/
def intersection_count (h : Hyperbola) (l : Line) : ℕ :=
  -- Placeholder implementation
  2

/-- Theorem stating that it's not always true that a line intersects a hyperbola at exactly two points -/
theorem not_always_two_intersections :
  ¬ ∀ (h : Hyperbola) (l : Line), intersection_count h l = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_two_intersections_l782_78280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l782_78299

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / (1 + a * x^2)

-- State the theorem
theorem monotonic_f_implies_a_range (a : ℝ) :
  a > 0 →
  (∀ x : ℝ, Monotone (f a)) →
  0 < a ∧ a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l782_78299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l782_78266

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (9 + n) = 11 → n = 112 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l782_78266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l782_78238

/-- Represents an ellipse with given parameters -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a line with slope k -/
structure Line where
  k : ℝ

/-- The standard form of the ellipse given the conditions -/
def standard_ellipse_equation (e : Ellipse) : Prop :=
  e.a = Real.sqrt 2 ∧ e.b = 1

/-- The maximum area of triangle BPQ -/
noncomputable def max_area_BPQ (e : Ellipse) (l : Line) : ℝ := 2 * Real.sqrt 2 / 3

/-- Main theorem stating the properties of the ellipse and the maximum area -/
theorem ellipse_properties (e : Ellipse) (l : Line) :
  (e.a^2 - e.b^2) / e.a^2 = 1/2 →  -- eccentricity condition
  e.b = 1 →                        -- minor axis endpoint condition
  standard_ellipse_equation e ∧ 
  ∃ k, l.k = k ∧ max_area_BPQ e l = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l782_78238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_region_l782_78273

-- Define the functions
def f₀ : ℝ → ℝ := fun x ↦ |x|
def f₁ : ℝ → ℝ := fun x ↦ |f₀ x - 1|
def f₂ : ℝ → ℝ := fun x ↦ |f₁ x - 2|

-- Define the area of the closed region
noncomputable def area : ℝ := ∫ x in Set.Icc (-3) 3, f₂ x

-- State the theorem
theorem area_of_closed_region : area = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_region_l782_78273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_distance_sum_l782_78255

/-- Predicate to check if four points form a rectangle -/
def IsRectangle (A B C D : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), 
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
    (D.1 - C.1)^2 + (D.2 - C.2)^2 = a^2 ∧
    (A.1 - D.1)^2 + (A.2 - D.2)^2 = b^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = (D.1 - B.1)^2 + (D.2 - B.2)^2

/-- Given a rectangle ABCD and an arbitrary point E, 
    the sum of squared distances from E to opposite corners A and C 
    is equal to the sum of squared distances from E to opposite corners B and D -/
theorem rectangle_diagonal_distance_sum (A B C D E : ℝ × ℝ) : 
  IsRectangle A B C D → 
  (dist A E)^2 + (dist C E)^2 = (dist B E)^2 + (dist D E)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_distance_sum_l782_78255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_intersection_length_bound_l782_78254

/-- Represents a set of non-overlapping arcs on a circle's circumference -/
structure ArcSet where
  arcs : Set (Set ℝ)
  nonoverlapping : ∀ a b, a ∈ arcs → b ∈ arcs → a ≠ b → a ∩ b = ∅

/-- Calculates the total length of arcs in an ArcSet -/
noncomputable def arcLength (S : ArcSet) : ℝ := sorry

/-- Rotates an ArcSet counterclockwise by a given angle -/
noncomputable def rotate (S : ArcSet) (angle : ℝ) : ArcSet := sorry

/-- Theorem: There exists a rotation of A that intersects B with sufficient length -/
theorem rotation_intersection_length_bound 
  (A B : ArcSet) 
  (m : ℕ) 
  (hm : m > 0) 
  (hB : ∀ b, b ∈ B.arcs → arcLength ⟨{b}, sorry⟩ = π / m) :
  ∃ k : ℕ, 
    arcLength ⟨(rotate A (k * π / m)).arcs ∩ B.arcs, sorry⟩ ≥ 
    (1 / (2 * π)) * arcLength A * arcLength B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_intersection_length_bound_l782_78254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l782_78225

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ := fun n => a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sumArithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem max_sum_arithmetic_sequence
  (a₁ : ℝ) (d : ℝ) (h₁ : a₁ > 0) (h₂ : d ≠ 0)
  (h₃ : sumArithmeticSequence a₁ d 5 = sumArithmeticSequence a₁ d 9) :
  ∃ (n : ℕ), n = 7 ∧ ∀ (m : ℕ), sumArithmeticSequence a₁ d m ≤ sumArithmeticSequence a₁ d n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l782_78225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_b_product_l782_78218

/-- A square is formed by the lines y=3, y=8, x=2, and x=b. -/
structure Square where
  b : ℝ
  is_square : ∃ (s : ℝ), s > 0 ∧ 
    (Set.Icc 2 b = Set.Icc 2 (2 + s) ∨ Set.Icc b 2 = Set.Icc (2 - s) 2) ∧
    Set.Icc 3 8 = Set.Icc 3 (3 + s)

/-- The product of the possible values for b in a square formed by y=3, y=8, x=2, and x=b is -21. -/
theorem square_b_product : 
  ∃ (b₁ b₂ : ℝ), (∃ (sq₁ : Square), sq₁.b = b₁) ∧ (∃ (sq₂ : Square), sq₂.b = b₂) ∧ b₁ * b₂ = -21 :=
by
  -- We'll prove this later
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_b_product_l782_78218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eiffel_tower_scale_model_ratio_l782_78281

/-- The height of the Eiffel Tower in feet -/
noncomputable def eiffel_tower_height : ℚ := 984

/-- The height of the scale model in inches -/
noncomputable def model_height : ℚ := 6

/-- The ratio of the Eiffel Tower's height to the model's height in feet per inch -/
noncomputable def height_ratio : ℚ := eiffel_tower_height / model_height

theorem eiffel_tower_scale_model_ratio :
  height_ratio = 164 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eiffel_tower_scale_model_ratio_l782_78281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_plus_one_l782_78229

-- Define the function (marked as noncomputable due to Real.sqrt)
noncomputable def f (x : ℝ) := Real.sqrt (x + 1)

-- State the theorem
theorem domain_of_sqrt_x_plus_one :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≥ -1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_plus_one_l782_78229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_single_digit_sum_l782_78242

theorem no_single_digit_sum (n : ℕ) (h1 : n < 10) : 
  ¬ (∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ : ℕ), 
    (∀ i ∈ [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, a₉, a₁₀, a₁₁, a₁₂, a₁₃], 
      ∃ k : ℕ, i = n * (10^k - 1) / 9) ∧ 
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ + a₁₂ + a₁₃ = 8900098) :=
by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_single_digit_sum_l782_78242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_midlines_properties_l782_78222

/-- A triangle with midlines -/
structure TriangleWithMidlines where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  K₁ : ℝ
  K₂ : ℝ
  K₃ : ℝ

/-- Predicate to check if midlines intersect at a point -/
def midlines_intersect_at (P : ℝ × ℝ) (T : TriangleWithMidlines) : Prop :=
  sorry

/-- Predicate to check if a triangle is isosceles -/
def is_isosceles (A B C : ℝ × ℝ) : Prop :=
  sorry

/-- Theorem about properties of a triangle with midlines -/
theorem triangle_midlines_properties (T : TriangleWithMidlines) :
  (∃ P : ℝ × ℝ, midlines_intersect_at P T) ∧
  (T.K₂ = T.K₃ → is_isosceles T.A T.B T.C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_midlines_properties_l782_78222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_B_wins_l782_78215

/-- Represents a polynomial of degree 10 with 9 variable coefficients -/
structure Polynomial10 where
  coeffs : Fin 9 → ℝ

/-- Represents the state of the game after each move -/
inductive GameState
  | ongoing (poly : Polynomial10) (moveCount : Nat)
  | finished (poly : Polynomial10)

/-- Represents a player's strategy -/
def Strategy := GameState → ℝ

/-- Determines if a polynomial has a real root -/
def hasRealRoot (p : Polynomial10) : Prop := sorry

/-- The game play function -/
def play (stratA stratB : Strategy) : GameState := sorry

/-- Main theorem: Player B has a winning strategy -/
theorem player_B_wins :
  ∃ (stratB : Strategy), ∀ (stratA : Strategy),
    let finalState := play stratA stratB
    match finalState with
    | GameState.finished poly => hasRealRoot poly
    | _ => False := by
  sorry

#check player_B_wins


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_B_wins_l782_78215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l782_78290

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x + 1/x) + 1/(x^2 + 1/x^2)

theorem f_min_value (x : ℝ) (h : x > 0) : f x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l782_78290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_base_l782_78275

/-- Given a function f(x) = log_a(3-x) + log_a(x+1) where 0 < a < 1,
    if the minimum value of f(x) is -2, then a = 1/2 -/
theorem min_value_implies_base (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∃ f : ℝ → ℝ, (∀ x, f x = Real.log (3 - x) / Real.log a + Real.log (x + 1) / Real.log a) ∧
   (∃ m, ∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = -2) ∧ (∀ x, f x ≥ -2)) →
  a = 1 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_base_l782_78275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncoverable_subset_exists_l782_78250

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Predicate to check if a set of points is covered by a set of lines -/
def covered_by_lines (points : Finset Point) (lines : Finset Line) : Prop :=
  ∀ p ∈ points, ∃ l ∈ lines, p.on_line l

/-- The main theorem -/
theorem uncoverable_subset_exists
  (points : Finset Point)
  (h_card : points.card = 666)
  (h_uncoverable : ¬ ∃ lines : Finset Line, lines.card = 10 ∧ covered_by_lines points lines) :
  ∃ subset : Finset Point, subset.card = 66 ∧
    subset ⊆ points ∧
    ¬ ∃ lines : Finset Line, lines.card = 10 ∧ covered_by_lines subset lines :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncoverable_subset_exists_l782_78250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_total_operations_for_f_l782_78207

/-- The polynomial f(x) = 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 7 -/
def f (x : ℝ) : ℝ := 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 7

/-- The number of terms in the polynomial -/
def n : ℕ := 7

/-- A function to represent the number of operations in Horner's method -/
def number_of_operations_horner (f : ℝ → ℝ) (n : ℕ) : ℕ := 2*n - 2

/-- The theorem stating that the number of operations in Horner's method is 2n - 2 -/
theorem horner_method_operations (f : ℝ → ℝ) (n : ℕ) :
  (n > 0) → (number_of_operations_horner f n = 2*n - 2) :=
by sorry

/-- The main theorem proving that the total number of operations for f(x) using Horner's method is 12 -/
theorem total_operations_for_f :
  number_of_operations_horner f n = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_total_operations_for_f_l782_78207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_and_area_l782_78245

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ A < Real.pi ∧ B > 0 ∧ B < Real.pi ∧ C > 0 ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_side_and_area 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_a : a = 1)
  (h_b : b = Real.sqrt 3)
  (h_C : C = Real.pi/6) :
  c = 1 ∧ (1/2 * a * b * Real.sin C) = Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_and_area_l782_78245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_example_l782_78248

/-- Calculates the speed of a train in km/hr given its length in meters and time in seconds to pass a fixed point. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3600 / 1000

/-- Theorem: A train 270 m long that passes a tree in 9 seconds has a speed of 108 km/hr. -/
theorem train_speed_example : train_speed 270 9 = 108 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  simp [div_eq_mul_inv]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_example_l782_78248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_in_interval_l782_78237

open Real

/-- The original function g(x) -/
noncomputable def g (x : ℝ) : ℝ := 3 * sin (2 * x + π / 6)

/-- The transformed function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 3 * cos (4 * x)

/-- Theorem stating that f(x) is decreasing in the interval (0, π/4) -/
theorem f_decreasing_in_interval :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/4 → f x₂ < f x₁ := by
  sorry

#check f_decreasing_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_in_interval_l782_78237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_is_40_l782_78205

/-- Represents a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three side lengths can form a valid triangle. -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Calculates the perimeter of a triangle. -/
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Generates the next triangle in the sequence based on the incircle tangent points. -/
noncomputable def next_triangle (t : Triangle) : Triangle :=
  { a := (t.b + t.c - t.a) / 2,
    b := (t.a + t.c - t.b) / 2,
    c := (t.a + t.b - t.c) / 2 }

/-- The main theorem stating that the perimeter of the last triangle in the sequence is 40. -/
theorem last_triangle_perimeter_is_40 (t : Triangle) 
  (h : t.a = 8 ∧ t.b = 15 ∧ t.c = 17) : 
  perimeter t = 40 ∧ ¬is_valid_triangle (next_triangle t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_is_40_l782_78205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l782_78213

noncomputable def f (a b x : ℝ) : ℝ := (2 * a * Real.log x) / (x + 1) + b

theorem function_properties (a b : ℝ) 
  (h1 : f a b 1 = 2) 
  (h2 : deriv (f a b) 1 = -1) :
  (a = -1 ∧ b = 2) ∧ 
  (∀ x : ℝ, x > 0 → x ≠ 1 → f (-1) 2 x > (2 * Real.log x) / (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l782_78213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_to_negative_one_equals_one_half_l782_78244

theorem two_to_negative_one_equals_one_half : (2 : ℝ)^(-1 : ℤ) = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_to_negative_one_equals_one_half_l782_78244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xz_length_in_similar_triangles_l782_78292

-- Define the triangles and their properties
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  angle : ℝ

-- Define the similarity ratio
noncomputable def similarityRatio (t1 t2 : Triangle) : ℝ := t1.x / t2.x

theorem xz_length_in_similar_triangles 
  (xyz wuv : Triangle) 
  (h_similar : similarityRatio xyz wuv = xyz.y / wuv.y) 
  (h_xy : xyz.x = 9) 
  (h_yz : xyz.y = 21) 
  (h_wv : wuv.x = 4.5) 
  (h_uv : wuv.y = 7.5) 
  (h_wu : wuv.z = 7.5) 
  (h_angle : xyz.angle = 110 ∧ wuv.angle = 110) : 
  xyz.z = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xz_length_in_similar_triangles_l782_78292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_is_588_l782_78282

/-- Calculates the volume of a frustum given the dimensions of the original and smaller pyramids --/
noncomputable def frustum_volume (base_edge : ℝ) (altitude : ℝ) (small_base_edge : ℝ) (small_altitude : ℝ) : ℝ :=
  let original_volume := (1/3) * base_edge^2 * altitude
  let scale_factor := small_base_edge / base_edge
  let small_volume := scale_factor^3 * original_volume
  original_volume - small_volume

/-- The volume of the frustum is 588 cubic centimeters --/
theorem frustum_volume_is_588 :
  frustum_volume 15 10 9 6 = 588 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_is_588_l782_78282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l782_78287

open Real

/-- The function f(x) = tan(πx/6 + π/3) -/
noncomputable def f (x : ℝ) : ℝ := tan ((π / 6 * x) + (π / 3))

/-- The domain of f -/
def domain_f : Set ℝ := {x | ∀ k : ℤ, x ≠ 1 + 6 * k}

/-- Theorem stating that domain_f is the correct domain of f -/
theorem domain_of_f : 
  ∀ x : ℝ, f x ≠ 0 / 0 ↔ x ∈ domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l782_78287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_minimum_value_f_period_l782_78228

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x

theorem f_properties :
  (∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x₀ : ℝ), f x₀ = m)) ∧
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x) := by
  sorry

theorem f_minimum_value : 
  ∃ (m : ℝ), m = -Real.sqrt 2 ∧ (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x₀ : ℝ), f x₀ = m) := by
  sorry

theorem f_period : 
  ∃ (p : ℝ), p = 2 * Real.pi ∧ p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_minimum_value_f_period_l782_78228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_max_value_max_at_pi_third_l782_78252

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - (Real.cos x)^2 * Real.sin (2 * x)

-- Theorem for the center of symmetry
theorem center_of_symmetry : 
  ∀ x : ℝ, f (π / 2 - x) + f (π / 2 + x) = 0 := by sorry

-- Theorem for the maximum value
theorem max_value : 
  ∃ x : ℝ, f x = 3 * Real.sqrt 3 / 8 ∧ ∀ y : ℝ, f y ≤ 3 * Real.sqrt 3 / 8 := by sorry

-- Theorem for the point where maximum occurs
theorem max_at_pi_third : 
  f (π / 3) = 3 * Real.sqrt 3 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_max_value_max_at_pi_third_l782_78252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primitive_roots_l782_78214

/-- Generalized Euler's totient function -/
noncomputable def L (m : ℕ) : ℕ := sorry

/-- Euler's totient function -/
noncomputable def φ (m : ℕ) : ℕ := sorry

/-- A number is of the form 2, 4, p^α, or 2p^α where p is an odd prime -/
def is_special_form (m : ℕ) : Prop := sorry

theorem no_primitive_roots (m : ℕ) (h : ¬ is_special_form m) :
  ∀ x : ℕ, Nat.Coprime x m → (x ^ L m) % m = 1 → L m < φ m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primitive_roots_l782_78214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_digits_l782_78261

def sum_of_digits (m : ℕ) : ℕ :=
  if m = 0 then 0 else (m % 10) + sum_of_digits (m / 10)

theorem smallest_sum_of_digits :
  (∃ (n : ℕ), n > 0 ∧ sum_of_digits (5^n + 6^n + 2022^n) = 8) ∧
  (∀ (n : ℕ), n > 0 → sum_of_digits (5^n + 6^n + 2022^n) ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_digits_l782_78261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_square_identity_l782_78268

theorem cosine_square_identity (θ : ℝ) : 
  (Real.cos θ) ^ 2 = (1 / 2) * Real.cos (2 * θ) + 0 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_square_identity_l782_78268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pairs_count_l782_78274

theorem pairs_count : 
  let count := Finset.filter (fun p : ℕ × ℕ => 
    p.1 ≤ 1000 ∧ 
    p.2 ≤ 1000 ∧ 
    (p.1 : ℝ) / ((p.2 : ℝ) + 1) < Real.sqrt 2 ∧ 
    Real.sqrt 2 < ((p.1 : ℝ) + 1) / (p.2 : ℝ))
    (Finset.product (Finset.range 1001) (Finset.range 1001))
  Finset.card count = 1706 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pairs_count_l782_78274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l782_78294

-- Define the quadratic inequality
noncomputable def quadratic_inequality (a b x : ℝ) : Prop := a * x^2 + 2 * x + b > 0

-- Define the solution set
def solution_set (c : ℝ) : Set ℝ := {x : ℝ | x ≠ c}

-- Define the expression we're interested in
noncomputable def expression (a b c : ℝ) : ℝ := (a^2 + b^2 + 7) / (a + c)

-- State the theorem
theorem quadratic_inequality_range (a b c : ℝ) :
  (∀ x, x ∈ solution_set c ↔ quadratic_inequality a b x) →
  a + c ≠ 0 →
  ∃ y, expression a b c = y ↔ (y ≤ -6 ∨ y ≥ 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l782_78294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_paving_cost_l782_78263

/-- The cost of paving a rectangular floor -/
theorem floor_paving_cost 
  (length width rate_per_sq_meter : ℝ) 
  (h1 : length = 8) 
  (h2 : width = 4.75) 
  (h3 : rate_per_sq_meter = 900) : 
  length * width * rate_per_sq_meter = 34200 := by
  -- Replace all occurrences with their values
  rw [h1, h2, h3]
  -- Simplify the arithmetic
  norm_num

#check floor_paving_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_paving_cost_l782_78263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_45_implies_a_eq_b_l782_78224

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_a_gt_b : a > b

/-- The angle between the asymptotes of a hyperbola -/
noncomputable def asymptote_angle (h : Hyperbola) : ℝ := 2 * Real.arctan (h.b / h.a)

theorem hyperbola_asymptote_angle_45_implies_a_eq_b (h : Hyperbola) 
  (h_angle : asymptote_angle h = Real.pi / 4) : h.a = h.b := by
  sorry

#check hyperbola_asymptote_angle_45_implies_a_eq_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_45_implies_a_eq_b_l782_78224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_shaded_cubes_l782_78269

/-- Represents a 4×4×4 cube with a specific shading pattern. -/
structure ShadedCube where
  /-- The number of smaller cubes along each edge of the large cube. -/
  edge_length : Nat
  /-- The total number of smaller cubes in the large cube. -/
  total_cubes : Nat
  /-- The number of shaded cubes on each face. -/
  shaded_per_face : Nat
  /-- The property that opposite faces have the same shading pattern. -/
  opposite_faces_same : Bool

/-- A function to calculate the number of uniquely shaded cubes. -/
def number_of_uniquely_shaded_cubes (c : ShadedCube) : Nat :=
  sorry

/-- The theorem stating the number of uniquely shaded cubes. -/
theorem unique_shaded_cubes (c : ShadedCube)
  (h1 : c.edge_length = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.shaded_per_face = 5)
  (h4 : c.opposite_faces_same = true) :
  ∃ n : Nat, n = 30 ∧ n = number_of_uniquely_shaded_cubes c :=
by
  sorry

#check unique_shaded_cubes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_shaded_cubes_l782_78269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l782_78262

-- Define the ellipse C
noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define a point on the ellipse
def point_on_ellipse (a b : ℝ) : Prop :=
  ellipse a b (Real.sqrt 3) (1/2)

-- Define the line l
def line (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

-- Define the condition for OA + OB = OP
def sum_vectors (xa ya xb yb xp yp : ℝ) : Prop :=
  xa + xb = xp ∧ ya + yb = yp

-- Main theorem
theorem ellipse_line_intersection :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  eccentricity a b = Real.sqrt 3 / 2 →
  point_on_ellipse a b →
  ∃ (k m : ℝ), 
    ((k = 3/8 ∧ m = 5/8) ∨ (k = 0 ∧ m = 1)) ∧
    line k m 1 1 ∧
    ∃ (xa ya xb yb xp yp : ℝ),
      ellipse a b xa ya ∧
      ellipse a b xb yb ∧
      ellipse a b xp yp ∧
      line k m xa ya ∧
      line k m xb yb ∧
      sum_vectors xa ya xb yb xp yp :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l782_78262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_branches_after_ten_weeks_l782_78283

/-- Represents the number of new branches grown in a specific week -/
def new_branches (week : ℕ) : ℕ := 2^week

/-- Represents the total number of branches after a given number of weeks -/
def total_branches (weeks : ℕ) : ℕ :=
  Finset.sum (Finset.range weeks) (λ i => new_branches (i + 1))

/-- The theorem states that the total number of branches after 10 weeks is 2046 -/
theorem branches_after_ten_weeks :
  total_branches 10 = 2046 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_branches_after_ten_weeks_l782_78283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_restricted_polynomial_l782_78251

/-- A polynomial with coefficients in {0, -1, 1} -/
def RestrictedPolynomial (R : Type*) [Ring R] := 
  { p : Polynomial R // ∀ i, p.coeff i ∈ ({0, -1, 1} : Set R) }

/-- The statement to be proven -/
theorem existence_of_restricted_polynomial (n : ℕ) :
  ∃ (P : RestrictedPolynomial ℤ),
    (P.val ≠ 0) ∧
    (P.val.degree ≤ 2^n) ∧
    ((X - 1)^n ∣ P.val) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_restricted_polynomial_l782_78251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l782_78260

noncomputable def f (a b x : ℝ) := a^x + b

theorem function_properties :
  ∀ a b : ℝ,
  a > 0 →
  a ≠ 1 →
  f a b 0 = 2 →
  (∃ x y : ℝ, x ∈ Set.Icc 2 3 ∧ y ∈ Set.Icc 2 3 ∧
    (∀ z ∈ Set.Icc 2 3, 
      f a b x ≥ f a b z ∧ 
      f a b y ≤ f a b z) ∧
    f a b x - f a b y = a^2 / 2) →
  b = 1 ∧ (a = 1/2 ∨ a = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l782_78260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l782_78206

/-- Given real numbers x and y, and vectors a, b, and c in ℝ², 
    where a is perpendicular to c and b is parallel to c,
    prove that the magnitude of a + b is √10. -/
theorem vector_sum_magnitude (x y : ℝ) 
  (a b c : ℝ × ℝ) 
  (ha : a = (x, 1)) 
  (hb : b = (1, y)) 
  (hc : c = (2, -4)) 
  (perp : a.1 * c.1 + a.2 * c.2 = 0)
  (para : ∃ (k : ℝ), b.1 = k * c.1 ∧ b.2 = k * c.2) :
  ‖(a.1 + b.1, a.2 + b.2)‖ = Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l782_78206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l782_78201

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The ratio of side lengths in the triangle -/
def side_ratio (t : Triangle) : Prop :=
  ∃ (k : ℝ), t.a = 4 * k ∧ t.b = 5 * k ∧ t.c = 6 * k

/-- The theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) (h : side_ratio t) :
  Real.sin t.A + Real.sin t.C = 2 * Real.sin t.B ∧
  Real.cos t.C = 1 / 8 ∧
  3 * Real.sin t.A = 8 * Real.sin (2 * t.C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l782_78201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_diagonals_not_implies_rhombus_l782_78291

/-- A quadrilateral -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Diagonals of a quadrilateral -/
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (q.vertices 0, q.vertices 2)

/-- Perpendicular diagonals -/
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  let (d1, d2) := diagonals q
  (d2.1 - d1.1) * (d2.2 - d1.2) = 0

/-- A rhombus -/
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, i ≠ j → 
    let (xi, yi) := q.vertices i
    let (xj, yj) := q.vertices j
    (xi - xj)^2 + (yi - yj)^2 = 
      let (xk, yk) := q.vertices ((i + 1) % 4)
      (xk - xi)^2 + (yk - yi)^2

/-- The statement "A quadrilateral with perpendicular diagonals is a rhombus" is false -/
theorem perpendicular_diagonals_not_implies_rhombus :
  ¬ (∀ q : Quadrilateral, has_perpendicular_diagonals q → is_rhombus q) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_diagonals_not_implies_rhombus_l782_78291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_for_positive_f_l782_78232

open Real MeasureTheory

-- Define the function f
noncomputable def f (x θ : ℝ) : ℝ := 2 * x^2 * sin θ - 4 * x * (1 - x) * cos θ + 3 * (1 - x)^2

-- State the theorem
theorem theta_range_for_positive_f :
  ∀ θ : ℝ, θ ∈ Set.Icc 0 (2 * π) →
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → f x θ > 0) ↔ θ ∈ Set.Ioo (π / 6) π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_for_positive_f_l782_78232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l782_78202

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point on the ellipse -/
def PointOnEllipse (e : Ellipse) :=
  {p : ℝ × ℝ | (p.1^2 / e.a^2) + (p.2^2 / e.b^2) = 1}

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b^2 / e.a^2))

/-- The left focus of an ellipse -/
noncomputable def leftFocus (e : Ellipse) : ℝ × ℝ :=
  (-e.a * eccentricity e, 0)

theorem ellipse_eccentricity_range (e : Ellipse) (A B : PointOnEllipse e) (F : ℝ × ℝ) (θ : ℝ) :
  F = leftFocus e →
  (B.val.1 = -A.val.1 ∧ B.val.2 = -A.val.2) →
  (A.val.1 - F.1) * (B.val.1 - F.1) + (A.val.2 - F.2) * (B.val.2 - F.2) = 0 →
  θ = Real.arccos ((B.val.1 - A.val.1) / (2 * e.a * eccentricity e)) →
  π/6 ≤ θ ∧ θ ≤ π/3 →
  Real.sqrt (5 - Real.sqrt 13) ≤ eccentricity e ∧ eccentricity e ≤ Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l782_78202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l782_78227

/-- Represents a parabola with equation y² = 8x -/
structure Parabola where
  equation : ∀ x y : ℝ, y^2 = 8*x

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-- The latus rectum of a parabola y² = 8x has equation x = -2 -/
def latus_rectum (p : Parabola) : ℝ → Prop :=
  λ x ↦ x = -2

/-- A focus of a hyperbola x²/a² - y²/b² = 1 has coordinates (c, 0) where c² = a² + b² -/
noncomputable def focus (h : Hyperbola) : ℝ × ℝ :=
  (Real.sqrt (h.a^2 + h.b^2), 0)

/-- The eccentricity of a hyperbola is defined as c/a where c² = a² + b² -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2) / h.a

theorem hyperbola_equation (p : Parabola) (h : Hyperbola) 
  (h_latus : latus_rectum p (focus h).1)
  (h_eccentricity : eccentricity h = 2) :
  ∀ x y : ℝ, x^2 - y^2/3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l782_78227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_two_circles_l782_78285

/-- Two non-intersecting circles in a metric space -/
structure TwoCircles (α : Type*) [MetricSpace α] where
  center1 : α
  center2 : α
  radius1 : ℝ
  radius2 : ℝ
  non_intersecting : dist center1 center2 > radius1 + radius2

/-- The maximum distance between points on two non-intersecting circles
    is the sum of their radii plus the distance between their centers -/
theorem max_distance_two_circles {α : Type*} [MetricSpace α] (c : TwoCircles α) :
  ∃ (x y : α), x ∈ Metric.sphere c.center1 c.radius1 ∧ y ∈ Metric.sphere c.center2 c.radius2 ∧
    ∀ (x' y' : α), x' ∈ Metric.sphere c.center1 c.radius1 → y' ∈ Metric.sphere c.center2 c.radius2 →
      dist x' y' ≤ dist x y ∧
      dist x y = c.radius1 + c.radius2 + dist c.center1 c.center2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_two_circles_l782_78285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squared_radius_theorem_l782_78209

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℚ
  height : ℚ

/-- Represents the configuration of two intersecting cones and a sphere -/
structure ConeSphereProblem where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℚ
  sphereRadius : ℚ

/-- The maximum possible squared radius of a sphere fitting in two intersecting cones -/
noncomputable def maxSquaredRadius (problem : ConeSphereProblem) : ℚ :=
  (24 ^ 2) / (problem.cone1.baseRadius ^ 2 + problem.cone1.height ^ 2)

/-- The theorem stating the maximum squared radius of the sphere -/
theorem max_squared_radius_theorem (problem : ConeSphereProblem) 
  (h1 : problem.cone1 = problem.cone2)
  (h2 : problem.cone1.baseRadius = 4)
  (h3 : problem.cone1.height = 10)
  (h4 : problem.intersectionDistance = 4) :
  maxSquaredRadius problem = 144 / 29 := by
  sorry

#eval (144 : Nat) + 29

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squared_radius_theorem_l782_78209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l782_78217

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 12) :
  (∀ A' M' C' : ℕ, A' + M' + C' = 12 →
    A * M * C + A * M + M * C + C * A ≥ A' * M' * C' + A' * M' + M' * C' + C' * A') ∧
  A * M * C + A * M + M * C + C * A = 112 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l782_78217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flowers_per_row_l782_78249

theorem flowers_per_row 
  (rows : ℕ) 
  (yellow_flowers : ℕ) 
  (green_flowers : ℕ) 
  (red_flowers : ℕ) 
  (h1 : rows = 6) 
  (h2 : yellow_flowers = 12) 
  (h3 : green_flowers = 2 * yellow_flowers) 
  (h4 : red_flowers = 42) : 
  (yellow_flowers + green_flowers + red_flowers) / rows = 13 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flowers_per_row_l782_78249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ratio_values_l782_78276

/-- A geometric sequence with positive terms where a_3, a_5, a_6 form an arithmetic sequence -/
def SpecialGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ r > 0, ∀ n, a (n + 1) = r * a n) ∧
  (2 * a 5 = a 3 + a 6)

/-- The ratio of sums of specific terms in the special geometric sequence -/
noncomputable def SpecialRatio (a : ℕ → ℝ) : ℝ :=
  (a 3 + a 5) / (a 4 + a 6)

/-- The theorem stating the possible values of the special ratio -/
theorem special_ratio_values (a : ℕ → ℝ) (h : SpecialGeometricSequence a) :
  SpecialRatio a = 1 ∨ SpecialRatio a = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ratio_values_l782_78276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_MBA_l782_78289

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define points A and B
def point_A : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (-1, 0)

-- Define a point M on the parabola
def point_M (x y : ℝ) : Prop := parabola x y

-- Define the angle MBA
noncomputable def angle_MBA (x y : ℝ) : ℝ := 
  Real.arctan ((y - 0) / (x - (-1))) - Real.arctan ((0 - 0) / (1 - (-1)))

-- Theorem statement
theorem max_angle_MBA :
  ∃ (θ : ℝ), 
    (∀ (x y : ℝ), point_M x y → angle_MBA x y ≤ θ) ∧ 
    (∃ (x y : ℝ), point_M x y ∧ angle_MBA x y = θ) ∧ 
    θ = π/4 :=
sorry

-- Note: We've defined angle_MBA as a function that calculates the angle
-- given the coordinates of point M. This is a simplification and may not
-- be mathematically accurate in all cases, but it allows the code to compile.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_MBA_l782_78289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_in_interval_l782_78226

-- Define the function f(x) = x^2 - 2^x
noncomputable def f (x : ℝ) : ℝ := x^2 - 2^x

-- State the theorem
theorem exists_zero_in_interval :
  ∃ c ∈ Set.Ioo (-1 : ℝ) 0, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_in_interval_l782_78226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_increase_after_weight_cut_speed_increase_is_10_l782_78234

/-- Proves that the speed increase after cutting weight is 10 mph -/
theorem speed_increase_after_weight_cut (original_speed supercharge_increase final_speed : ℝ) : ℝ :=
  by
  -- Define the given conditions
  have h1 : original_speed = 150 := by sorry
  have h2 : supercharge_increase = 0.3 := by sorry
  have h3 : final_speed = 205 := by sorry

  -- Calculate the speed after supercharging
  let speed_after_supercharge := original_speed * (1 + supercharge_increase)

  -- Calculate the speed increase after cutting weight
  let speed_increase := final_speed - speed_after_supercharge

  -- Return the speed increase
  exact speed_increase

/-- Proves that the calculated speed increase is indeed 10 mph -/
theorem speed_increase_is_10 : 
  speed_increase_after_weight_cut 150 0.3 205 = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_increase_after_weight_cut_speed_increase_is_10_l782_78234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_is_real_l782_78277

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / ((x + 1) * (x - a))

-- Define what it means for f to be an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem range_is_real (a : ℝ) :
  is_odd_function (f a) → Set.range (f a) = Set.univ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_is_real_l782_78277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l782_78286

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => if x ≥ 0 then 2 * x - 1 else 2 * x + 1

-- State the theorem
theorem odd_function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧                   -- f is odd
  (∀ x : ℝ, x ≥ 0 → f x = 2 * x - 1) →         -- f(x) = 2x - 1 for x ≥ 0
  (∀ x : ℝ, x < 0 → f x = 2 * x + 1) ∧         -- f(x) = 2x + 1 for x < 0
  (∀ a : ℝ, f a ≤ 3 → a ≤ 2) :=                -- f(a) ≤ 3 implies a ≤ 2
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l782_78286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l782_78271

noncomputable def f (x : ℝ) : ℝ := ⌊x⌋ * (x - ⌊x⌋)

def g (x : ℝ) : ℝ := x - 1

theorem solution_set (x : ℝ) :
  x ∈ {x : ℝ | 0 ≤ x ∧ x ≤ 2012 ∧ f x ≤ g x} ↔ x ∈ Set.Icc 1 2012 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l782_78271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_min_value_reciprocal_sum_l782_78264

-- Statement B
theorem max_value_sqrt_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 1) :
  Real.sqrt (2 * x) + Real.sqrt y ≤ Real.sqrt 2 :=
by sorry

-- Statement C
theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > -1) (hy : y > 0) (h_sum : x + y = 2) :
  1 / (x + 1) + 4 / y ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_min_value_reciprocal_sum_l782_78264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_y_coordinate_specific_parabola_vertex_y_l782_78295

/-- The y-coordinate of the vertex of a parabola y = ax^2 + bx + c is given by -b^2 / (4a) + c -/
theorem parabola_vertex_y_coordinate (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c
  let vertex_y := -b^2 / (4 * a) + c
  ∀ x, f x ≤ f (-b / (2 * a)) ∧ f (-b / (2 * a)) = vertex_y :=
by
  sorry

/-- The y-coordinate of the vertex of the parabola y = -2x^2 - 16x - 50 is -18 -/
theorem specific_parabola_vertex_y :
  let f : ℝ → ℝ := fun x ↦ -2 * x^2 - 16 * x - 50
  let vertex_y := -(-16)^2 / (4 * (-2)) + (-50)
  vertex_y = -18 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_y_coordinate_specific_parabola_vertex_y_l782_78295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_scaled_rotated_l782_78239

def z₁ : ℂ := -5 + 7*Complex.I
def z₂ : ℂ := 7 - 3*Complex.I

theorem midpoint_scaled_rotated (z₁ z₂ : ℂ) : 
  2*Complex.I * ((z₁ + z₂) / 2) = -4 + 2*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_scaled_rotated_l782_78239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_g_intersection_l782_78270

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x - 4) / (x + 2)

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := -x - 2*a

-- State the theorem
theorem f_properties_and_g_intersection (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ∈ Set.Icc (-2 : ℝ) (-1)) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, ∀ y ∈ Set.Icc (-1 : ℝ) 0, x < y → f x > f y) ∧
  (∀ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, x < y → f x < f y) ∧
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∃ x₂ ∈ Set.Icc 0 1, g a x₂ = f x₁) →
  a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_g_intersection_l782_78270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_transformed_function_l782_78297

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x - b * Real.cos x

theorem symmetry_of_transformed_function (a b : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, f a b (π/4 + x) = f a b (π/4 - x)) →
  (∀ x : ℝ, f a b (3*π/4 - (π + x)) = f a b (3*π/4 - (π - x))) :=
by
  intro h
  intro x
  -- The proof goes here
  sorry

#check symmetry_of_transformed_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_transformed_function_l782_78297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_false_contrapositive_true_l782_78233

-- Define the universe of shapes
variable (Shape : Type)

-- Define predicates for square and parallelogram
variable (isSquare : Shape → Prop)
variable (isParallelogram : Shape → Prop)

-- Define the original statement
def originalStatement (Shape : Type) (isSquare isParallelogram : Shape → Prop) : Prop :=
  ∀ s : Shape, isSquare s → isParallelogram s

-- Define the negation of the statement
def negationStatement (Shape : Type) (isSquare isParallelogram : Shape → Prop) : Prop :=
  ∃ s : Shape, isSquare s ∧ ¬isParallelogram s

-- Define the contrapositive of the statement
def contrapositiveStatement (Shape : Type) (isSquare isParallelogram : Shape → Prop) : Prop :=
  ∀ s : Shape, ¬isParallelogram s → ¬isSquare s

-- Theorem stating that the negation is false and the contrapositive is true
theorem negation_false_contrapositive_true (Shape : Type) (isSquare isParallelogram : Shape → Prop) :
  (¬negationStatement Shape isSquare isParallelogram) ∧ contrapositiveStatement Shape isSquare isParallelogram :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_false_contrapositive_true_l782_78233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_bc_ratio_l782_78211

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  /-- Point A of the quadrilateral -/
  A : ℝ × ℝ
  /-- Point B of the quadrilateral -/
  B : ℝ × ℝ
  /-- Point C of the quadrilateral -/
  C : ℝ × ℝ
  /-- Point D of the quadrilateral -/
  D : ℝ × ℝ
  /-- Point E inside the quadrilateral -/
  E : ℝ × ℝ
  /-- ABCD has right angles at A and C -/
  right_angle_A : (A.1 - B.1) * (A.1 - D.1) + (A.2 - B.2) * (A.2 - D.2) = 0
  right_angle_C : (C.1 - B.1) * (C.1 - D.1) + (C.2 - B.2) * (C.2 - D.2) = 0
  /-- Triangle ABC is similar to triangle BCD -/
  similar_ABC_BCD : ∃ k : ℝ, k > 0 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = k * ((C.1 - B.1)^2 + (C.2 - B.2)^2) ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = k * ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  /-- AB < BC -/
  AB_less_BC : (B.1 - A.1)^2 + (B.2 - A.2)^2 < (C.1 - B.1)^2 + (C.2 - B.2)^2
  /-- Triangle ABC is similar to triangle ABE -/
  similar_ABC_ABE : ∃ k : ℝ, k > 0 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = k * ((E.1 - A.1)^2 + (E.2 - A.2)^2) ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = k * ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  /-- Area of triangle AED is 10 times the area of triangle ABE -/
  area_ratio : 
    abs ((E.1 - A.1) * (D.2 - A.2) - (E.2 - A.2) * (D.1 - A.1)) = 
    10 * abs ((E.1 - A.1) * (B.2 - A.2) - (E.2 - A.2) * (B.1 - A.1))

/-- The main theorem to be proved -/
theorem ab_bc_ratio (q : SpecialQuadrilateral) : 
  ((q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2) / ((q.C.1 - q.B.1)^2 + (q.C.2 - q.B.2)^2) = (2 + Real.sqrt 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_bc_ratio_l782_78211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jisha_walking_distance_l782_78200

/-- Proves that given the walking conditions, Jisha walked 18 miles on the first day --/
theorem jisha_walking_distance 
  (hours_day1 : ℕ) 
  (speed_day1 : ℕ) 
  (speed_increase : ℕ) 
  (total_distance : ℕ) : 
  speed_day1 = 3 →
  speed_increase = 1 →
  total_distance = 62 →
  hours_day1 * speed_day1 + 
  (hours_day1 - 1) * (speed_day1 + speed_increase) + 
  hours_day1 * (speed_day1 + speed_increase) = total_distance →
  hours_day1 * speed_day1 = 18 := by
  intro h1 h2 h3 h4
  sorry

#check jisha_walking_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jisha_walking_distance_l782_78200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_problem_l782_78221

theorem complex_sum_problem (a b c d e f : ℂ) : 
  b = 2 → 
  e = -2*a - c → 
  (a + b*Complex.I) + (c + d*Complex.I) + (e + f*Complex.I) = 3 - 2*Complex.I → 
  d + f = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_problem_l782_78221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l782_78220

noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

theorem min_value_of_f :
  ∀ x y : ℝ, 1/3 ≤ x ∧ x ≤ 3/5 ∧ 1/4 ≤ y ∧ y ≤ 1/2 →
  f x y ≥ 60/169 ∧ ∃ x₀ y₀ : ℝ, 1/3 ≤ x₀ ∧ x₀ ≤ 3/5 ∧ 1/4 ≤ y₀ ∧ y₀ ≤ 1/2 ∧ f x₀ y₀ = 60/169 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l782_78220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_of_complex_sum_l782_78265

/-- Given complex numbers z and u, if their sum is 4/5 + 3/5i, 
    then the tangent of the sum of their arguments is 24/7 -/
theorem tan_sum_of_complex_sum (α β : ℝ) (z u : ℂ) :
  z = Complex.exp (I * α) →
  u = Complex.exp (I * β) →
  z + u = (4/5 : ℂ) + (3/5 : ℂ) * I →
  Real.tan (α + β) = 24/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_of_complex_sum_l782_78265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidecomposable_same_volume_centrally_symmetric_l782_78203

/-- A polyhedron with centrally symmetric faces -/
structure CentrallySymmetricPolyhedron where
  -- Define the structure of a polyhedron with centrally symmetric faces
  -- (Details omitted for brevity)

/-- Equidecomposability relation between two polyhedra -/
def Equidecomposable (p1 p2 : CentrallySymmetricPolyhedron) : Prop :=
  -- Define the equidecomposability relation
  -- (Details omitted for brevity)
  True  -- Placeholder, replace with actual definition

/-- Volume of a polyhedron -/
def Volume (p : CentrallySymmetricPolyhedron) : ℝ :=
  -- Define how to calculate the volume of a polyhedron
  -- (Details omitted for brevity)
  0  -- Placeholder, replace with actual definition

/-- Theorem: Two polyhedra of the same volume with centrally symmetric faces are equidecomposable -/
theorem equidecomposable_same_volume_centrally_symmetric 
  (p1 p2 : CentrallySymmetricPolyhedron) 
  (h : Volume p1 = Volume p2) : 
  Equidecomposable p1 p2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidecomposable_same_volume_centrally_symmetric_l782_78203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mean_score_l782_78267

/-- Calculates the combined mean score of two classes given their individual mean scores and the ratio of students -/
theorem combined_mean_score 
  (morning_mean afternoon_mean morning_students afternoon_students : ℚ) 
  (h1 : morning_mean = 78) 
  (h2 : afternoon_mean = 65) 
  (h3 : morning_students / afternoon_students = 2 / 3) : 
  (morning_mean * morning_students + afternoon_mean * afternoon_students) / (morning_students + afternoon_students) = 70 := by
  sorry

#eval Float.toString ((78 * 2 + 65 * 3) / (2 + 3))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mean_score_l782_78267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janes_chocolate_chips_percentage_l782_78258

/-- Represents the composition of a trail mix -/
structure TrailMix where
  nuts : ℚ
  dried_fruit : ℚ
  chocolate_chips : ℚ
  composition_sum : nuts + dried_fruit + chocolate_chips = 1

/-- Sue's trail mix composition -/
def sue_mix : TrailMix where
  nuts := 3/10
  dried_fruit := 7/10
  chocolate_chips := 0
  composition_sum := by norm_num

/-- Jane's trail mix composition -/
def jane_mix : TrailMix where
  nuts := 6/10
  dried_fruit := 0
  chocolate_chips := 4/10
  composition_sum := by norm_num

/-- The combined mixture of Sue and Jane's trail mix in equal parts -/
noncomputable def combined_mix : TrailMix where
  nuts := (sue_mix.nuts + jane_mix.nuts) / 2
  dried_fruit := (sue_mix.dried_fruit + jane_mix.dried_fruit) / 2
  chocolate_chips := (sue_mix.chocolate_chips + jane_mix.chocolate_chips) / 2
  composition_sum := by sorry

theorem janes_chocolate_chips_percentage :
  jane_mix.chocolate_chips = 4/10 ∧
  combined_mix.nuts = 9/20 ∧
  combined_mix.dried_fruit = 7/20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janes_chocolate_chips_percentage_l782_78258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l782_78278

/-- The time it takes for Pipe A to fill the tank without the leak -/
noncomputable def T : ℝ := sorry

/-- The rate at which the leak empties the tank -/
noncomputable def leak_rate : ℝ := 1 / 6

/-- The rate at which Pipe A fills the tank with the leak present -/
noncomputable def combined_rate : ℝ := 1 / 3

theorem pipe_fill_time :
  (1 / T - leak_rate = combined_rate) →
  T = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l782_78278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_test_probability_l782_78235

theorem disease_test_probability (population : ℕ) 
  (disease_rate : ℚ) (false_positive_rate : ℚ) :
  disease_rate = 1 / 200 →
  false_positive_rate = 5 / 100 →
  let true_positive_rate := 1
  let positive_test_prob := disease_rate * true_positive_rate + 
    (1 - disease_rate) * false_positive_rate
  (disease_rate * true_positive_rate) / positive_test_prob = 20 / 219 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_test_probability_l782_78235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_theorem_l782_78231

noncomputable def m : ℝ × ℝ := (1, 1)
noncomputable def q : ℝ × ℝ := (1, 0)

noncomputable def n : ℝ × ℝ := (0, -1)

noncomputable def p (A : ℝ) : ℝ × ℝ := (2 * Real.sin A, 4 * (Real.cos (A / 2))^2)

theorem vector_magnitude_theorem (A : ℝ) :
  (m.1 * n.1 + m.2 * n.2 = -1) →
  (Real.cos (3 * Real.pi / 4) * Real.sqrt ((m.1^2 + m.2^2) * (n.1^2 + n.2^2)) = m.1 * n.1 + m.2 * n.2) →
  (q.1 * n.1 + q.2 * n.2 = 0) →
  Real.sqrt ((2 * n.1 + (p A).1)^2 + (2 * n.2 + (p A).2)^2) = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_theorem_l782_78231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_layer_pyramid_blocks_l782_78247

/-- Calculates the number of blocks in a pyramid layer given the number of blocks in the layer above it -/
def blocksInLayer (blocksAbove : ℕ) : ℕ := 5 * blocksAbove

/-- Calculates the total number of blocks in a pyramid with a given number of layers -/
def totalBlocksInPyramid (layers : ℕ) (topLayerBlocks : ℕ) : ℕ :=
  (List.range layers).foldl (fun acc _ => acc + blocksInLayer acc) topLayerBlocks

/-- Theorem stating that an 8-layer pyramid with 3 blocks in the top layer has 312093 total blocks -/
theorem eight_layer_pyramid_blocks :
  totalBlocksInPyramid 8 3 = 312093 := by
  sorry

#eval totalBlocksInPyramid 8 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_layer_pyramid_blocks_l782_78247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_in_election_l782_78240

def election_votes (total_votes : ℕ) : Prop :=
  ∃ (votes_A votes_B votes_C votes_D : ℕ),
    votes_A + votes_B + votes_C + votes_D = total_votes ∧
    votes_A = (22 * total_votes) / 100 ∧
    votes_B = (32 * total_votes) / 100 ∧
    votes_C = (25 * total_votes) / 100 ∧
    votes_D = total_votes - votes_A - votes_B - votes_C ∧
    votes_B - votes_C = 12000

theorem total_votes_in_election :
  ∃ total_votes : ℕ, election_votes total_votes ∧ total_votes = 171429 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_in_election_l782_78240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_example_l782_78210

/-- Calculates the compound interest for a given principal, rate, time, and compounding frequency. -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time) - principal

/-- The compound interest on $2500 for 6 years at 25% per annum, compounded annually, is approximately $9707.03. -/
theorem compound_interest_example : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |compound_interest 2500 0.25 6 1 - 9707.03| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_example_l782_78210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_C_is_four_years_l782_78288

/-- Represents the loan details and interest calculation -/
structure LoanDetails where
  principal_B : ℚ
  principal_C : ℚ
  time_B : ℚ
  rate : ℚ
  total_interest : ℚ

/-- Calculates the time for which A lent money to C -/
def calculate_time_C (loan : LoanDetails) : ℚ :=
  let interest_B := loan.principal_B * loan.rate * loan.time_B
  let interest_C := loan.total_interest - interest_B
  interest_C / (loan.principal_C * loan.rate)

/-- Theorem stating that the time A lent money to C is 4 years -/
theorem time_C_is_four_years (loan : LoanDetails) 
  (h1 : loan.principal_B = 5000)
  (h2 : loan.principal_C = 3000)
  (h3 : loan.time_B = 2)
  (h4 : loan.rate = 8/100)
  (h5 : loan.total_interest = 1760) :
  calculate_time_C loan = 4 := by
  sorry

def main : IO Unit := do
  let loan : LoanDetails := {
    principal_B := 5000,
    principal_C := 3000,
    time_B := 2,
    rate := 8/100,
    total_interest := 1760
  }
  IO.println s!"Time C: {calculate_time_C loan}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_C_is_four_years_l782_78288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_unit_interval_l782_78256

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem f_increasing_on_unit_interval :
  StrictMonoOn f (Set.Ioo 0 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_unit_interval_l782_78256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subinterval_fraction_existence_l782_78272

theorem subinterval_fraction_existence (p q : ℕ) (hp : p > 0) (hq : q > 0) :
  ∀ k : ℕ, k ∈ Finset.range (p + q - 1) \ {0} →
    (∃! i : ℕ, i ∈ Finset.range p \ {0} ∧ k / (p + q : ℚ) ≤ i / p ∧ i / p < (k + 1) / (p + q : ℚ)) ∨
    (∃! j : ℕ, j ∈ Finset.range q \ {0} ∧ k / (p + q : ℚ) ≤ j / q ∧ j / q < (k + 1) / (p + q : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subinterval_fraction_existence_l782_78272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_expression_l782_78212

-- Define the expression as a noncomputable function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) / (x - 1)

-- State the theorem
theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, f x = y) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

#check meaningful_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_expression_l782_78212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l782_78241

noncomputable def binomial_expansion (x : ℝ) : ℝ := (x + 2 / Real.sqrt x) ^ 6

theorem constant_term_of_expansion :
  ∃ (k : ℕ), k ≤ 6 ∧ 
    Nat.choose 6 k * 2^k = 240 ∧
    ∀ (j : ℕ), j ≤ 6 → j ≠ k → 
      (Nat.choose 6 j * 2^j : ℝ) * x^(6 - 3/2 * ↑j) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l782_78241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_currency_conversion_l782_78293

/-- Represents Chinese currency in yuan -/
structure ChineseCurrency where
  yuan : ℚ
  jiao : ℚ
  fen : ℚ

/-- Conversion function from ChineseCurrency to yuan -/
def toYuan (c : ChineseCurrency) : ℚ :=
  c.yuan + c.jiao / 10 + c.fen / 100

/-- Theorem stating that 2 yuan 3 jiao 4 fen equals 2.34 yuan -/
theorem currency_conversion :
  toYuan ⟨2, 3, 4⟩ = 2.34 := by
  -- Unfold the definition of toYuan
  unfold toYuan
  -- Simplify the arithmetic
  simp [add_assoc, div_eq_mul_inv]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_currency_conversion_l782_78293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_isosceles_triangle_angle_l782_78204

/-- Proves that in a right isosceles triangle where the square of the hypotenuse
    is equal to twice the product of the leg lengths, one of the acute angles measures 45°. -/
theorem right_isosceles_triangle_angle (a : ℝ) (h : a > 0) :
  let c := Real.sqrt (2 * a^2)
  (c^2 = 2 * a^2) →
  ∃ θ : ℝ, θ * π / 180 = π / 4 ∧ 
    Real.sin (θ * π / 180) = a / c ∧
    Real.cos (θ * π / 180) = a / c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_isosceles_triangle_angle_l782_78204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_triangle_l782_78296

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3) - Real.cos (2 * x + Real.pi / 6) - Real.sqrt 3 * Real.cos (2 * x)

theorem max_perimeter_triangle (A B C : ℝ) : 
  0 < B ∧ B < Real.pi / 2 →  -- B is an acute angle
  f B = Real.sqrt 3 →
  ∃ (a b c : ℝ), 
    a + b + c ≤ 3 * Real.sqrt 3 ∧  -- Max perimeter
    0 < a ∧ 0 < b ∧ 0 < c ∧  -- Positive side lengths
    c = Real.sqrt 3 ∧  -- AC = √3
    Real.cos a * Real.cos c + Real.sin a * Real.sin c = Real.cos B ∧  -- Law of cosines
    Real.cos b * Real.cos c + Real.sin b * Real.sin c = Real.cos A ∧
    Real.cos a * Real.cos b + Real.sin a * Real.sin b = Real.cos C ∧
    A + B + C = Real.pi  -- Angle sum in a triangle
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_triangle_l782_78296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_condition_l782_78216

/-- Predicate to check if a point is within a square of side length s -/
def is_in_square (s : ℝ) (p : ℝ × ℝ) : Prop :=
  0 ≤ p.1 ∧ p.1 ≤ s ∧ 0 ≤ p.2 ∧ p.2 ≤ s

/-- Predicate to check if three points form an equilateral triangle -/
def is_equilateral_triangle (p q r : ℝ × ℝ) : Prop :=
  dist p q = dist q r ∧ dist q r = dist r p

/-- Given two squares with side lengths a and b, this theorem states the condition
    for the existence of points M such that for any point P in the first square,
    there exists a point Q in the second square forming an equilateral triangle MPQ. -/
theorem equilateral_triangle_condition (a b : ℝ) :
  (∃ M : ℝ × ℝ, ∀ P : ℝ × ℝ, is_in_square a P →
    ∃ Q : ℝ × ℝ, is_in_square b Q ∧ is_equilateral_triangle M P Q) ↔
  b ≥ (a / 2) * (Real.sqrt 3 + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_condition_l782_78216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_two_l782_78243

-- Define the functions
noncomputable def f_A (x : ℝ) : ℝ := x + 1/x
noncomputable def f_B (x : ℝ) : ℝ := x + 1/x
noncomputable def f_C (x : ℝ) : ℝ := x + 4/x
noncomputable def f_D (x : ℝ) : ℝ := Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2)

-- State the theorem
theorem min_value_is_two :
  (∃ (x : ℝ), x > 0 ∧ f_B x = 2) ∧
  (∀ (x : ℝ), x > 0 → f_B x ≥ 2) ∧
  (¬∃ (x : ℝ), f_A x < 2) ∧
  (∀ (x : ℝ), x > 0 → f_C x > 2) ∧
  (∀ (x : ℝ), f_D x > 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_two_l782_78243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrow_arrangement_probability_l782_78259

def word : String := "ARROW"

def total_arrangements (w : String) : ℕ :=
  Nat.factorial w.length / (Nat.factorial 2)

def favorable_arrangements (w : String) : ℕ :=
  Nat.factorial (w.length - 1)

theorem arrow_arrangement_probability :
  (favorable_arrangements word : ℚ) / (total_arrangements word : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrow_arrangement_probability_l782_78259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_intersection_l782_78279

/-- Given that the terminal side of angle α intersects the unit circle at (1/2, y),
    prove that sin(π/2 + α) = 1/2 -/
theorem unit_circle_intersection (α : ℝ) :
  (∃ y : ℝ, (1/2)^2 + y^2 = 1 ∧ 
   ∃ θ : ℝ, Real.cos θ = 1/2 ∧ Real.sin θ = y ∧ θ = α + 2 * π * Int.floor (α / (2 * π)))
  → Real.sin (π/2 + α) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_intersection_l782_78279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_midpoint_implies_k_l782_78208

-- Define the line
def line (k : ℝ) (x : ℝ) : ℝ := k * x - 2

-- Define the parabola (marked as noncomputable due to sqrt)
noncomputable def parabola (x : ℝ) : ℝ := Real.sqrt (8 * x)

-- Define the condition for intersection
def intersects (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  line k x₁ = parabola x₁ ∧ 
  line k x₂ = parabola x₂

-- Define the midpoint condition
def midpoint_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  line k x₁ = parabola x₁ ∧ 
  line k x₂ = parabola x₂ ∧ 
  (x₁ + x₂) / 2 = 2

-- Theorem statement
theorem intersection_midpoint_implies_k (k : ℝ) :
  intersects k → midpoint_condition k → k > -1 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_midpoint_implies_k_l782_78208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_slope_l782_78246

/-- The Point₂ type represents a point in 2D space. -/
def Point₂ := ℝ × ℝ

/-- The Line₂ type represents a line in 2D space. -/
structure Line₂ where
  equation : ℝ → ℝ → Prop
  slope : ℝ
  contains : Point₂ → Prop

/-- area_triangle calculates the area of a triangle given its three vertices. -/
noncomputable def area_triangle (A B C : Point₂) : ℝ := sorry

/-- Given two lines l₁ and l₂, and a point A, we define a triangle ABC and prove its properties. -/
theorem triangle_abc_slope (l₁ l₂ l₃ : Line₂) (A B C : Point₂) 
  (h1 : l₁.equation = fun x y ↦ 3*x - 2*y = 2)
  (h2 : A = (-2, -3))
  (h3 : l₁.contains A)
  (h4 : l₂.equation = fun x y ↦ y = 2)
  (h5 : l₂.contains B)
  (h6 : l₁.contains B)
  (h7 : l₃.slope > 0)
  (h8 : l₃.contains A)
  (h9 : l₃.contains C)
  (h10 : l₂.contains C)
  (h11 : area_triangle A B C = 6) : l₃.slope = 25/32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_slope_l782_78246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_shape_area_ratio_l782_78253

theorem rectangle_shape_area_ratio : 
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := 8
  let semicircle_area : ℝ := π * (rectangle_length^2 + rectangle_width^2) / 4
  let corner_circle_radius : ℝ := rectangle_width / 2
  let corner_circles_area : ℝ := 4 * π * corner_circle_radius^2
  let total_area : ℝ := semicircle_area + corner_circles_area
  ∃ ε > 0, |total_area / semicircle_area - 2.23| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_shape_area_ratio_l782_78253
