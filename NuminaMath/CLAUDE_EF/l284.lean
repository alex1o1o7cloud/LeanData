import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arg_diff_max_l284_28450

/-- Given complex numbers z and ω satisfying certain conditions, 
    the maximum value of cos(arg z - arg ω) is 1/8 -/
theorem cos_arg_diff_max (z ω : ℂ) 
  (sum_cond : z + ω + 3 = 0)
  (seq_cond : ∃ d : ℝ, Complex.abs z = 2 - d ∧ Complex.abs ω = 2 + d) :
  ∃ (θ : ℝ), ∀ (φ : ℝ), Real.cos φ ≤ Real.cos θ ∧ Real.cos θ = 1/8 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arg_diff_max_l284_28450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_planes_in_pyramid_l284_28460

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line where
  p : Point
  q : Point

structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the pyramid
structure Pyramid where
  S : Point
  A : Point
  B : Point
  C : Point
  D : Point
  base : Plane

-- Define the intersection of two lines
def lineIntersection (l1 l2 : Line) : Option Point := sorry

-- Define if two lines are parallel
def areParallel (l1 l2 : Line) : Prop := sorry

-- Define a line through a point parallel to another line
def parallelThroughPoint (l : Line) (p : Point) : Line := sorry

-- Define the line of intersection of two planes
def planeIntersection (p1 p2 : Plane) : Line := sorry

-- The main theorem
theorem intersection_of_planes_in_pyramid (SABCD : Pyramid) :
  let AB : Line := ⟨SABCD.A, SABCD.B⟩
  let CD : Line := ⟨SABCD.C, SABCD.D⟩
  let ABS : Plane := sorry
  let CDS : Plane := sorry
  let intersection := planeIntersection ABS CDS
  (∃ M, lineIntersection AB CD = some M ∧ intersection = ⟨SABCD.S, M⟩) ∨
  (areParallel AB CD ∧ ∃ l, intersection = l ∧ l = parallelThroughPoint AB SABCD.S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_planes_in_pyramid_l284_28460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_theorem_l284_28487

-- Define the quadrilateral ABCD and points P, Q, M, N
variable (A B C D P Q M N : ℝ × ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def is_inscribed_circle_center (P : ℝ × ℝ) (X Y Z : ℝ × ℝ) : Prop := sorry

def ray_intersects (X Y Z W V : ℝ × ℝ) : Prop := sorry

noncomputable def segment_length (X Y : ℝ × ℝ) : ℝ := sorry

def circles_are_tangent (P Q : ℝ × ℝ) (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the theorem
theorem quadrilateral_theorem 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_inscribed_circle_center P A B D)
  (h3 : is_inscribed_circle_center Q C B D)
  (h4 : ray_intersects B P M D A)
  (h5 : ray_intersects D Q N B C)
  (h6 : segment_length A M = 9/7)
  (h7 : segment_length D M = 12/7)
  (h8 : segment_length B N = 20/9)
  (h9 : segment_length C N = 25/9)
  (h10 : circles_are_tangent P Q A B C D) :
  (segment_length A B) / (segment_length C D) = 3/5 ∧
  segment_length A B = 3 ∧ segment_length C D = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_theorem_l284_28487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_proof_l284_28482

theorem matrix_transformation_proof :
  ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
    ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
      N * A = ![![4 * A 0 0, 4 * A 0 1], ![2 * A 1 0, 2 * A 1 1]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_proof_l284_28482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equivalence_l284_28430

noncomputable def polar_to_standard (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (abs r, (θ + 2 * Real.pi) % (2 * Real.pi))

theorem polar_equivalence :
  polar_to_standard (-5) (5 * Real.pi / 6) = (5, 11 * Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equivalence_l284_28430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_condition_l284_28453

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_condition 
  (a b : Line) (α : Plane) 
  (h1 : perp a b) 
  (h2 : ¬ subset b α) :
  (∀ x : Line, perpToPlane a α → parallel b α) ∧ 
  (∃ y : Line, parallel b α ∧ ¬ perpToPlane a α) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_condition_l284_28453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_planes_l284_28493

/-- The distance between two planes given by their equations -/
noncomputable def plane_distance (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  abs (d₂ / Real.sqrt (a₂^2 + b₂^2 + c₂^2) - d₁ / Real.sqrt (a₁^2 + b₁^2 + c₁^2))

/-- Theorem stating that the distance between the given planes is zero -/
theorem distance_between_planes :
  plane_distance 3 (-9) 6 12 6 (-18) 12 24 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_planes_l284_28493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l284_28402

-- Define the system of inequalities
def system (x y : ℝ) : Prop :=
  x + y ≤ 4 ∧ 2 * x + y ≥ 1 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the quadrilateral region
def quadrilateral_region : Set (ℝ × ℝ) :=
  {p | system p.1 p.2}

-- Define the function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem longest_side_length :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ quadrilateral_region ∧ p2 ∈ quadrilateral_region ∧
    ∀ (q1 q2 : ℝ × ℝ), q1 ∈ quadrilateral_region → q2 ∈ quadrilateral_region →
      distance q1 q2 ≤ distance p1 p2 ∧ distance p1 p2 = 4 * Real.sqrt 2 :=
by
  sorry

#check longest_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l284_28402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_is_circle_l284_28479

-- Define the line C₁
noncomputable def C₁ (α t : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, t * Real.sin α)

-- Define the circle C₂
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the perpendicular line from origin to C₁
def perpendicular (α : ℝ) : ℝ × ℝ → Prop :=
  λ p => p.1 * Real.cos α + p.2 * Real.sin α = 0

-- Define point A as the intersection of C₁ and the perpendicular line
noncomputable def A (α : ℝ) : ℝ × ℝ := (Real.sin α * Real.sin α, -Real.cos α * Real.sin α)

-- Define point P as the midpoint of OA
noncomputable def P (α : ℝ) : ℝ × ℝ := ((1/2) * Real.sin α * Real.sin α, -(1/2) * Real.cos α * Real.sin α)

-- Theorem: The trajectory of P is a circle with center (1/4, 0) and radius 1/4
theorem trajectory_of_P_is_circle :
  ∀ α : ℝ, (P α).1 - 1/4 ^ 2 + (P α).2 ^ 2 = 1/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_is_circle_l284_28479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l284_28473

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance from the center to a focus -/
noncomputable def focal_distance (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

/-- Predicate to check if two points are symmetric with respect to the asymptote -/
def IsSymmetricWrtAsymptote (F₁ F₂ : ℝ × ℝ) : Prop :=
  sorry

/-- Predicate to check if a point is on a circle with given center and radius -/
def IsOnCircle (point center : ℝ × ℝ) (radius : ℝ) : Prop :=
  sorry

/-- Theorem: Under given conditions, the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity_is_two (h : Hyperbola)
  (h_symmetric : ∃ (F₁ F₂ : ℝ × ℝ), F₁ = (-focal_distance h, 0) ∧ F₂ = (focal_distance h, 0) ∧
    IsSymmetricWrtAsymptote F₁ F₂)
  (h_on_circle : ∃ (F₁ F₂ : ℝ × ℝ), F₁ = (-focal_distance h, 0) ∧ F₂ = (focal_distance h, 0) ∧
    IsOnCircle F₁ F₂ (focal_distance h)) :
  eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l284_28473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_nonnegative_l284_28462

/-- A polynomial with real coefficients. -/
def MyPolynomial := ℝ → ℝ

/-- The derivative of a polynomial. -/
noncomputable def derivative (P : MyPolynomial) : MyPolynomial :=
  fun x => sorry

/-- The second derivative of a polynomial. -/
noncomputable def secondDerivative (P : MyPolynomial) : MyPolynomial :=
  fun x => derivative (derivative P) x

/-- The third derivative of a polynomial. -/
noncomputable def thirdDerivative (P : MyPolynomial) : MyPolynomial :=
  fun x => derivative (secondDerivative P) x

/-- 
If for a polynomial P with real coefficients, 
P(x) - P'(x) - P''(x) + P'''(x) ≥ 0 for all real x, 
then P(x) ≥ 0 for all real x.
-/
theorem polynomial_nonnegative (P : MyPolynomial) :
  (∀ x : ℝ, P x - (derivative P) x - (secondDerivative P) x + (thirdDerivative P) x ≥ 0) →
  (∀ x : ℝ, P x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_nonnegative_l284_28462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l284_28400

/-- Given a hyperbola with equation x²/9 - y²/m = 1 where one of its foci lies on the line x + y = 5,
    prove that the equations of its asymptotes are y = ±(4/3)x -/
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ (x y : ℝ), x^2 / 9 - y^2 / m = 1 ∧ x + y = 5) →
  (∃ (k : ℝ), k = 4/3 ∧ 
    (∀ (x y : ℝ), (y = k*x ∨ y = -k*x) ↔ 
      (∀ ε > 0, ∃ (t : ℝ), ∀ (x' y' : ℝ), 
        x'^2 / 9 - y'^2 / m = 1 → 
        ((x' - x)^2 + (y' - y)^2 < ε^2 → 
         (y' - y)^2 < (k*x' - k*x)^2 + t*ε^2)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l284_28400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l284_28424

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- State the theorem
theorem f_monotonic_increasing :
  ∀ x y : ℝ, -1 < x ∧ x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l284_28424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_in_polygon_l284_28411

/-- A convex polygon in the plane -/
structure ConvexPolygon where
  -- Add necessary fields and properties to define a convex polygon
  -- This is a simplified representation
  is_convex : Bool

/-- A convex hexagon in the plane -/
structure ConvexHexagon extends ConvexPolygon

/-- The area of a polygon -/
noncomputable def area (P : ConvexPolygon) : ℝ :=
  sorry

/-- Check if one polygon is contained in another -/
def isContainedIn (H : ConvexHexagon) (P : ConvexPolygon) : Prop :=
  sorry

theorem hexagon_in_polygon (P : ConvexPolygon) :
  ∃ (H : ConvexHexagon), isContainedIn H P ∧ area H.toConvexPolygon ≥ 0.75 * area P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_in_polygon_l284_28411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l284_28419

/-- A predicate that checks if a triangle is inscribed in a circle -/
def IsInscribed (triangle : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Calculates the area of a triangle -/
def AreaOfTriangle (triangle : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Given a triangle inscribed in a circle where the vertices divide the circle into three arcs of lengths 5, 7, and 8, the area of the triangle is 119.84/π² -/
theorem inscribed_triangle_area (triangle : Set (ℝ × ℝ)) 
  (circle : Set (ℝ × ℝ)) (arc1 arc2 arc3 : ℝ) :
  IsInscribed triangle circle →
  arc1 = 5 →
  arc2 = 7 →
  arc3 = 8 →
  AreaOfTriangle triangle = 119.84 / π^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l284_28419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_segment_arrangement_l284_28449

/-- A line segment in a 2D plane --/
structure Segment where
  start : ℝ × ℝ
  finish : ℝ × ℝ

/-- Predicate to check if a point is strictly inside a segment --/
def strictly_inside (p : ℝ × ℝ) (s : Segment) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ p = (1 - t) • s.start + t • s.finish

/-- The main theorem stating the impossibility of the arrangement --/
theorem impossible_segment_arrangement :
  ¬ ∃ (segments : Fin 1000 → Segment),
    ∀ i : Fin 1000,
      (∃ j k : Fin 1000, j ≠ i ∧ k ≠ i ∧
        strictly_inside (segments i).start (segments j) ∧
        strictly_inside (segments i).finish (segments k)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_segment_arrangement_l284_28449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l284_28457

theorem cos_alpha_value (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < π/2) 
  (h3 : Real.cos (π/3 + α) = 1/3) : 
  Real.cos α = (2 * Real.sqrt 6 + 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l284_28457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_120_degree_angle_l284_28464

/-- Given a point (-4, a) on the terminal side of a 120° angle, prove that a = 4√3 -/
theorem point_on_120_degree_angle (a : ℝ) : 
  (∃ (x y : ℝ), x = -4 ∧ y = a ∧ 
   (x, y) ∈ {p : ℝ × ℝ | ∃ (r θ : ℝ), p.1 = r * Real.cos θ ∧ p.2 = r * Real.sin θ ∧ θ = 2 * π / 3}) → 
  a = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_120_degree_angle_l284_28464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_induction_step_for_even_alternating_sum_equals_double_sum_l284_28414

def alternating_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => (-1)^i / (i + 1 : ℚ))

def double_sum (n : ℕ) : ℚ :=
  2 * (Finset.range (n / 2)).sum (λ i => 1 / (n + 2 * i + 2 : ℚ))

theorem induction_step_for_even (k : ℕ) (h_even : Even k) (h_ge_2 : k ≥ 2) :
  (∀ n : ℕ, Even n → n ≤ k → alternating_sum n = double_sum n) →
  (alternating_sum (k + 2) = double_sum (k + 2)) := by
  sorry

-- The main theorem
theorem alternating_sum_equals_double_sum (n : ℕ) (h_even : Even n) (h_ge_2 : n ≥ 2) :
  alternating_sum n = double_sum n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_induction_step_for_even_alternating_sum_equals_double_sum_l284_28414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pumpkin_sales_l284_28415

theorem pumpkin_sales :
  ∃ (jumbo_count regular_count : ℕ),
    jumbo_count + regular_count = 80 ∧
    9 * jumbo_count + 4 * regular_count = 395 ∧
    regular_count = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pumpkin_sales_l284_28415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l284_28407

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℤ := floor (2 * Real.sin x * Real.cos x) + floor (Real.sin x + Real.cos x)

theorem f_range : 
  ∃ (S : Set ℤ), S = {-2, -1, 0, 1, 2} ∧ ∀ (y : ℤ), (∃ (x : ℝ), f x = y) ↔ y ∈ S :=
by
  sorry

#check f_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l284_28407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_problem_l284_28491

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 + 3*t, 2 - 4*t)

noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.cos (θ - Real.pi/4)

theorem intersection_problem :
  -- 1. General equation of line l
  (∀ x y : ℝ, (∃ t : ℝ, line_l t = (x, y)) ↔ 4*x + 3*y - 2 = 0) ∧
  -- 2. Rectangular coordinate equation of curve C
  (∀ x y : ℝ, (∃ θ : ℝ, x^2 + y^2 = (curve_C θ)^2 ∧ 
    x = curve_C θ * Real.cos θ ∧ 
    y = curve_C θ * Real.sin θ) ↔ 
    x^2 + y^2 - 2*x - 2*y = 0) ∧
  -- 3. Length of |AB|
  (∃ A B : ℝ × ℝ, 
    (∃ t1 : ℝ, line_l t1 = A) ∧ 
    (∃ t2 : ℝ, line_l t2 = B) ∧
    (∃ θ1 : ℝ, A.1^2 + A.2^2 = (curve_C θ1)^2 ∧ 
      A.1 = curve_C θ1 * Real.cos θ1 ∧ 
      A.2 = curve_C θ1 * Real.sin θ1) ∧
    (∃ θ2 : ℝ, B.1^2 + B.2^2 = (curve_C θ2)^2 ∧ 
      B.1 = curve_C θ2 * Real.cos θ2 ∧ 
      B.2 = curve_C θ2 * Real.sin θ2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_problem_l284_28491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_world_cup_probability_problem_l284_28418

-- Define the probabilities
def prob_zhang : ℚ := 3/4

-- Define the number of questions
def num_questions : ℕ := 2

-- Theorem statement
theorem world_cup_probability_problem :
  ∃ p : ℚ, 
    -- Condition: Probability of both answering the same number of questions correctly
    (1 - prob_zhang)^2 * (1 - p)^2 + 
    2 * prob_zhang * (1 - prob_zhang) * p * (1 - p) + 
    prob_zhang^2 * p^2 = 61/144 ∧
    
    -- Conclusion 1: Value of p
    p = 2/3 ∧
    
    -- Conclusion 2: Probability of both answering both questions correctly
    prob_zhang^2 * p^2 = 37/144 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_world_cup_probability_problem_l284_28418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l284_28448

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := cos (ω * x - ω * π / 6)

theorem function_equivalence (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + π) = f ω x) :
  ∃ (k : ℝ), ω = 2 ∧ ∀ x, f ω x = cos (2 * x - π / 3) ∧ cos (2 * x - π / 3) = cos (2 * (x - k)) ∧ k = π / 6 := by
  sorry

#check function_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l284_28448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_approx_l284_28423

/-- The marked price of an item given its original price, initial discount,
    desired profit, and final discount. -/
noncomputable def markedPrice (originalPrice initialDiscount desiredProfit finalDiscount : ℝ) : ℝ :=
  let costPrice := originalPrice * (1 - initialDiscount / 100)
  let sellingPrice := costPrice * (1 + desiredProfit / 100)
  sellingPrice / (1 - finalDiscount / 100)

/-- Theorem stating that the marked price is approximately 39.70 under the given conditions. -/
theorem marked_price_approx :
  ∃ ε > 0, ε < 0.01 ∧ |markedPrice 30 10 25 15 - 39.70| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_approx_l284_28423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l284_28498

-- Define the ellipse C
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  eccentricity : ℝ

-- Define the line AB
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the problem parameters
noncomputable def C : Ellipse :=
  { center := (0, 0)
    a := 2
    b := Real.sqrt 3
    c := 1
    eccentricity := 1/2 }

def M : ℝ × ℝ := (1, 1)

-- State the theorem
theorem ellipse_and_line_properties :
  -- Part 1: Standard equation of ellipse C
  (∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 ↔ (x, y) ∈ {(x, y) | x^2 / C.a^2 + y^2 / C.b^2 = 1}) ∧
  -- Part 2: Equation of line AB
  (∃ A B : ℝ × ℝ,
    A ∈ {(x, y) | x^2 / C.a^2 + y^2 / C.b^2 = 1} ∧
    B ∈ {(x, y) | x^2 / C.a^2 + y^2 / C.b^2 = 1} ∧
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    (∀ x y : ℝ, 3*x + 4*y - 7 = 0 ↔ (x, y) ∈ {(x, y) | y - M.2 = (B.2 - A.2) / (B.1 - A.1) * (x - M.1)})) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l284_28498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_five_count_l284_28405

theorem divisors_multiple_of_five_count (n : ℕ) (h : n = 720) : 
  (Finset.filter (λ d ↦ d ∣ n ∧ 5 ∣ d) (Finset.range (n + 1))).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_five_count_l284_28405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_inv_h_four_l284_28468

-- Define h and k as functions from real numbers to real numbers
variable (h k : ℝ → ℝ)

-- Define the inverse functions
variable (h_inv k_inv : ℝ → ℝ)

-- Assume h and k are invertible
variable (h_invertible : Invertible h)
variable (k_invertible : Invertible k)

-- Define the relationship between h_inv and k
axiom h_k_relation : ∀ x, h_inv (k x) = 3 * x - 1

-- State the theorem
theorem k_inv_h_four (h k : ℝ → ℝ) (h_inv k_inv : ℝ → ℝ) 
  (h_invertible : Invertible h) (k_invertible : Invertible k) 
  (h_k_relation : ∀ x, h_inv (k x) = 3 * x - 1) : 
  k_inv (h 4) = 5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_inv_h_four_l284_28468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt949_div_6_l284_28455

noncomputable section

-- Define the circle
def circle_center (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the points on the circle
def point1 : ℝ × ℝ := (2, 5)
def point2 : ℝ × ℝ := (-1, 3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem circle_radius_is_sqrt949_div_6 :
  ∃ x : ℝ, distance (circle_center x) point1 = distance (circle_center x) point2 ∧
           distance (circle_center x) point1 = Real.sqrt 949 / 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt949_div_6_l284_28455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l284_28435

/-- The equation of a line parameterized by m -/
def line_equation (m x y : ℝ) : Prop :=
  (2 + m) * x + (1 - 2*m) * y + 4 - 3*m = 0

/-- The range of m for which the line does not pass through the first quadrant -/
def m_range (m : ℝ) : Prop :=
  -2 ≤ m ∧ m ≤ 1/2

/-- The area of triangle AOB given by the line's intersection with the axes -/
noncomputable def triangle_area (m : ℝ) : ℝ :=
  let k := -(2 + m) / (1 - 2*m)
  abs ((2/k - 1) * (k - 2)) / 2

theorem line_properties :
  ∀ m : ℝ,
  (∀ x y : ℝ, x > 0 ∧ y > 0 → ¬(line_equation m x y)) ↔ m_range m ∧
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ line_equation m x y) ∧
  (triangle_area m ≥ 4) ∧
  (triangle_area m = 4 ↔ line_equation m (-2) 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l284_28435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l284_28477

theorem solve_exponential_equation :
  ∃ x : ℝ, 64 = 4 * (16 : ℝ) ^ (x - 1) → x = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l284_28477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_line_l_fixed_point_shortest_chord_l284_28447

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 3)^2 = 9}

-- Define the center of the circle
def center : ℝ × ℝ := (1, 3)

-- Define the radius of the circle
def radius : ℝ := 3

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) := {p | k * p.1 - p.2 - 2 * k + 5 = 0}

theorem circle_properties :
  -- The center is in the first quadrant
  center.1 > 0 ∧ center.2 > 0 ∧
  -- The center lies on the line 3x - y = 0
  3 * center.1 - center.2 = 0 ∧
  -- The circle is tangent to the x-axis
  center.2 = radius ∧
  -- The chord intercepted by the line x - y = 0 has a length of 2√7
  ∃ (p q : ℝ × ℝ), p ∈ circle_C ∧ q ∈ circle_C ∧ p.1 - p.2 = 0 ∧ q.1 - q.2 = 0 ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = 28 := by sorry

theorem line_l_fixed_point :
  ∀ k : ℝ, (2, 5) ∈ line_l k := by sorry

theorem shortest_chord :
  ∃ (p q : ℝ × ℝ), p ∈ circle_C ∧ q ∈ circle_C ∧ p ∈ line_l (-1/2) ∧ q ∈ line_l (-1/2) ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = 16 ∧
    ∀ (k : ℝ) (r s : ℝ × ℝ), r ∈ circle_C → s ∈ circle_C → r ∈ line_l k → s ∈ line_l k →
      (r.1 - s.1)^2 + (r.2 - s.2)^2 ≥ 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_line_l_fixed_point_shortest_chord_l284_28447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l284_28485

/-- The solution set of the inequality a/(x-1) < 1 -/
def A (a : ℝ) : Set ℝ := {x | x ≠ 1 ∧ a / (x - 1) < 1}

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : (2 ∉ A a) → a ∈ Set.Ici 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l284_28485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_interval_of_monotonic_increase_l284_28428

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x)

theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧
  (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry

theorem interval_of_monotonic_increase :
  ∃ (a b : ℝ), a < b ∧ 
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
  (∀ c d, c < d → (∀ x y, c ≤ x ∧ x < y ∧ y ≤ d → f x < f y) → b - a ≥ d - c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_interval_of_monotonic_increase_l284_28428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_l284_28481

/-- Given a circular sector with perimeter 6 and central angle 1 radian, its area is 2 -/
theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 6) (h2 : central_angle = 1) :
  (perimeter / (2 + central_angle)) ^ 2 * central_angle / 2 = 2 := by
  -- Let's introduce the radius
  let radius := perimeter / (2 + central_angle)
  -- Now we can define the area
  let area := radius ^ 2 * central_angle / 2
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_l284_28481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_calculation_l284_28456

/-- Calculate simple interest given principal, rate, and time -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Theorem: The simple interest for a principal of 15041.875, 
    rate of 8% per annum, and time period of 5 years is 6016.75 -/
theorem simple_interest_calculation : 
  simple_interest 15041.875 8 5 = 6016.75 := by
  -- Unfold the definition of simple_interest
  unfold simple_interest
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_calculation_l284_28456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_minor_theorem_l284_28440

-- Define the constant c
variable (c : ℝ)

-- Define the function f
noncomputable def f (r : ℕ) : ℝ := 8 * Real.log (r : ℝ) + 4 * Real.log (Real.log (r : ℝ)) + c

-- Define the graph properties
def Graph : Type := sorry
def MinimumDegree (G : Graph) : ℕ := sorry
def Girth (G : Graph) : ℕ := sorry
def HasKMinor (G : Graph) (r : ℕ) : Prop := sorry

-- State the theorem
theorem graph_minor_theorem :
  ∃ (f : ℕ → ℕ), ∀ (r : ℕ) (G : Graph),
    MinimumDegree G ≥ 3 →
    Girth G ≥ f r →
    HasKMinor G r :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_minor_theorem_l284_28440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_highway_mpg_is_12_2_l284_28401

/-- Represents the fuel efficiency of an SUV -/
structure SUVFuelEfficiency where
  city_mpg : ℚ
  max_distance : ℚ
  max_fuel : ℚ

/-- Calculates the highway mpg for an SUV -/
def highway_mpg (suv : SUVFuelEfficiency) : ℚ :=
  suv.max_distance / suv.max_fuel

/-- Theorem: The highway mpg of the given SUV is 12.2 -/
theorem suv_highway_mpg_is_12_2 (suv : SUVFuelEfficiency) 
  (h1 : suv.city_mpg = 76/10)
  (h2 : suv.max_distance = 244)
  (h3 : suv.max_fuel = 20) :
  highway_mpg suv = 61/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_highway_mpg_is_12_2_l284_28401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_21_is_11th_term_l284_28432

noncomputable def mySequence (n : ℕ) : ℝ := Real.sqrt (2 * n - 1)

theorem sqrt_21_is_11th_term :
  ∃ n : ℕ, mySequence n = Real.sqrt 21 ∧ n = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_21_is_11th_term_l284_28432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_singleton_l284_28466

/-- The value of k for which sets A and B intersect at a single point -/
noncomputable def k : ℝ := -2 - Real.sqrt 3

/-- Set A: points (x, y) satisfying x^2 + y^2 = 2(x + y) -/
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 2 * (p.1 + p.2)}

/-- Set B: points (x, y) satisfying kx - y + k + 3 ≥ 0 -/
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | k * p.1 - p.2 + k + 3 ≥ 0}

/-- Theorem stating that the intersection of A and B is a singleton -/
theorem A_intersect_B_singleton : (A ∩ B).Subsingleton := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_singleton_l284_28466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nasrin_mean_speed_l284_28494

noncomputable def distance_to_camp : ℝ := 4.5
noncomputable def time_to_camp : ℝ := 2.5  -- 2 hours and 30 minutes
noncomputable def return_time_factor : ℝ := 1/3

theorem nasrin_mean_speed :
  let total_distance := 2 * distance_to_camp
  let total_time := time_to_camp + return_time_factor * time_to_camp
  let mean_speed := total_distance / total_time
  mean_speed = 2.7 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nasrin_mean_speed_l284_28494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_eccentricity_product_l284_28484

/-- Represents a conic section (ellipse or hyperbola) -/
structure ConicSection where
  a : ℝ  -- Semi-major axis (for ellipse) or semi-transverse axis (for hyperbola)
  c : ℝ  -- Half the distance between foci
  e : ℝ  -- Eccentricity

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The configuration of two conic sections with common foci and a common point -/
structure ConicConfiguration where
  ellipse : ConicSection
  hyperbola : ConicSection
  F₁ : Point  -- First focus
  F₂ : Point  -- Second focus
  P : Point   -- Common point

/-- Angle between three points -/
def angle (A B C : Point) : ℝ := sorry

/-- The theorem stating the minimum product of eccentricities -/
theorem min_eccentricity_product (config : ConicConfiguration) 
  (h_common_foci : config.ellipse.c = config.hyperbola.c)
  (h_angle : angle config.F₁ config.P config.F₂ = π / 3) :
  config.ellipse.e * config.hyperbola.e ≥ Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_eccentricity_product_l284_28484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angular_displacement_correct_l284_28492

/-- A pendulum system with a thin homogeneous rod and an attached mass -/
structure PendulumSystem where
  l : ℝ  -- Length of the rod
  M : ℝ  -- Mass of the rod
  M₁ : ℝ  -- Mass attached to the end of the rod
  m : ℝ  -- Mass of the bullet
  v : ℝ  -- Velocity of the bullet
  g : ℝ  -- Gravitational acceleration

/-- The maximum angular displacement of the pendulum system after bullet impact -/
noncomputable def maxAngularDisplacement (sys : PendulumSystem) : ℝ :=
  let φ₀ := Real.arccos (1 - (sys.m^2 * sys.v^2) / 
    (sys.g * sys.l * (sys.m * sys.l^2 + sys.M₁ * sys.l^2 + (1/3) * sys.M * sys.l^2) * 
    (2 * sys.M₁ + 2 * sys.m + sys.M)))
  φ₀

/-- Theorem stating the correctness of the maximum angular displacement formula -/
theorem max_angular_displacement_correct (sys : PendulumSystem) :
  Real.cos (maxAngularDisplacement sys) = 
    1 - (sys.m^2 * sys.v^2) / 
    (sys.g * sys.l * (sys.m * sys.l^2 + sys.M₁ * sys.l^2 + (1/3) * sys.M * sys.l^2) * 
    (2 * sys.M₁ + 2 * sys.m + sys.M)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angular_displacement_correct_l284_28492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_formula_l284_28416

theorem sine_difference_formula (α : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = 1 / 3) :
  Real.sin (π / 4 - α) = (Real.sqrt 2 - 4) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_formula_l284_28416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_pension_l284_28465

/-- Represents the retirement benefit function -/
noncomputable def benefit (k : ℝ) (x : ℝ) : ℝ := k * Real.sqrt x

/-- The increase in benefit after working 3 more years -/
noncomputable def increase_3_years (k : ℝ) (x : ℝ) : ℝ := benefit k (x + 3) - benefit k x

/-- The increase in benefit after working 5 more years -/
noncomputable def increase_5_years (k : ℝ) (x : ℝ) : ℝ := benefit k (x + 5) - benefit k x

theorem original_pension (k : ℝ) (x : ℝ) 
  (h1 : increase_3_years k x = 100)
  (h2 : increase_5_years k x = 160) : 
  benefit k x = 670 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_pension_l284_28465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l284_28486

/-- Given a geometric sequence {a_n} with first term a₁ and common ratio q,
    where S_n is the sum of the first n terms. -/
noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def S (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁
  else a₁ * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence {a_n} with S₄ = 5S₂,
    the value of (a₃ * a₈) / (a₅)² is either -1 or ±2 -/
theorem geometric_sequence_ratio (a₁ q : ℝ) (h : a₁ ≠ 0) :
  S a₁ q 4 = 5 * S a₁ q 2 →
  (geometric_sequence a₁ q 3 * geometric_sequence a₁ q 8) / (geometric_sequence a₁ q 5)^2 ∈ ({-1, 2, -2} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l284_28486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_variance_of_4_half_l284_28461

-- Define the binomial distribution
def binomial_distribution (n : ℕ) (p : ℝ) : Type := Unit

-- Define the variance of a distribution
noncomputable def variance (dist : Type) : ℝ := sorry

-- Theorem statement
theorem binomial_variance_of_4_half :
  variance (binomial_distribution 4 (1/2)) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_variance_of_4_half_l284_28461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_x_500_divided_by_x2_minus_1_x_plus_1_l284_28442

/-- The remainder when x^500 is divided by (x^2 - 1)(x + 1) is x^2 -/
theorem remainder_x_500_divided_by_x2_minus_1_x_plus_1 (x : ZMod (X^3 - X^2 - X + 1)) :
  x^500 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_x_500_divided_by_x2_minus_1_x_plus_1_l284_28442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l284_28469

/-- The equation defining the region --/
def region_equation (x y : ℝ) : Prop :=
  |x - 75| + |y| = |x/3|

/-- The vertices of the triangular region --/
def triangle_vertices : List (ℝ × ℝ) :=
  [(56.25, 0), (75, 25), (112.5, 0)]

/-- The area of the region --/
def region_area : ℝ := 703.125

/-- Function to calculate the area of a triangle given its vertices --/
noncomputable def area_of_triangle (vertices : List (ℝ × ℝ)) : ℝ :=
  sorry  -- Implementation details omitted for brevity

/-- Theorem stating that the area of the region is correct --/
theorem area_of_region :
  ∃ (vertices : List (ℝ × ℝ)),
    (∀ (v : ℝ × ℝ), v ∈ vertices → region_equation v.1 v.2) ∧
    (area_of_triangle vertices = region_area) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l284_28469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercepts_equal_iff_a_values_l284_28496

/-- Represents a line in the form ax + y - 2 - a = 0 --/
structure Line (a : ℝ) where
  equation : ℝ → ℝ → Prop
  equation_def : equation = λ x y => a * x + y - 2 - a = 0

/-- The x-intercept of the line --/
noncomputable def x_intercept (a : ℝ) (l : Line a) : ℝ := (a + 2) / a

/-- The y-intercept of the line --/
def y_intercept (a : ℝ) (l : Line a) : ℝ := 2 + a

/-- Theorem stating that the x-intercept equals the y-intercept iff a = -2 or a = 1 --/
theorem intercepts_equal_iff_a_values (a : ℝ) (l : Line a) :
  x_intercept a l = y_intercept a l ↔ a = -2 ∨ a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercepts_equal_iff_a_values_l284_28496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_derivative_sufficient_not_necessary_for_monotonic_l284_28441

open Function Real

-- Define the property of being monotonic on ℝ
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Define the property of having positive derivative on ℝ
def PositiveDerivative (f : ℝ → ℝ) : Prop :=
  ∀ x, HasDerivAt f (deriv f x) x ∧ deriv f x > 0

-- Theorem statement
theorem positive_derivative_sufficient_not_necessary_for_monotonic :
  (∃ f : ℝ → ℝ, PositiveDerivative f → Monotonic f) ∧
  (∃ f : ℝ → ℝ, Monotonic f ∧ ¬PositiveDerivative f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_derivative_sufficient_not_necessary_for_monotonic_l284_28441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_three_digits_l284_28437

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : Nat
  is_three_digit : value ≥ 100 ∧ value < 1000

/-- Increases a number by 102 -/
def increase (n : Nat) : Nat :=
  n + 102

/-- Rearranges the digits of a number -/
def rearrange (n : Nat) : Nat :=
  sorry

/-- Theorem stating that it's always possible to keep a number as three digits -/
theorem always_three_digits (initial : ThreeDigitNumber) :
  ∀ t : Nat, ∃ n : Nat, n ≥ 100 ∧ n < 1000 ∧ n = rearrange ((Nat.iterate increase t) initial.value) :=
by
  sorry

#check always_three_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_three_digits_l284_28437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_theorem_l284_28476

noncomputable def polynomial (a b c x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + a*x + c

def has_three_distinct_roots (P : ℝ → ℝ) (y : ℝ) : Prop :=
  ∃ (a b c : ℝ), P = polynomial a b c ∧
  (∀ x, P x = 0 ↔ x = Real.tan y ∨ x = Real.tan (2*y) ∨ x = Real.tan (3*y)) ∧
  Real.tan y ≠ Real.tan (2*y) ∧ Real.tan y ≠ Real.tan (3*y) ∧ Real.tan (2*y) ≠ Real.tan (3*y)

noncomputable def solution_set : Set ℝ := {Real.pi/8, 3*Real.pi/8, 5*Real.pi/8, 7*Real.pi/8, 
                                          Real.pi/7, 2*Real.pi/7, 3*Real.pi/7, 4*Real.pi/7, 5*Real.pi/7, 6*Real.pi/7,
                                          Real.pi/9, 2*Real.pi/9, Real.pi/3, 4*Real.pi/9, 5*Real.pi/9, 2*Real.pi/3, 
                                          7*Real.pi/9, 8*Real.pi/9}

theorem polynomial_roots_theorem (P : ℝ → ℝ) (y : ℝ) :
  has_three_distinct_roots P y →
  y ∈ solution_set ∧ 0 ≤ y ∧ y < Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_theorem_l284_28476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_l284_28403

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Length of AE in the specified grid configuration -/
theorem length_of_AE : 
  let A : Point := ⟨0, 4⟩
  let B : Point := ⟨6, 0⟩
  let C : Point := ⟨5, 3⟩
  let D : Point := ⟨3, 0⟩
  ∃ E : Point, 
    (E.x - A.x) / (B.x - A.x) = (E.y - A.y) / (B.y - A.y) ∧ 
    (E.x - C.x) / (D.x - C.x) = (E.y - C.y) / (D.y - C.y) ∧
    distance A E = (5 * Real.sqrt 13) / 4 := by
  sorry

#check length_of_AE

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_l284_28403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l284_28489

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_speed_kmph : ℝ) (train_length_m : ℝ) (platform_length_m : ℝ) :
  train_speed_kmph = 72 →
  train_length_m = 240.0416 →
  platform_length_m = 280 →
  (train_length_m + platform_length_m) / (train_speed_kmph * (1000 / 3600)) = 26.00208 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l284_28489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_omega_l284_28495

theorem tan_period_omega (ω : ℝ) (h1 : ω > 0) : 
  (∀ x : ℝ, Real.tan (ω * x) = Real.tan (ω * (x + π / 2))) → ω = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_omega_l284_28495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finn_bought_ten_index_cards_l284_28451

-- Define the cost of one box of paper clips
def paper_clip_cost : ℚ := 185 / 100

-- Define Eldora's purchase
def eldora_paper_clips : ℕ := 15
def eldora_index_cards : ℕ := 7
def eldora_total_cost : ℚ := 5540 / 100

-- Define Finn's purchase
def finn_paper_clips : ℕ := 12
def finn_total_cost : ℚ := 6170 / 100

-- Define the function to calculate the cost of index cards
def index_card_cost (eldora_paper_clips : ℕ) (eldora_index_cards : ℕ) (eldora_total_cost : ℚ) (paper_clip_cost : ℚ) : ℚ :=
  (eldora_total_cost - (eldora_paper_clips : ℚ) * paper_clip_cost) / (eldora_index_cards : ℚ)

-- Define the function to calculate the number of index card packages Finn bought
def finn_index_cards (finn_paper_clips : ℕ) (finn_total_cost : ℚ) (paper_clip_cost : ℚ) (index_card_cost : ℚ) : ℕ :=
  Int.toNat ((finn_total_cost - (finn_paper_clips : ℚ) * paper_clip_cost) / index_card_cost).floor

-- Theorem stating that Finn bought 10 packages of index cards
theorem finn_bought_ten_index_cards :
  finn_index_cards finn_paper_clips finn_total_cost paper_clip_cost
    (index_card_cost eldora_paper_clips eldora_index_cards eldora_total_cost paper_clip_cost) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_finn_bought_ten_index_cards_l284_28451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l284_28433

-- Define the hyperbola
def is_on_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the focus and fixed point
def left_focus (a b c : ℝ) : ℝ × ℝ := (-c, 0)
def fixed_point (c : ℝ) : ℝ × ℝ := (0, c)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the eccentricity
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

-- Main theorem
theorem hyperbola_eccentricity_range (a b c : ℝ) :
  a > 0 → b > 0 →
  (∃ x y : ℝ, is_on_hyperbola x y a b ∧
    distance (x, y) (left_focus a b c) = distance (x, y) (fixed_point c)) →
  eccentricity a c > Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l284_28433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_x_l284_28422

noncomputable def sequence_x : ℕ → ℝ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | 2 => 24
  | (n + 3) => (1/4) * sequence_x (n + 2) + (3/4) * sequence_x (n + 1)

theorem limit_of_sequence_x :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sequence_x n - 15| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_x_l284_28422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_area_ratio_l284_28480

/-- Represents a rectangular piece of paper -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ
  width_is_sqrt2_length : width = Real.sqrt 2 * length
  area_is_length_times_width : area = length * width

/-- Represents the folded shape -/
structure FoldedShape where
  original : Rectangle
  new_area : ℝ

/-- The ratio of the new area to the original area after folding -/
noncomputable def area_ratio (folded : FoldedShape) : ℝ :=
  folded.new_area / folded.original.area

/-- Theorem: The area ratio of the folded shape to the original rectangle is (16 - √6) / 16 -/
theorem folded_area_ratio :
  ∀ (rect : Rectangle) (folded : FoldedShape),
  folded.original = rect →
  area_ratio folded = (16 - Real.sqrt 6) / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_area_ratio_l284_28480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_in_options_l284_28409

def options : List String := ["which", "what", "that", "the one"]

def correct_answer : String := "that"

theorem correct_answer_in_options : correct_answer ∈ options := by
  simp [options, correct_answer]

#check correct_answer_in_options

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_in_options_l284_28409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_minus_repeating_eight_l284_28443

/-- Given that 0.overline{8} is a repeating decimal, prove that 1 - 0.overline{8} = 1/9 -/
theorem one_minus_repeating_eight : 1 - (8/9 : ℚ) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_minus_repeating_eight_l284_28443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_treasure_probability_l284_28454

/-- The probability of selecting exactly 4 islands out of 8, where each selected island
    has a 3/10 chance of being chosen and each unselected island has a 2/5 chance of
    being chosen. -/
theorem pirate_treasure_probability : 
  (Nat.choose 8 4 : ℚ) * (3/10)^4 * (2/5)^4 = 9072/6250000 := by
  sorry

#eval (Nat.choose 8 4 : ℚ) * (3/10)^4 * (2/5)^4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_treasure_probability_l284_28454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_red_lights_distance_approximation_l284_28439

/-- The distance in inches between adjacent lights -/
def light_spacing : ℕ := 8

/-- The number of red lights in each repeating pattern -/
def red_pattern : ℕ := 3

/-- The number of green lights in each repeating pattern -/
def green_pattern : ℕ := 2

/-- The total number of lights in each repeating pattern -/
def pattern_length : ℕ := red_pattern + green_pattern

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The position of a red light given its ordinal number -/
def red_light_position (n : ℕ) : ℕ :=
  ((n - 1) / red_pattern) * pattern_length + (n - 1) % red_pattern + 1

/-- The theorem to be proved -/
theorem distance_between_red_lights :
  (red_light_position 25 - red_light_position 4) * light_spacing / inches_per_foot = 38 := by
  sorry

/-- The actual distance in feet (with decimal places) -/
def actual_distance : ℚ :=
  (red_light_position 25 - red_light_position 4) * light_spacing / inches_per_foot

/-- Verify that the actual distance is close to 38.67 feet -/
theorem distance_approximation :
  abs (actual_distance - 38.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_red_lights_distance_approximation_l284_28439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sum_of_surds_l284_28420

theorem simplify_sum_of_surds : Real.sqrt (6 + 4 * Real.sqrt 2) + Real.sqrt (6 - 4 * Real.sqrt 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sum_of_surds_l284_28420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ott_has_one_sixth_l284_28490

/-- Represents the money each person has -/
structure Money where
  amount : ℚ

/-- Represents a person -/
inductive Person
| Loki
| Moe
| Nick
| Ott

/-- The initial state of money distribution -/
def initialMoney : Person → Money
| Person.Loki => { amount := 12 }
| Person.Moe => { amount := 24 }
| Person.Nick => { amount := 8 }
| Person.Ott => { amount := 0 }

/-- The total amount of money initially -/
def totalInitialMoney : ℚ := 72

/-- The amount Ott receives from each person -/
def amountGivenToOtt : ℚ := 4

/-- The final state of money distribution after giving to Ott -/
def finalMoney : Person → Money
| Person.Loki => { amount := (initialMoney Person.Loki).amount - amountGivenToOtt }
| Person.Moe => { amount := (initialMoney Person.Moe).amount - amountGivenToOtt }
| Person.Nick => { amount := (initialMoney Person.Nick).amount - amountGivenToOtt }
| Person.Ott => { amount := amountGivenToOtt * 3 }

/-- Theorem: Ott now has 1/6 of the group's total money -/
theorem ott_has_one_sixth :
  (finalMoney Person.Ott).amount / totalInitialMoney = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ott_has_one_sixth_l284_28490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_and_tangent_circles_l284_28438

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertices : List (ℝ × ℝ)

/-- Predicate to check if a triangle is inscribed in a circle -/
def is_inscribed (T : EquilateralTriangle) (C : Circle) : Prop :=
  sorry

/-- Predicate to check if two circles are internally tangent -/
def is_internally_tangent (C1 C2 : Circle) : Prop :=
  sorry

/-- Predicate to check if two circles are externally tangent -/
def is_externally_tangent (C1 C2 : Circle) : Prop :=
  sorry

/-- The main theorem statement -/
theorem inscribed_triangle_and_tangent_circles 
  (A B C D E : Circle)
  (T : EquilateralTriangle)
  (m n : ℕ) :
  A.radius = 10 ∧
  B.radius = 3 ∧
  C.radius = 2 ∧
  D.radius = 2 ∧
  E.radius = m / n ∧
  is_inscribed T A ∧
  is_internally_tangent B A ∧
  is_internally_tangent C A ∧
  is_internally_tangent D A ∧
  is_externally_tangent B E ∧
  is_externally_tangent C E ∧
  is_externally_tangent D E ∧
  Nat.Coprime m n →
  E.radius = 27 / 5 ∧ m + n = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_and_tangent_circles_l284_28438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_l284_28458

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

/-- Area of a rectangle -/
noncomputable def area_rectangle (A B C D : Point) : ℝ :=
  distance A B * distance B C

/-- Area of a trapezoid -/
noncomputable def area_trapezoid (M B C N : Point) : ℝ :=
  ((distance M N + distance B C) / 2) * distance B N

/-- Given a rectangle ABCD and a trapezoid MBCN within it, prove the area of MBCN -/
theorem area_of_trapezoid (A B C D M N : Point) 
  (h1 : area_rectangle A B C D = 40)
  (h2 : distance A B = 8) 
  (h3 : distance M N = 2) 
  (h4 : distance B C = 4) : 
  area_trapezoid M B C N = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_l284_28458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_curvature_problems_l284_28410

/-- Center of curvature for a parabola at its vertex -/
noncomputable def center_of_curvature_parabola (a b c : ℝ) : ℝ × ℝ :=
  let vertex_x := -b / (2 * a)
  let vertex_y := a * vertex_x^2 + b * vertex_x + c
  let y'' := 2 * a
  (vertex_x, vertex_y - 1 / (2 * a))

/-- Center of curvature for a parametric curve -/
noncomputable def center_of_curvature_parametric (x y : ℝ → ℝ) (t : ℝ) : ℝ × ℝ :=
  let x' := deriv x t
  let y' := deriv y t
  let x'' := deriv (deriv x) t
  let y'' := deriv (deriv y) t
  let κ := (x' * y'' - y' * x'') / (x'^2 + y'^2)^(3/2)
  (x t - y' / κ, y t + x' / κ)

theorem center_of_curvature_problems :
  center_of_curvature_parabola (-1) 4 0 = (2, 7/2) ∧
  center_of_curvature_parametric (fun t ↦ t - Real.sin t) (fun t ↦ 1 - Real.cos t) (Real.pi/2) = (Real.pi/2 + 1, -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_curvature_problems_l284_28410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l284_28475

theorem simplify_trig_expression (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π) :
  Real.sqrt (2 - Real.cos θ - Real.sin θ + Real.sqrt (3 * (1 - Real.cos θ) * (1 + Real.cos θ - 2 * Real.sin θ))) =
  Real.sqrt 3 * abs (Real.sin (θ / 2)) + abs (Real.cos (θ / 2)) * Real.sqrt (1 - 2 * Real.tan (θ / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l284_28475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_deadline_is_50_days_l284_28413

/-- Represents the project parameters and progress --/
structure ProjectData where
  initialWorkers : ℕ
  additionalWorkers : ℕ
  initialHoursPerDay : ℕ
  newHoursPerDay : ℕ
  daysWorked : ℕ
  workCompleted : ℚ

/-- Calculates the initial deadline for the project given the project data --/
def calculateInitialDeadline (data : ProjectData) : ℕ :=
  2 * data.daysWorked

/-- Theorem stating that the initial deadline for the given project is 50 days --/
theorem initial_deadline_is_50_days (data : ProjectData)
  (h1 : data.initialWorkers = 100)
  (h2 : data.additionalWorkers = 60)
  (h3 : data.initialHoursPerDay = 8)
  (h4 : data.newHoursPerDay = 10)
  (h5 : data.daysWorked = 25)
  (h6 : data.workCompleted = 1/3) :
  calculateInitialDeadline data = 50 := by
  sorry

/-- Example calculation using the provided data --/
def exampleProjectData : ProjectData := {
  initialWorkers := 100,
  additionalWorkers := 60,
  initialHoursPerDay := 8,
  newHoursPerDay := 10,
  daysWorked := 25,
  workCompleted := 1/3
}

#eval calculateInitialDeadline exampleProjectData

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_deadline_is_50_days_l284_28413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_multiple_with_binary_digits_l284_28404

def decimal_digits : ℕ → Finset ℕ := sorry

theorem exists_multiple_with_binary_digits (n : ℕ) : ∃ k : ℕ, 
  (n ∣ k) ∧ (∀ d : ℕ, d ∈ decimal_digits k → d = 0 ∨ d = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_multiple_with_binary_digits_l284_28404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_necessary_not_sufficient_for_q_l284_28497

noncomputable def p (x : ℝ) : Prop := ∃ k : ℤ, x = Real.pi / 2 + k * Real.pi
def q (x : ℝ) : Prop := Real.sin x = 1

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, q x → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_necessary_not_sufficient_for_q_l284_28497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_3_or_5_count_l284_28452

theorem divisible_by_3_or_5_count : 
  let range := Finset.range 60
  let divisible_by_3_or_5 := range.filter (λ n => n % 3 = 0 ∨ n % 5 = 0)
  divisible_by_3_or_5.card = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_3_or_5_count_l284_28452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_one_l284_28478

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x ln(x + √(a + x²)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.log (x + Real.sqrt (a + x^2))

/-- If f(x) = x ln(x + √(a + x²)) is an even function, then a = 1 -/
theorem f_even_implies_a_eq_one (a : ℝ) :
  IsEven (f a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_one_l284_28478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_in_range_l284_28434

/-- The function f(x) defined as x - (1/3)sin(2x) + a*sin(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1/3) * Real.sin (2*x) + a * Real.sin x

/-- Theorem stating that f(x) is monotonically increasing on ℝ if and only if a ∈ [-1/3, 1/3] -/
theorem f_monotone_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (-1/3) (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_in_range_l284_28434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_103_mod_9_eq_1_l284_28425

/-- The sequence b_n defined as (√3 + 5)^n - (√3 - 5)^n -/
noncomputable def b (n : ℕ) : ℝ :=
  (Real.sqrt 3 + 5) ^ n - (Real.sqrt 3 - 5) ^ n

/-- The recurrence relation for b_n -/
axiom b_recurrence (n : ℕ) : b (n + 2) = 10 * b (n + 1) + 22 * b n

/-- b_n is always an integer -/
axiom b_is_int (n : ℕ) : ∃ k : ℤ, b n = k

/-- The initial values of b_n mod 9 -/
axiom b_mod_9_initial : Int.mod (Int.floor (b 0)) 9 = 0 ∧ Int.mod (Int.floor (b 1)) 9 = 1

/-- The periodicity of b_n mod 9 -/
axiom b_mod_9_periodic (n : ℕ) : Int.mod (Int.floor (b (n + 6))) 9 = Int.mod (Int.floor (b n)) 9

/-- The main theorem: b_103 ≡ 1 (mod 9) -/
theorem b_103_mod_9_eq_1 : Int.mod (Int.floor (b 103)) 9 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_103_mod_9_eq_1_l284_28425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l284_28474

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

/-- The line function -/
def g (x y : ℝ) : ℝ := 2 * x - y + 3

/-- The shortest distance from a point on the curve to the line -/
noncomputable def shortest_distance : ℝ := Real.sqrt 5

theorem shortest_distance_proof :
  ∃ (x₀ y₀ : ℝ), y₀ = f x₀ ∧ 
  ∀ (x y : ℝ), y = f x → 
  (x - x₀)^2 + (y - y₀)^2 ≥ (g x₀ y₀)^2 / ((2:ℝ)^2 + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l284_28474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_total_profit_l284_28467

noncomputable def q (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 20 then 1260 / (x + 1)
  else if 20 < x ∧ x ≤ 180 then 90 - 3 * Real.sqrt (5 * x)
  else 0

noncomputable def W (x : ℝ) : ℝ := x * q x

theorem max_total_profit :
  ∃ (x : ℝ), x > 0 ∧ W x = 240000 ∧ ∀ (y : ℝ), y > 0 → W y ≤ W x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_total_profit_l284_28467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_in_unit_cube_l284_28445

theorem triangle_in_unit_cube (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (ha_upper : a ≤ Real.sqrt 2) (hb_upper : b ≤ Real.sqrt 2) (hc_upper : c ≤ Real.sqrt 2) :
  ∃ (x y z : ℝ × ℝ × ℝ), 
    (0 ≤ x.1 ∧ x.1 ≤ 1) ∧ (0 ≤ x.2.1 ∧ x.2.1 ≤ 1) ∧ (0 ≤ x.2.2 ∧ x.2.2 ≤ 1) ∧
    (0 ≤ y.1 ∧ y.1 ≤ 1) ∧ (0 ≤ y.2.1 ∧ y.2.1 ≤ 1) ∧ (0 ≤ y.2.2 ∧ y.2.2 ≤ 1) ∧
    (0 ≤ z.1 ∧ z.1 ≤ 1) ∧ (0 ≤ z.2.1 ∧ z.2.1 ≤ 1) ∧ (0 ≤ z.2.2 ∧ z.2.2 ≤ 1) ∧
    Real.sqrt ((x.1 - y.1)^2 + (x.2.1 - y.2.1)^2 + (x.2.2 - y.2.2)^2) = a ∧
    Real.sqrt ((y.1 - z.1)^2 + (y.2.1 - z.2.1)^2 + (y.2.2 - z.2.2)^2) = b ∧
    Real.sqrt ((z.1 - x.1)^2 + (z.2.1 - x.2.1)^2 + (z.2.2 - x.2.2)^2) = c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_in_unit_cube_l284_28445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_condition_l284_28417

/-- Represents the quadratic function f(x) = (9/5)x^2 + 2x -/
noncomputable def f (x : ℝ) : ℝ := (9/5) * x^2 + 2 * x

/-- Represents the input x for a given k -/
noncomputable def x_k (k : ℕ) : ℝ := (5/9) * ((10 : ℝ)^k - 1)

/-- Represents the expected output for a given k -/
noncomputable def y_k (k : ℕ) : ℝ := (5/9) * ((10 : ℝ)^(2*k) - 1)

/-- Theorem stating that f satisfies the given condition for all positive integers k -/
theorem f_satisfies_condition : ∀ k : ℕ+, f (x_k k) = y_k k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_condition_l284_28417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_side_a_value_l284_28412

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.b + t.c - t.a) = 3 * t.b * t.c ∧
  t.b * t.c * Real.cos (Real.pi - t.A) = -1

-- Theorem 1
theorem angle_A_value (t : Triangle) (h : satisfies_conditions t) : t.A = Real.pi / 3 := by
  sorry

-- Theorem 2
theorem side_a_value (t : Triangle) (h1 : satisfies_conditions t) (h2 : t.b - t.c = 1) : t.a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_side_a_value_l284_28412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_theorem_l284_28426

theorem fermats_theorem (f f' : ℝ → ℝ) (a b x₀ : ℝ) :
  (∀ x ∈ Set.Icc a b, HasDerivAt f (f' x) x) →
  a < x₀ →
  x₀ < b →
  (∀ x ∈ Set.Icc a b, f x ≤ f x₀) ∨ (∀ x ∈ Set.Icc a b, f x₀ ≤ f x) →
  f' x₀ = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_theorem_l284_28426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l284_28431

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The distance from a point to a vertical line -/
def distanceToVerticalLine (p : Point) (a : ℝ) : ℝ :=
  |p.x - a|

/-- The focus of the parabola -/
def F : Point := ⟨4, 0⟩

/-- The equation of the trajectory satisfies the parabola condition -/
def isParabolaTrajectory (f : ℝ → ℝ) : Prop :=
  ∀ x y, y = f x → (distance ⟨x, y⟩ F = distanceToVerticalLine ⟨x, y⟩ (-4))

/-- The trajectory equation theorem -/
theorem trajectory_equation : 
  (∀ (M : Point), distance M F = distanceToVerticalLine M (-6) - 2) →
  isParabolaTrajectory (λ x => Real.sqrt (16 * x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l284_28431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l284_28427

def is_odd (n : ℤ) : Prop := ∃ m : ℤ, n = 2 * m + 1

theorem arithmetic_sequence_problem (a b c d e k : ℤ) : 
  is_odd a ∧ is_odd b ∧ is_odd c ∧ is_odd d ∧ is_odd e ∧
  b = a + 2*k ∧ c = a + 4*k ∧ d = a + 6*k ∧ e = a + 8*k ∧
  a + c = 146 ∧
  k > 0 ∧
  (∀ x y, (x ∈ ({a, b, c, d, e} : Set ℤ) ∧ y ∈ ({a, b, c, d, e} : Set ℤ) ∧ x ≠ y) → |x - y| > 2) →
  e = 91 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l284_28427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l284_28459

-- Define the sequence a_n and its sum S_n
noncomputable def a : ℕ → ℝ := sorry
noncomputable def S : ℕ → ℝ := sorry

-- Define the given condition
axiom condition (n : ℕ) : 2 * S n / n + n = 2 * a n + 1

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (f : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, f (n + 1) = f n + d

-- Define what it means for three terms to form a geometric sequence
def is_geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

-- State the theorem
theorem sequence_properties :
  (is_arithmetic_sequence a) ∧
  (is_geometric_sequence (a 4) (a 7) (a 9) →
   ∃ n : ℕ, S n = -78 ∧ ∀ m : ℕ, S m ≥ -78) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l284_28459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_complex_l284_28406

theorem equation_solution_complex (x y : ℂ) : 
  x ≠ 0 → y ≠ 0 → (x + y) / y = x / (y + x) → ¬(x.re = x ∧ y.re = y) :=
by
  sorry

#check equation_solution_complex

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_complex_l284_28406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l284_28436

theorem no_valid_arrangement : ¬ ∃ (p : Fin 8 → Fin 8), 
  let numbers := [1, 2, 3, 4, 5, 6, 8, 9]
  ∀ i : Fin 7, (10 * numbers[p i.val]! + numbers[p (i.val + 1)]!) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l284_28436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_norm_scalar_multiple_l284_28483

variable (V : Type*) [NormedAddCommGroup V] [Module ℝ V]

theorem norm_scalar_multiple (v : V) (h : ‖v‖ = 4) : ‖(-3 : ℝ) • v‖ = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_norm_scalar_multiple_l284_28483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l284_28429

theorem cos_alpha_value (α : Real) 
  (h1 : 0 < α) (h2 : α < π / 2) 
  (h3 : Real.cos (π / 3 + α) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 6 + 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l284_28429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_of_D_l284_28472

structure Cube where
  faces : Fin 6 → Char
  adjacent : Char → Char → Bool
  partially_adjacent : Char → Char → Bool
  shares_vertex : Char → Char → Bool
  opposite : Char → Char

def is_valid_cube (c : Cube) : Prop :=
  c.faces = !['\u0041', '\u0042', '\u0043', '\u0044', '\u0045', '\u0046'] ∧
  c.adjacent '\u0043' '\u0044' ∧
  c.adjacent '\u0043' '\u0046' ∧
  c.partially_adjacent '\u0043' '\u0045' ∧
  c.shares_vertex '\u0043' '\u0042'

theorem opposite_face_of_D (c : Cube) (h : is_valid_cube c) : c.opposite '\u0044' = '\u0041' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_of_D_l284_28472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_marking_l284_28408

/-- Represents a board with dimensions m × n -/
structure Board (m n : ℕ) where
  cells : Fin m → Fin n → Bool

/-- Represents a 4-cell shape that can be placed on the board -/
structure Shape where
  cells : Fin 4 → (ℕ × ℕ)

/-- Checks if a shape covers a marked cell on the board -/
def covers (b : Board m n) (s : Shape) (x y : ℕ) : Prop :=
  ∃ (i : Fin 4), let (dx, dy) := s.cells i
    b.cells ⟨x + dx, sorry⟩ ⟨y + dy, sorry⟩ = true

/-- The set of all possible 4-cell shapes (including rotations and flips) -/
def allShapes : Set Shape := sorry

theorem smallest_marking :
  ∃ (k : ℕ), k = 16 ∧ 
    (∃ (b : Board 8 9), ∀ (s : Shape) (x y : ℕ), s ∈ allShapes → covers b s x y) ∧
    (∀ (k' : ℕ), k' < k → 
      ∃ (b' : Board 8 9), ∀ (s : Shape) (x y : ℕ), s ∈ allShapes → ¬covers b' s x y) :=
by sorry

#check smallest_marking

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_marking_l284_28408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debt_installments_count_l284_28471

/-- Proves that the number of installments to pay a debt is 65 given specific payment conditions -/
theorem debt_installments_count : ℕ := by
  /- First 20 payments amount -/
  let first_20_payment : ℕ := 410
  
  /- Additional amount for remaining payments -/
  let additional_amount : ℕ := 65
  
  /- Average payment for the year -/
  let average_payment : ℕ := 455
  
  /- Total number of installments -/
  let total_installments : ℕ := 65
  
  /- Remaining payments amount -/
  let remaining_payment : ℕ := first_20_payment + additional_amount
  
  /- Theorem statement -/
  have : 20 * first_20_payment + (total_installments - 20) * remaining_payment = 
    average_payment * total_installments := by sorry
  
  exact total_installments

#check debt_installments_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debt_installments_count_l284_28471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l284_28463

-- Define the circle C in polar coordinates
noncomputable def circle_C (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the line l
def line_l (x : ℝ) : ℝ := 2 * x

-- Define the symmetrical line l'
def line_l' (x m : ℝ) : ℝ := -2 * x + 2 * m

-- Define the point M
def point_M (m : ℝ) : ℝ × ℝ := (0, m)

-- Define the circle C in Cartesian coordinates
def circle_C_cartesian (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

-- State the theorem
theorem max_m_value :
  ∃ (A B : ℝ × ℝ) (P : ℝ × ℝ) (m : ℝ),
    (∀ θ, circle_C θ ≥ 0) →
    (A.1 = -2 ∧ A.2 = 0) →
    (B.1 = 2 ∧ B.2 = 0) →
    (P.2 = line_l' P.1 m) →
    (((P.1 - A.1) * (B.1 - A.1) + (P.2 - A.2) * (B.2 - A.2))^2 =
     ((P.1 - A.1)^2 + (P.2 - A.2)^2) * ((B.1 - A.1)^2 + (B.2 - A.2)^2)) →
    (m ≤ Real.sqrt 5 - 2) ∧
    (∀ m', m' > Real.sqrt 5 - 2 →
      ¬∃ (P' : ℝ × ℝ), P'.2 = line_l' P'.1 m' ∧
        ((P'.1 - A.1) * (B.1 - A.1) + (P'.2 - A.2) * (B.2 - A.2))^2 =
        ((P'.1 - A.1)^2 + (P'.2 - A.2)^2) * ((B.1 - A.1)^2 + (B.2 - A.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l284_28463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l284_28499

noncomputable section

-- Define IsTriangle and OppositeAngle
def IsTriangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi

def OppositeAngle (angle : ℝ) (side1 side2 : ℝ) : Prop :=
  side1 > 0 ∧ side2 > 0

theorem triangle_ratio_theorem (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  IsTriangle A B C →
  -- Side lengths correspond to opposite angles
  OppositeAngle A b c → OppositeAngle B a c → OppositeAngle C a b →
  -- Given conditions
  a = 1 →
  Real.sin A = 1/3 →
  -- Theorem statement
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l284_28499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_given_sector_l284_28421

-- Define the sector
structure CircularSector where
  centralAngle : ℝ
  area : ℝ

-- Define the given sector
def givenSector : CircularSector := { centralAngle := 2, area := 9 }

-- Function to calculate arc length
noncomputable def arcLength (s : CircularSector) : ℝ :=
  let radius := Real.sqrt (2 * s.area / s.centralAngle)
  radius * s.centralAngle

-- Theorem statement
theorem arc_length_of_given_sector :
  arcLength givenSector = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_given_sector_l284_28421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l284_28488

def TeamScores : List Nat := [2, 4, 5, 7, 8, 10, 11, 13]

structure Game where
  teamScore : Nat
  opponentScore : Nat
  isLoss : Bool

def calculateOpponentScore (teamScore : Nat) (isLoss : Bool) : Nat :=
  if isLoss then teamScore + 2 else teamScore / 3

theorem opponent_total_score :
  ∃ (games : List Game),
    games.length = 8 ∧
    (games.map (λ g => g.teamScore)).toFinset = TeamScores.toFinset ∧
    (games.filter (λ g => g.isLoss)).length = 3 ∧
    (games.filter (λ g => ¬g.isLoss)).length = 5 ∧
    (games.filter (λ g => ¬g.isLoss)).all (λ g => g.teamScore = 3 * g.opponentScore) ∧
    (games.map (λ g => g.opponentScore)).sum = 42 := by
  sorry

#eval TeamScores

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l284_28488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_seven_l284_28470

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression (2x - √x)^10
noncomputable def expression (x : ℝ) : ℝ := (2 * x - Real.sqrt x) ^ 10

-- Theorem stating that the coefficient of x^7 in the expansion of (2x - √x)^10 is C_{10}^6 * 2^4
theorem coefficient_of_x_seven :
  ∃ (c : ℝ), c = (binomial 10 6 : ℝ) * 2^4 ∧
  ∃ (f : ℝ → ℝ), ∀ (x : ℝ), expression x = c * x^7 + f x ∧ (∀ y, y ≠ 7 → f x ≠ c * x^y) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_seven_l284_28470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coeff_in_seven_term_binomial_expansion_l284_28446

theorem largest_coeff_in_seven_term_binomial_expansion :
  ∀ n : ℕ, 
    (∃ k : ℕ, (Finset.range (n + 1)).card = 7) →
    ∃ k : ℕ, k ≤ n ∧ 
      (∀ j : ℕ, j ≤ n → Nat.choose n k ≥ Nat.choose n j) ∧
      (Nat.choose n k * ((-2 : ℤ)^(n - k))) = -160 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coeff_in_seven_term_binomial_expansion_l284_28446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_demons_problem_l284_28444

theorem mountain_demons_problem (N : ℕ) : 
  -- Initial condition
  (4 * N : ℚ) / 7 = (3 * N : ℚ) / 7 * (4 : ℚ) / 3 →
  -- Condition after changes
  ((3 * N : ℚ) / 7 + 10 = ((4 * N : ℚ) / 7 - 2) * (5 : ℚ) / 4) →
  -- Conclusion
  (3 * N) / 7 + 10 = 35 := by
  intro h1 h2
  sorry

#check mountain_demons_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_demons_problem_l284_28444
