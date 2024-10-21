import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_coefficient_l206_20630

/-- Given vectors a and b in a real vector space, this theorem states that 
    if k*a + (2/3)*b lies on the line passing through points defined by a and b, 
    then k must equal 1/3. -/
theorem line_point_coefficient 
  {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] [SMul ℝ V]
  (a b : V) (k : ℝ) :
  (∃ t : ℝ, k • a + (2/3) • b = a + t • (b - a)) → k = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_coefficient_l206_20630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_theorem_l206_20641

/-- The length of the angle bisector segment inside a triangle --/
noncomputable def angle_bisector_length (a b : ℝ) (φ : ℝ) : ℝ :=
  (2 * a * b * Real.cos (φ / 2)) / (a + b)

/-- Represents a triangle in 2D space --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Checks if a set of points forms a valid triangle with given side lengths and angle --/
def is_triangle (t : Triangle) (a b : ℝ) (φ : ℝ) : Prop :=
  sorry

/-- Checks if a line segment is an angle bisector of the triangle --/
def is_angle_bisector (t : Triangle) (bisector : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Calculates the length of a line segment --/
def segment_length (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The length of the angle bisector segment inside a triangle
    with sides a and b, and angle φ between them, is (2ab cos(φ/2)) / (a + b) --/
theorem angle_bisector_theorem (a b φ : ℝ) 
    (ha : a > 0) (hb : b > 0) (hφ : 0 < φ ∧ φ < π) :
    let l := angle_bisector_length a b φ
    ∃ (t : Triangle) (bisector : Set (ℝ × ℝ)),
      is_triangle t a b φ ∧
      is_angle_bisector t bisector ∧
      segment_length bisector = l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_theorem_l206_20641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_even_function_l206_20635

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x - Real.cos x

theorem min_translation_for_even_function :
  ∃ m : ℝ, m > 0 ∧ 
  (∀ x : ℝ, f (x + m) = f (-x + m)) ∧
  (∀ m' : ℝ, m' > 0 → (∀ x : ℝ, f (x + m') = f (-x + m')) → m' ≥ m) ∧
  m = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_even_function_l206_20635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l206_20609

theorem complex_modulus_problem (a : ℝ) (z : ℂ) : 
  z = a + Complex.I ∧ Complex.abs z = 2 → a = Real.sqrt 3 ∨ a = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l206_20609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_two_identical_triangles_l206_20623

/-- Represents a right-angled triangle with legs of length p and q -/
structure RightTriangle where
  p : ℝ
  q : ℝ
  is_positive : 0 < p ∧ 0 < q

/-- Represents the set of triangles at any given step -/
def TriangleSet : Type := Multiset RightTriangle

/-- The operation of dividing a triangle by its altitude -/
def divide_triangle (t : RightTriangle) : TriangleSet :=
  sorry

/-- Applies a sequence of division operations to a set of triangles -/
def apply_divisions (initial : TriangleSet) (divisions : List Nat) : TriangleSet :=
  sorry

/-- The initial set of four identical right-angled triangles -/
def initial_triangles : TriangleSet :=
  sorry

/-- The main theorem to be proved -/
theorem always_two_identical_triangles 
  (divisions : List Nat) : 
  ∃ (t1 t2 : RightTriangle), 
    t1 ∈ (apply_divisions initial_triangles divisions).toList ∧ 
    t2 ∈ (apply_divisions initial_triangles divisions).toList ∧ 
    t1 ≠ t2 ∧ 
    t1.p = t2.p ∧ 
    t1.q = t2.q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_two_identical_triangles_l206_20623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_ellipse_l206_20643

noncomputable section

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of point S -/
def S : ℝ × ℝ := (0, -1/3)

/-- Definition of point M -/
def M : ℝ × ℝ := (0, 1)

/-- Definition of a line passing through S with slope k -/
def line_through_S (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 1/3

/-- Definition of intersection points A and B -/
def intersection_points (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  line_through_S k A.1 A.2 ∧ line_through_S k B.1 B.2

/-- Definition of a circle with AB as diameter passing through M -/
def circle_through_M (A B : ℝ × ℝ) : Prop :=
  (M.1 - A.1) * (M.1 - B.1) + (M.2 - A.2) * (M.2 - B.2) = 0

/-- The main theorem -/
theorem fixed_point_on_ellipse :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
    intersection_points k A B →
    circle_through_M A B :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_ellipse_l206_20643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chip_arrangement_l206_20692

/-- The number of chips on one side of the equilateral triangle -/
def triangle_side : ℕ → ℕ := sorry

/-- The number of chips on one side of the square -/
def square_side : ℕ → ℕ := sorry

/-- The total number of chips in the arrangement -/
def total_chips : ℕ → ℕ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem chip_arrangement (n : ℕ) :
  (square_side n = triangle_side n - 2) →
  (total_chips n = 3 * (triangle_side n) - 3) →
  (total_chips n = 4 * (square_side n) - 4) →
  total_chips n = 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chip_arrangement_l206_20692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_greater_than_sqrt2_over_4_enclosed_area_not_greater_than_half_l206_20697

-- Define the curve M
def M (x y : ℝ) : Prop := Real.sqrt x + Real.sqrt y = 1 ∧ x ≥ 0 ∧ y ≥ 0 ∧ x * y ≠ 0

-- Theorem for the minimum distance
theorem min_distance_greater_than_sqrt2_over_4 :
  ∀ x y : ℝ, M x y → Real.sqrt (x^2 + y^2) > Real.sqrt 2 / 4 :=
by
  sorry

-- Theorem for the enclosed area
theorem enclosed_area_not_greater_than_half :
  (∫ (x : ℝ) in Set.Icc 0 1, (1 - Real.sqrt x)^2) ≤ 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_greater_than_sqrt2_over_4_enclosed_area_not_greater_than_half_l206_20697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_phi_l206_20602

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem min_abs_phi : 
  ∃ φ_min : ℝ, 
    (∀ φ : ℝ, (∃ k : ℤ, φ = k * π - 5 * π / 6) → |φ_min| ≤ |φ|) ∧ 
    φ_min = π / 6 :=
by
  -- We claim that φ_min = π/6
  let φ_min := π / 6

  -- We'll prove that this φ_min satisfies the conditions
  apply Exists.intro φ_min

  apply And.intro

  -- First part: ∀ φ, (∃ k : ℤ, φ = k * π - 5 * π / 6) → |φ_min| ≤ |φ|
  · intro φ h
    -- We know that φ = k * π - 5 * π / 6 for some integer k
    -- The minimum absolute value occurs when k = 1
    -- So we just need to show that |π/6| ≤ |k * π - 5 * π / 6| for all integers k
    sorry -- This part requires more detailed proof

  -- Second part: φ_min = π / 6
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_phi_l206_20602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_minor_axis_l206_20671

/-- Given an ellipse with center (2, -4), one focus at (2, -6), and one endpoint of the semi-major axis at (2, -1),
    prove that the semi-minor axis is √5. -/
theorem ellipse_semi_minor_axis :
  let center : ℝ × ℝ := (2, -4)
  let focus : ℝ × ℝ := (2, -6)
  let semi_major_endpoint : ℝ × ℝ := (2, -1)
  let c : ℝ := |center.2 - focus.2|
  let a : ℝ := |center.2 - semi_major_endpoint.2|
  let b : ℝ := Real.sqrt (a^2 - c^2)
  b = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_minor_axis_l206_20671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_sum_l206_20675

theorem consecutive_even_sum (a b c : ℕ) : 
  (Even a ∧ b = a + 2 ∧ c = b + 2) →  -- Three consecutive even numbers
  (a * b * c = 960) →                 -- Their product is 960
  (a + b + c = 30) :=                 -- Their sum is 30
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_sum_l206_20675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_not_uses_85169_l206_20658

def horner_polynomial (x : ℝ) : ℝ := 7 * x^3 + 3 * x^2 - 5 * x + 11

def horner_method (x : ℝ) : List ℝ :=
  [7 * x + 3, (7 * x + 3) * x - 5, ((7 * x + 3) * x - 5) * x + 11]

theorem horner_method_not_uses_85169 :
  (85169 : ℝ) ∉ horner_method 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_not_uses_85169_l206_20658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_when_tan_is_two_l206_20689

theorem sin_cos_value_when_tan_is_two (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_when_tan_is_two_l206_20689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l206_20677

theorem sqrt_calculations :
  (∀ (x y : ℝ), x > 0 → y > 0 → (Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y)) →
  (Real.sqrt 18 * Real.sqrt (2/3) - Real.sqrt 3 = Real.sqrt 3) ∧
  (Real.sqrt 28 - Real.sqrt (4/7) + 3 * Real.sqrt 7 = (33 * Real.sqrt 7) / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l206_20677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_2_B_subset_A_condition_l206_20615

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) ≤ 0}

-- Theorem 1: Intersection of A and B when a = 2
theorem intersection_when_a_2 : A 2 ∩ B 2 = Set.Icc 4 5 := by sorry

-- Theorem 2: Condition for B to be a subset of A
theorem B_subset_A_condition (a : ℝ) : B a ⊆ A a ↔ a ∈ Set.Ioo 1 3 ∪ {3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_2_B_subset_A_condition_l206_20615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_2pi_l206_20698

open Set
open MeasureTheory
open Interval

-- Define the integrand
noncomputable def f (x : ℝ) := x^2 * Real.sin x + Real.sqrt (4 - x^2)

-- Define the odd part of the integrand
noncomputable def g (x : ℝ) := x^2 * Real.sin x

-- Define the even part of the integrand
noncomputable def h (x : ℝ) := Real.sqrt (4 - x^2)

theorem integral_equals_2pi :
  (∫ x in (-2)..2, f x) = 2 * Real.pi :=
by
  -- Assume g is an odd function
  have h1 : ∀ x, g (-x) = -g x := by sorry
  -- Assume h represents the upper half of a circle with radius 2
  have h2 : ∀ x ∈ Set.Icc (-2) 2, h x = Real.sqrt (4 - x^2) := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_2pi_l206_20698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_conditions_l206_20653

def is_valid_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → (d = 1 ∨ d = 2)

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_conditions :
  ∀ n : ℕ,
    is_valid_number n ∧ digit_sum n = 12 →
    n ≤ 222222 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_conditions_l206_20653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l206_20670

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {0, 1, 2}
def B : Set Int := {-1, 2}

theorem intersection_A_complement_B : A ∩ (U \ B) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l206_20670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sequences_to_determine_l206_20678

theorem min_sequences_to_determine (a₁ a₂ a₃ a₄ : ℕ) :
  ∃ (b₁₁ b₁₂ b₁₃ b₁₄ b₂₁ b₂₂ b₂₃ b₂₄ : ℕ),
    (∀ (a₁' a₂' a₃' a₄' : ℕ),
      (a₁' * b₁₁ + a₂' * b₁₂ + a₃' * b₁₃ + a₄' * b₁₄ = a₁ * b₁₁ + a₂ * b₁₂ + a₃ * b₁₃ + a₄ * b₁₄ ∧
       a₁' * b₂₁ + a₂' * b₂₂ + a₃' * b₂₃ + a₄' * b₂₄ = a₁ * b₂₁ + a₂ * b₂₂ + a₃ * b₂₃ + a₄ * b₂₄) →
      (a₁' = a₁ ∧ a₂' = a₂ ∧ a₃' = a₃ ∧ a₄' = a₄)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sequences_to_determine_l206_20678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euclid_number_prime_factor_uniqueness_l206_20632

def euclid_number (n : ℕ) : ℕ := (Nat.factorial n) + 1

theorem euclid_number_prime_factor_uniqueness :
  ∀ p : ℕ, Nat.Prime p → 
    ∃! n : ℕ, p ∣ euclid_number n :=
by
  sorry

#check euclid_number_prime_factor_uniqueness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euclid_number_prime_factor_uniqueness_l206_20632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_sum_trapezoid_twice_triangle_triangle_twice_trapezoid_l206_20649

-- Define the semicircle and construction
noncomputable def Semicircle (r : ℝ) : Type :=
  { x : ℝ × ℝ // (x.1)^2 + (x.2)^2 = r^2 ∧ x.2 ≥ 0 }

noncomputable def ParallelLines (r : ℝ) (x : ℝ) : Prop :=
  ∃ (A C D : Semicircle r), 
    A.1 = (-r, 0) ∧ 
    C.1.1 = -r * Real.cos x ∧ C.1.2 = r * Real.sin x ∧
    D.1.1 = 0 ∧ D.1.2 = r * Real.sin x

noncomputable def Trapezoid (r : ℝ) (x : ℝ) : ℝ :=
  r^2 / 2 * (Real.sin x + 2 * Real.sin x * Real.cos x)

noncomputable def Triangle (r : ℝ) (x : ℝ) : ℝ :=
  r^2 * Real.sin x * (1 - Real.cos x)

-- Theorem statements
theorem max_area_sum (r : ℝ) : 
  ∀ x, Trapezoid r x + Triangle r x ≤ 3 * r^2 / 2 := by
  sorry

theorem trapezoid_twice_triangle (r : ℝ) :
  ∃ x, Trapezoid r x = 2 * Triangle r x ↔ x = Real.pi / 3 := by
  sorry

theorem triangle_twice_trapezoid (r : ℝ) :
  ∃ x, Triangle r x = 2 * Trapezoid r x ↔ x = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_sum_trapezoid_twice_triangle_triangle_twice_trapezoid_l206_20649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_negative_range_l206_20601

/-- An even function f with the given properties -/
noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_properties :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1, f x = f (-x)) ∧ 
  f (1 / Real.exp 1) = 0 ∧
  ∀ x ∈ Set.Ioo 0 1, (1 - x^2) * Real.log (1 - x^2) * (deriv f x) > 2 * x * f x :=
sorry

/-- The main theorem -/
theorem f_negative_range :
  {x ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 | f x < 0} = 
  Set.Ioo (-1 : ℝ) (-1/Real.exp 1) ∪ Set.Ioo (1/Real.exp 1) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_negative_range_l206_20601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_triangle_perimeter_l206_20656

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  base_side : ℝ
  height : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The perimeter of a triangle given its three sides -/
def triangle_perimeter (a b c : ℝ) : ℝ := a + b + c

/-- The distance between two points in 3D space -/
noncomputable def distance3D (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

theorem prism_triangle_perimeter (prism : RightPrism) 
  (hbase : prism.base_side = 10)
  (hheight : prism.height = 20) : 
  ∃ (V W X : Point3D),
    triangle_perimeter (distance3D V W) (distance3D W X) (distance3D X V) = 2 * Real.sqrt 425 + 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_triangle_perimeter_l206_20656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_container_volume_and_optimal_angle_l206_20660

/-- The radius of the circular iron sheet -/
noncomputable def R : ℝ := 3 * Real.sqrt 3

/-- The volume of the conical container as a function of height -/
noncomputable def V (h : ℝ) : ℝ := -1/3 * Real.pi * h^3 + 9 * Real.pi * h

/-- The central angle that maximizes the container's volume -/
noncomputable def α : ℝ := (6 - 2 * Real.sqrt 6) / 3 * Real.pi

theorem conical_container_volume_and_optimal_angle :
  (∀ h : ℝ, V h = -1/3 * Real.pi * h^3 + 9 * Real.pi * h) ∧
  (∀ h : ℝ, h > 0 → V h ≤ V 3) ∧
  (2 * Real.pi * R - α * R = 2 * Real.pi * 3 * Real.sqrt 2) := by
  sorry

#check conical_container_volume_and_optimal_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_container_volume_and_optimal_angle_l206_20660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raisin_water_percentage_l206_20622

/-- Calculates the percentage of water in raisins given the initial grape weight,
    final raisin weight, and the percentage of water in grapes. -/
noncomputable def water_percentage_in_raisins (grape_weight : ℝ) (raisin_weight : ℝ) (grape_water_percentage : ℝ) : ℝ :=
  let solid_content := grape_weight * (1 - grape_water_percentage / 100)
  let water_in_raisins := raisin_weight - solid_content
  (water_in_raisins / raisin_weight) * 100

/-- Theorem stating that the percentage of water in raisins is 20%
    given the specific weights and water content of grapes. -/
theorem raisin_water_percentage :
  water_percentage_in_raisins 50 5 92 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_raisin_water_percentage_l206_20622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_Q_satisfies_conditions_l206_20680

-- Define the two intersecting planes
def plane1 (x y z : ℝ) : Prop := x + y + z = 1
def plane2 (x y z : ℝ) : Prop := x + 3*y - z = 2

-- Define line R as the intersection of plane1 and plane2
def lineR (x y z : ℝ) : Prop := plane1 x y z ∧ plane2 x y z

-- Define plane Q
def planeQ (x y z : ℝ) : Prop := x - y + 3*z = 0

-- Define the distance function from a point to a plane
noncomputable def distance_to_plane (A B C D x₀ y₀ z₀ : ℝ) : ℝ :=
  |A*x₀ + B*y₀ + C*z₀ + D| / Real.sqrt (A^2 + B^2 + C^2)

-- Theorem statement
theorem plane_Q_satisfies_conditions :
  (∀ x y z, lineR x y z → planeQ x y z) ∧
  (distance_to_plane 1 (-1) 3 0 2 1 0 = 3 / Real.sqrt 14) ∧
  (1 > 0) ∧
  (Nat.gcd (Nat.gcd (Nat.gcd 1 1) 3) 0 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_Q_satisfies_conditions_l206_20680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_C_matches_illustrated_shape_l206_20614

/-- A shape that can be rotated -/
structure RotatableShape where
  id : Char
  has_180_degree_symmetry : Bool

/-- The illustrated shape that we're comparing against -/
def illustrated_shape : RotatableShape :=
  { id := 'I', has_180_degree_symmetry := true }

/-- The set of given shapes -/
def given_shapes : List RotatableShape :=
  [
    { id := 'A', has_180_degree_symmetry := false },
    { id := 'B', has_180_degree_symmetry := false },
    { id := 'C', has_180_degree_symmetry := true },
    { id := 'D', has_180_degree_symmetry := false },
    { id := 'E', has_180_degree_symmetry := false }
  ]

/-- Function to check if a shape matches the illustrated shape after 180-degree rotation -/
def matches_after_rotation (shape : RotatableShape) : Bool :=
  shape.has_180_degree_symmetry

theorem shape_C_matches_illustrated_shape :
  ∃! shape, shape ∈ given_shapes ∧ matches_after_rotation shape ∧ shape.id = 'C' :=
by
  sorry

#check shape_C_matches_illustrated_shape

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_C_matches_illustrated_shape_l206_20614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_rationals_are_integers_l206_20636

/-- Definition of a "good" rational number -/
def is_good (x : ℚ) : Prop :=
  ∃ (p q : ℕ) (α : ℝ) (N : ℕ),
    x = p / q ∧ x > 1 ∧ Nat.Coprime p q ∧
    ∀ (n : ℕ), n ≥ N →
      |((x^n : ℚ) - (x^n : ℚ).floor) - α| ≤ 1 / (2 * (p + q))

/-- Theorem: All "good" rational numbers are integers greater than 1 -/
theorem good_rationals_are_integers (x : ℚ) :
  is_good x → ∃ (n : ℕ), x = n ∧ x > 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_rationals_are_integers_l206_20636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l206_20618

open Real

/-- A function f(x) = ax + cos(x) with two extreme points in [0, π] -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.cos x

/-- The condition that f has two extreme points in [0, π] -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ π ∧
    (∀ x, 0 ≤ x ∧ x ≤ π → f a x ≤ max (f a x₁) (f a x₂)) ∧
    (∃ y, 0 < y ∧ y < π ∧ f a y < min (f a x₁) (f a x₂))

/-- Theorem stating the properties of f -/
theorem f_properties (a : ℝ) (h : has_two_extreme_points a) :
  (0 < a ∧ a < 1) ∧
  (∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ π →
    0 < f a x₁ - f a x₂ ∧ f a x₁ - f a x₂ < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l206_20618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_on_interval_l206_20652

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - 2 * (Real.sin x) ^ 2

-- Statement for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

-- Statement for the range on the given interval
theorem range_on_interval :
  Set.Icc (-2 : ℝ) (Real.sqrt 2 - 1) =
  {y | ∃ x ∈ Set.Icc (- Real.pi / 4) (Real.pi / 8), f x = y} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_on_interval_l206_20652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l206_20669

noncomputable def f (x : ℝ) := (Real.sin x - Real.cos x) * Real.sin x

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l206_20669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_range_of_m_l206_20646

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ := if a > b then a else b

-- Theorem statement
theorem otimes_range_of_m (m : ℝ) : 
  (otimes (2 * m - 5) 3 = 3) ↔ (m ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_range_of_m_l206_20646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_proof_l206_20693

/-- The number of ways to arrange 5 people in a row with exactly one person between A and B -/
def arrangement_count : ℕ := 36

/-- The number of people in the arrangement -/
def total_people : ℕ := 5

/-- The number of people that must be between A and B -/
def people_between : ℕ := 1

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem arrangement_count_proof :
  arrangement_count = (total_people - 2) * 2 * factorial (total_people - 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_proof_l206_20693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martian_clock_alignment_l206_20647

/-- Represents the number of hours in a Martian day -/
def martian_hours : ℕ := 13

/-- Represents the number of complete revolutions the hour hand makes in a Martian day -/
def hour_hand_revolutions : ℕ := 1

/-- Represents the number of complete revolutions the minute hand makes in a Martian day -/
def minute_hand_revolutions : ℕ := 13

/-- Theorem stating that the number of times all three clock hands align in a Martian day is a divisor of 12 -/
theorem martian_clock_alignment (n : ℕ) :
  ∃ (k : ℕ), k ∣ 12 ∧ k = Nat.gcd 12 n := by
  sorry

#check martian_clock_alignment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martian_clock_alignment_l206_20647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_correct_sum_abcd_correct_l206_20638

-- Define the trapezoid and circles
structure Trapezoid :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℝ)
  (circle_EF_radius : ℝ)
  (circle_GH_radius : ℝ)

-- Define the properties of the specific trapezoid in the problem
def problem_trapezoid : Trapezoid :=
  { EF := 10
  , FG := 7
  , GH := 7
  , HE := 8
  , circle_EF_radius := 4
  , circle_GH_radius := 3
  }

-- Define the function to calculate the radius of the inner circle
noncomputable def inner_circle_radius (t : Trapezoid) : ℝ :=
  (-78 + 70 * Real.sqrt 3) / 26

-- State the theorem
theorem inner_circle_radius_correct (t : Trapezoid) :
  t = problem_trapezoid →
  ∃ (r : ℝ), r = inner_circle_radius t ∧
  r > 0 ∧
  -- Additional conditions to specify that the inner circle is
  -- fully enclosed and tangent to all four circles would be needed here
  True := by
  sorry

-- Define the sum of a, b, c, and d
def sum_abcd : ℕ := 78 + 70 + 3 + 26

-- Theorem stating that the sum is correct
theorem sum_abcd_correct :
  sum_abcd = 177 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_correct_sum_abcd_correct_l206_20638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_sixth_power_l206_20679

theorem fourth_root_sixteen_to_sixth_power : (16 : ℝ) ^ (1/4) ^ 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_sixth_power_l206_20679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_building_cost_l206_20665

/-- The optimal number of floors that minimizes the average comprehensive cost -/
def optimal_floors : ℕ := 16

/-- The minimum average comprehensive cost per square meter -/
noncomputable def min_avg_cost : ℝ := 2120

/-- The average comprehensive cost per square meter as a function of the number of floors -/
noncomputable def avg_cost (x : ℝ) : ℝ := 520 + 50 * x + 12800 / x

theorem optimal_building_cost (x : ℕ) (h : x ≥ 12) :
  avg_cost (optimal_floors : ℝ) ≤ avg_cost (x : ℝ) ∧
  avg_cost (optimal_floors : ℝ) = min_avg_cost := by
  sorry

#check optimal_building_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_building_cost_l206_20665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3_cos_2_positive_l206_20694

theorem tan_3_cos_2_positive :
  (∀ x : ℝ, π / 2 < x ∧ x < π → Real.cos x < 0) →
  (∀ x : ℝ, π / 2 < x ∧ x < π → Real.tan x < 0) →
  (∀ x : ℝ, π / 2 < x ∧ x < π → Real.sin x > 0) →
  π / 2 < 2 ∧ 2 < π →
  π / 2 < 3 ∧ 3 < π →
  Real.tan 3 * Real.cos 2 > 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3_cos_2_positive_l206_20694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l206_20674

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define Z₁ as a function of a
noncomputable def Z₁ (a : ℝ) : ℂ := 2 + a * i

-- State the theorem
theorem complex_number_problem (a : ℝ) (ha : a > 0) 
  (h_pure_imag : ∃ b : ℝ, (Z₁ a)^2 = b * i) :
  a = 2 ∧ Complex.abs (Z₁ a / (1 - i)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l206_20674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_radius_is_125_over_23_l206_20617

/-- A tangential quadrilateral is a quadrilateral that has an inscribed circle touching all four sides. -/
structure TangentialQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  tangential : True  -- This is a placeholder for the tangential property

/-- The radius of the largest inscribed circle in a tangential quadrilateral. -/
noncomputable def largest_inscribed_circle_radius (q : TangentialQuadrilateral) : ℝ :=
  let s := (q.a + q.b + q.c + q.d) / 2
  let area := Real.sqrt ((s - q.a) * (s - q.b) * (s - q.c) * (s - q.d))
  area / s

/-- The theorem stating that for a specific tangential quadrilateral, 
    the radius of the largest inscribed circle is 125/23. -/
theorem largest_circle_radius_is_125_over_23 :
  ∃ q : TangentialQuadrilateral, 
    q.a = 15 ∧ q.b = 10 ∧ q.c = 8 ∧ q.d = 13 ∧ 
    largest_inscribed_circle_radius q = 125 / 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_radius_is_125_over_23_l206_20617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_transformed_set_l206_20699

noncomputable def T (x y : ℝ) : ℝ × ℝ := (x / (x^2 + y^2), -y / (x^2 + y^2))

def square_edge : Set (ℝ × ℝ) :=
  {p | (p.1 = 1 ∨ p.1 = -1) ∧ p.2 ∈ Set.Icc (-1) 1 ∨
       (p.2 = 1 ∨ p.2 = -1) ∧ p.1 ∈ Set.Icc (-1) 1}

def transformed_set : Set (ℝ × ℝ) :=
  {p | ∃ q ∈ square_edge, p = T q.1 q.2}

theorem area_of_transformed_set :
  MeasureTheory.volume transformed_set = 4 := by
  sorry

#check area_of_transformed_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_transformed_set_l206_20699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_101_equals_52_l206_20690

def F : ℕ → ℚ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | (n + 1) => (2 * F n + 1) / 2

theorem F_101_equals_52 : F 101 = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_101_equals_52_l206_20690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_minimized_at_four_thirds_l206_20612

/-- The coordinates of point A as a function of x -/
def point_A (x : ℝ) : Fin 3 → ℝ := ![x, 5 - x, 2 * x - 1]

/-- The coordinates of point B as a function of x -/
def point_B (x : ℝ) : Fin 3 → ℝ := ![1, x + 2, 2 - x]

/-- The squared distance between two points in 3D space -/
def squared_distance (p q : Fin 3 → ℝ) : ℝ :=
  (p 0 - q 0)^2 + (p 1 - q 1)^2 + (p 2 - q 2)^2

/-- The theorem stating that the distance between A and B is minimized when x = 4/3 -/
theorem distance_minimized_at_four_thirds :
  ∀ x : ℝ, squared_distance (point_A x) (point_B x) ≥ squared_distance (point_A (4/3)) (point_B (4/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_minimized_at_four_thirds_l206_20612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l206_20637

theorem power_equality (a b : ℝ) (h1 : (10 : ℝ)^a = 2) (h2 : (100 : ℝ)^b = 7) : 
  (10 : ℝ)^(2*a - 2*b) = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l206_20637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_special_angle_l206_20663

theorem cos_sum_special_angle (θ : ℝ) 
  (h1 : Real.cos θ = -12/13) 
  (h2 : θ ∈ Set.Ioo π (3*π/2)) : 
  Real.cos (θ + π/4) = -7*Real.sqrt 2/26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_special_angle_l206_20663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_4x_plus_2y_l206_20642

theorem range_of_4x_plus_2y (x y : ℝ) 
  (h1 : 1 ≤ x + y ∧ x + y ≤ 3) 
  (h2 : -1 ≤ x - y ∧ x - y ≤ 1) : 
  2 ≤ 4*x + 2*y ∧ 4*x + 2*y ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_4x_plus_2y_l206_20642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l206_20672

-- Define the circle and ellipse
def circleEq (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 1
def ellipseEq (x y : ℝ) : Prop := x^2/9 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem max_distance_circle_ellipse :
  ∀ x1 y1 x2 y2 : ℝ,
  circleEq x1 y1 → ellipseEq x2 y2 →
  distance x1 y1 x2 y2 ≤ 1 + 3 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l206_20672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l206_20685

noncomputable def f (x : ℝ) : ℝ := Real.sin x * (Real.cos x - Real.sqrt 3 * Real.sin x)

theorem f_properties :
  let a : ℝ := 0
  let b : ℝ := Real.pi / 2
  (f (Real.pi / 6) = 0) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 1 - Real.sqrt 3 / 2) ∧
  (∀ x ∈ Set.Icc a b, -Real.sqrt 3 ≤ f x) ∧
  (∃ x ∈ Set.Icc a b, f x = 1 - Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc a b, f x = -Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l206_20685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_count_l206_20691

theorem inequality_count (a b : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : a + b > 0) :
  (if a^2 * b < b^3 then 1 else 0) +
  (if (1 / a > 0 ∧ 0 > 1 / b) then 1 else 0) +
  (if a^3 < a * b^2 then 1 else 0) +
  (if a^3 > b^3 then 1 else 0) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_count_l206_20691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_scalene_no_option_correct_l206_20627

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := 2 * x + 3
noncomputable def line2 (x : ℝ) : ℝ := -4 * x + 3
noncomputable def line3 : ℝ := 1

-- Define the intersection points
noncomputable def point1 : ℝ × ℝ := (0, 3)
noncomputable def point2 : ℝ × ℝ := (-1, 1)
noncomputable def point3 : ℝ × ℝ := (1/2, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem: The triangle formed by the intersection of the given lines is scalene
theorem triangle_is_scalene :
  let side1 := distance point1 point2
  let side2 := distance point1 point3
  let side3 := distance point2 point3
  side1 ≠ side2 ∧ side1 ≠ side3 ∧ side2 ≠ side3 := by
  sorry

-- Theorem: None of the given options correctly describe the triangle formed
theorem no_option_correct : True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_scalene_no_option_correct_l206_20627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_value_l206_20696

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x^3 - x + 1 else x^2 - a * x

-- State the theorem
theorem function_composition_value (a : ℝ) :
  f a (f a 0) = -2 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_value_l206_20696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_287_is_blue_l206_20686

/-- Represents the color of a marble -/
inductive Color where
  | Blue
  | Green
  | Red

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : Color :=
  match n % 15 with
  | k => if k < 6 then Color.Blue
         else if k < 11 then Color.Green
         else Color.Red

/-- The main theorem: The 287th marble is blue -/
theorem marble_287_is_blue : marbleColor 287 = Color.Blue := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_287_is_blue_l206_20686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_tax_rate_is_30_percent_l206_20650

-- Define the given values
noncomputable def john_income : ℝ := 57000
noncomputable def ingrid_income : ℝ := 72000
noncomputable def ingrid_tax_rate : ℝ := 0.4
noncomputable def combined_tax_rate : ℝ := 0.35581395348837205

-- Define the function to calculate John's tax rate
noncomputable def john_tax_rate : ℝ :=
  let total_income := john_income + ingrid_income
  let total_tax := combined_tax_rate * total_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  (total_tax - ingrid_tax) / john_income

-- Theorem statement
theorem john_tax_rate_is_30_percent :
  john_tax_rate = 0.3 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_tax_rate_is_30_percent_l206_20650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_seven_pairs_l206_20634

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 15 ∧ 1 ≤ b ∧ b ≤ 15 ∧ (Int.natAbs (a - b)) % 7 = 0

def valid_pairs : List (ℕ × ℕ) :=
  [(8, 1), (9, 2), (10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (15, 1)]

theorem divisible_by_seven_pairs :
  ∀ a b : ℕ, is_valid_pair a b ↔ (a, b) ∈ valid_pairs :=
by sorry

#check divisible_by_seven_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_seven_pairs_l206_20634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_cube_root_at_one_l206_20684

-- Define the function f(x) = ∛x
noncomputable def f (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem derivative_cube_root_at_one :
  deriv f 1 = 1/3 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_cube_root_at_one_l206_20684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_commission_is_nine_percent_l206_20607

/-- Represents the commission structure for a salesman -/
structure CommissionStructure where
  totalCommission : ℚ
  bonus : ℚ
  bonusRate : ℚ
  baseSales : ℚ

/-- Calculates the initial commission percentage given a commission structure -/
def initialCommissionPercentage (cs : CommissionStructure) : ℚ :=
  let excessSales := cs.bonus / cs.bonusRate
  let totalSales := cs.baseSales + excessSales
  let initialCommission := cs.totalCommission - cs.bonus
  (initialCommission / totalSales) * 100

/-- Theorem stating that for the given commission structure, the initial commission percentage is 9% -/
theorem initial_commission_is_nine_percent :
  let cs : CommissionStructure := {
    totalCommission := 1380,
    bonus := 120,
    bonusRate := 3/100,
    baseSales := 10000
  }
  initialCommissionPercentage cs = 9 := by
  sorry

#eval initialCommissionPercentage {
  totalCommission := 1380,
  bonus := 120,
  bonusRate := 3/100,
  baseSales := 10000
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_commission_is_nine_percent_l206_20607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l206_20600

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 3/4

-- Define the tangent line
def tangent_line (x y k : ℝ) : Prop := y = k * x

-- Define the hyperbola
def hyperbola_eq (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Statement of the theorem
theorem hyperbola_eccentricity_range 
  (a b k : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hk : ∃ (x y : ℝ), circle_eq x y ∧ tangent_line x y k)
  (hi : ∃ (x1 y1 x2 y2 : ℝ), 
    x1 ≠ x2 ∧ 
    tangent_line x1 y1 k ∧ 
    tangent_line x2 y2 k ∧ 
    hyperbola_eq x1 y1 a b ∧ 
    hyperbola_eq x2 y2 a b) :
  eccentricity a b > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l206_20600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_grade_students_l206_20664

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (increase_percent : ℚ) : 
  initial = 10 → left = 4 → increase_percent = 70 / 100 →
  let remaining := initial - left
  let increase := (increase_percent * remaining).floor
  remaining + increase = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_grade_students_l206_20664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BFEC_is_700_l206_20603

/-- Rectangle ABCD with given dimensions and midpoints --/
structure Rectangle where
  AB : ℝ
  BC : ℝ
  D : ℝ × ℝ  -- Midpoint of AB
  E : ℝ × ℝ  -- Midpoint of BC
  F : ℝ × ℝ  -- Midpoint of AD

/-- Definition of the specific rectangle in the problem --/
def problem_rectangle : Rectangle where
  AB := 40
  BC := 28
  D := (20, 0)  -- Assuming A is at (0,0)
  E := (40, 14)
  F := (10, 0)

/-- Calculate the area of quadrilateral BFEC --/
noncomputable def area_BFEC (r : Rectangle) : ℝ := 
  r.AB * r.BC - (r.AB / 2 * r.BC / 2) - (r.BC * r.AB / 4)

/-- Theorem stating that the area of BFEC in the problem rectangle is 700 --/
theorem area_BFEC_is_700 : 
  area_BFEC problem_rectangle = 700 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BFEC_is_700_l206_20603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_sequence_existence_l206_20673

theorem bounded_sequence_existence (a : ℝ) (h : a > 1) :
  ∃ (x : ℕ → ℝ), 
    (∃ (M : ℝ), ∀ n, |x n| ≤ M) ∧ 
    (∀ i j : ℕ, i ≠ j → |x i - x j| * (|i - j| : ℝ)^a ≥ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_sequence_existence_l206_20673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_satisfies_conditions_P_unique_l206_20651

/-- A polynomial that satisfies the given conditions -/
def P (n : ℕ) (x : ℝ) : ℝ := (x - 1995) ^ n

/-- The theorem stating that P satisfies the required conditions -/
theorem P_satisfies_conditions (n : ℕ) (hn : n > 0) :
  (∀ a > 1995, (Set.ncard {x : ℝ | P n x = a} = n)) ∧
  (∀ a > 1995, ∀ x : ℝ, P n x = a → x > 1995) := by
  sorry

/-- The theorem stating that P is the only polynomial satisfying the conditions -/
theorem P_unique (Q : ℝ → ℝ) :
  (∀ a > 1995, (Set.ncard {x : ℝ | Q x = a} > 1995)) ∧
  (∀ a > 1995, ∀ x : ℝ, Q x = a → x > 1995) →
  ∃ n : ℕ, n > 0 ∧ ∀ x : ℝ, Q x = P n x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_satisfies_conditions_P_unique_l206_20651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_line_in_plane_relationship_l206_20605

-- Define the necessary structures
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

structure Plane3D where
  point : Fin 3 → ℝ
  normal : Fin 3 → ℝ

-- Define the relationships
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  ∀ t, (l.direction t) • (p.normal t) = 0

def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  ∀ t, (l.point t - p.point t) • (p.normal t) = 0

def parallel_lines (l1 l2 : Line3D) : Prop :=
  ∃ k : ℝ, ∀ t, l1.direction t = k * l2.direction t

def skew_lines (l1 l2 : Line3D) : Prop :=
  ¬ (parallel_lines l1 l2) ∧ ¬ (∃ p, line_in_plane l1 p ∧ line_in_plane l2 p)

-- The main theorem
theorem line_parallel_to_plane_line_in_plane_relationship 
  (a b : Line3D) (α : Plane3D) 
  (h1 : parallel_line_plane a α) 
  (h2 : line_in_plane b α) : 
  parallel_lines a b ∨ skew_lines a b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_line_in_plane_relationship_l206_20605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l206_20695

theorem trigonometric_problem (α β : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π)
  (h2 : Real.sin α = 1 / 3)
  (h3 : Real.sin (α + β) = -3 / 5)
  (h4 : β ∈ Set.Ioo 0 (π / 2)) :
  (Real.sin (2 * α) = -4 * Real.sqrt 2 / 9) ∧ 
  (Real.sin β = (6 * Real.sqrt 2 + 4) / 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l206_20695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l206_20616

/-- The function f(x) -/
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := 1 + 1/x - x^α

/-- Theorem stating the main results -/
theorem function_properties :
  ∃ (α : ℝ),
    (f α 3 = -5/3) ∧
    (α = 1) ∧
    (∀ x : ℝ, f 1 x = 0 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2) ∧
    (∀ x y : ℝ, x < y ∧ x < 0 ∧ y < 0 → f 1 x > f 1 y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l206_20616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_sqrt_and_cube_l206_20666

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x
def g (x : ℝ) : ℝ := x^3

-- Define the enclosed area
noncomputable def enclosed_area : ℝ :=
  ∫ x in (0 : ℝ)..1, f x - g x

-- Theorem statement
theorem area_between_sqrt_and_cube :
  enclosed_area = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_sqrt_and_cube_l206_20666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_20_is_maximum_l206_20620

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  condition : 3 * a 8 = 5 * a 13
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The sum of the first n terms of the arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * seq.a 1 + (n - 1 : ℝ) * (seq.a 2 - seq.a 1)) / 2

/-- The theorem stating that S_20 is the maximum sum -/
theorem S_20_is_maximum (seq : ArithmeticSequence) :
  ∀ n : ℕ, S seq 20 ≥ S seq n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_20_is_maximum_l206_20620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_binomial_coefficient_in_expansion_l206_20631

theorem largest_binomial_coefficient_in_expansion (n : ℕ) :
  let expansion := (1 - X : Polynomial ℚ)^(2*n - 1)
  let nth_coeff := abs (expansion.coeff n)
  let nplus1th_coeff := abs (expansion.coeff (n + 1))
  ∀ k, k ≠ n ∧ k ≠ n + 1 →
    nth_coeff ≥ abs (expansion.coeff k) ∧
    nplus1th_coeff ≥ abs (expansion.coeff k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_binomial_coefficient_in_expansion_l206_20631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_equals_half_l206_20613

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then sin θ = 1/2 -/
theorem sin_theta_equals_half (θ : ℝ) : 
  (∃ (t : ℝ), t > 0 ∧ t * Real.cos θ = -Real.sqrt 3 / 2 ∧ t * Real.sin θ = 1 / 2) → 
  Real.sin θ = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_equals_half_l206_20613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l206_20604

/-- A function that is even and monotonically increasing on (-∞, 0) -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x < y ∧ y ≤ 0 → f x < f y)

/-- The main theorem -/
theorem a_range_theorem (f : ℝ → ℝ) (a : ℝ) 
  (h_f : EvenIncreasingFunction f)
  (h_ineq : f (2^(a-1)) > f (-Real.sqrt 2)) : 
  1 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_theorem_l206_20604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_in_specific_triangle_l206_20644

/-- An isosceles triangle with a semicircle inscribed along its base -/
structure IsoscelesTriangleWithSemicircle where
  base : ℝ
  height : ℝ
  is_isosceles : base > 0 ∧ height > 0

/-- Calculate the radius of the inscribed semicircle -/
noncomputable def semicircle_radius (triangle : IsoscelesTriangleWithSemicircle) : ℝ :=
  (3 * triangle.base * triangle.height) / (2 * (triangle.base^2 + 4 * triangle.height^2).sqrt)

theorem semicircle_radius_in_specific_triangle :
  let triangle : IsoscelesTriangleWithSemicircle := ⟨24, 18, by norm_num⟩
  semicircle_radius triangle = 36 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_in_specific_triangle_l206_20644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l206_20687

/-- An ellipse with equation x²/3 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 3 + p.2^2 = 1}

/-- A line with equation y = kx + m -/
def Line (k m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 + m}

/-- The origin point (0, 0) -/
def O : ℝ × ℝ := (0, 0)

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (k m : ℝ) : ℝ :=
  abs (m - k * p.1 + p.2) / Real.sqrt (1 + k^2)

/-- The dot product of two vectors -/
def dotProduct (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2)) / 2

theorem ellipse_line_intersection
  (k m : ℝ)
  (A B : ℝ × ℝ)
  (hA : A ∈ Ellipse ∩ Line k m)
  (hB : B ∈ Ellipse ∩ Line k m)
  (hAB : A ≠ B) :
  (m = 1 ∧ dotProduct (A - O) (B - O) = 0 → k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3) ∧
  (distancePointToLine O k m = Real.sqrt 3 / 2 →
    ∀ C, C ∈ Ellipse ∩ Line k m → triangleArea O A C ≤ Real.sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l206_20687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_max_area_PMQN_l206_20633

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (0, 2)

-- Define the line y = x
def line_y_eq_x (x : ℝ) : ℝ := x

-- Define the line l: y = kx + 1
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the point (0, 1)
def point_0_1 : ℝ × ℝ := (0, 1)

-- Theorem for the equation of circle C
theorem circle_C_equation : circle_C = {p | p.1^2 + p.2^2 = 4} := by sorry

-- Helper function to calculate the area of a quadrilateral
noncomputable def area_quadrilateral (P Q M N : ℝ × ℝ) : ℝ := sorry

-- Theorem for the maximum area of quadrilateral PMQN
theorem max_area_PMQN : 
  ∃ (P Q M N : ℝ × ℝ), 
    P ∈ circle_C ∧ Q ∈ circle_C ∧ M ∈ circle_C ∧ N ∈ circle_C ∧
    (∃ k, P.2 = line_l k P.1 ∧ Q.2 = line_l k Q.1) ∧
    (∃ m, M.2 - point_0_1.2 = -1/m * (M.1 - point_0_1.1) ∧ 
          N.2 - point_0_1.2 = -1/m * (N.1 - point_0_1.1)) ∧
    (∀ P' Q' M' N' : ℝ × ℝ, 
      P' ∈ circle_C → Q' ∈ circle_C → M' ∈ circle_C → N' ∈ circle_C →
      (∃ k', P'.2 = line_l k' P'.1 ∧ Q'.2 = line_l k' Q'.1) →
      (∃ m', M'.2 - point_0_1.2 = -1/m' * (M'.1 - point_0_1.1) ∧ 
             N'.2 - point_0_1.2 = -1/m' * (N'.1 - point_0_1.1)) →
      area_quadrilateral P Q M N ≥ area_quadrilateral P' Q' M' N') ∧
    area_quadrilateral P Q M N = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_max_area_PMQN_l206_20633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_temperature_l206_20683

/-- Calculates the temperature on Friday given the temperatures for the other days of the week and the average temperature. -/
theorem friday_temperature 
  (sunday_temp : ℝ) 
  (monday_temp : ℝ) 
  (tuesday_temp : ℝ) 
  (wednesday_temp : ℝ) 
  (thursday_temp : ℝ) 
  (saturday_temp : ℝ) 
  (average_temp : ℝ) 
  (h1 : sunday_temp = 40) 
  (h2 : monday_temp = 50) 
  (h3 : tuesday_temp = 65) 
  (h4 : wednesday_temp = 36) 
  (h5 : thursday_temp = 82) 
  (h6 : saturday_temp = 26) 
  (h7 : average_temp = 53) :
  ∃ (friday_temp : ℝ), 
    friday_temp = 72 ∧ 
    (sunday_temp + monday_temp + tuesday_temp + wednesday_temp + thursday_temp + friday_temp + saturday_temp) / 7 = average_temp :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_temperature_l206_20683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_from_points_l206_20619

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A predicate that checks if a point is inside or on the edge of a triangle -/
def isInsideOrOnTriangle (p q r s : Point) : Prop := sorry

/-- A predicate that checks if a set of points forms a convex polygon -/
def isConvexPolygon (points : Finset Point) : Prop := sorry

theorem convex_polygon_from_points (n : ℕ) (points : Finset Point) 
  (h_card : points.card = n)
  (h_triangle : ∀ p q r s, p ∈ points → q ∈ points → r ∈ points → s ∈ points → 
    p ≠ q ∧ q ≠ r ∧ r ≠ p → ¬isInsideOrOnTriangle p q r s) :
  isConvexPolygon points :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_from_points_l206_20619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debby_candy_consumption_l206_20676

theorem debby_candy_consumption (total_candy : ℕ) (eaten_fraction : ℚ) (eaten_candy : ℕ) : 
  total_candy = 12 → eaten_fraction = 2/3 → eaten_candy = (eaten_fraction * total_candy).floor → eaten_candy = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debby_candy_consumption_l206_20676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_squares_l206_20611

def x : ℕ → ℤ
  | 0 => 2
  | 1 => 7
  | (n + 2) => 4 * x (n + 1) - x n

theorem no_perfect_squares : ∀ n : ℕ, ∀ k : ℤ, x n ≠ k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_squares_l206_20611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_difference_l206_20688

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_side_difference (x y : ℝ) :
  (∃ (k : ℝ), triangle_area 3 5 k = 6) →
  (x > 0 ∧ y > 0 ∧ x ≠ y) →
  (triangle_area 3 5 x = 6 ∧ triangle_area 3 5 y = 6) →
  |x^2 - y^2| = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_difference_l206_20688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sequence_reappearance_l206_20639

/-- Represents a cyclic sequence -/
structure CyclicSequence where
  length : Nat
  current_position : Nat

/-- Theorem: Two cyclic sequences of lengths 5 and 4 will first reappear 
    in their original order simultaneously after 20 cycles -/
theorem cyclic_sequence_reappearance : 
  ∀ (seq1 seq2 : CyclicSequence), 
  seq1.length = 5 → 
  seq2.length = 4 → 
  Nat.lcm seq1.length seq2.length = 20 := by
  sorry

#eval Nat.lcm 5 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sequence_reappearance_l206_20639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l206_20668

-- Define the variables and constants
variable (x y z k : ℝ)

-- Define the constraints
def constraint1 (x y : ℝ) : Prop := x - 4*y + 3 ≤ 0
def constraint2 (x y : ℝ) : Prop := 3*x + 5*y - 25 ≤ 0
def constraint3 (x : ℝ) : Prop := x ≥ 1

-- Define the objective function
def objective_function (x y k : ℝ) : ℝ := k*x + y

-- Define the theorem
theorem find_k : 
  (∀ x y, constraint1 x y) →
  (∀ x y, constraint2 x y) →
  (∀ x, constraint3 x) →
  (∃ x y, objective_function x y k = 12) →
  (∃ x y, objective_function x y k = 3) →
  k = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l206_20668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l206_20661

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (angle_sum : A + B + C = π)
  (cosine_law : a^2 = b^2 + c^2 - 2*b*c*(Real.cos A))

-- Define the given condition
def special_triangle (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = t.a^2 + t.b*t.c

-- Define the function f(x)
noncomputable def f (A x : ℝ) : ℝ :=
  Real.sin (x - A) + Real.sqrt 3 * Real.cos x

-- State the theorem
theorem special_triangle_properties (t : Triangle) (h : special_triangle t) :
  t.A = π/3 ∧ ∀ x, f t.A x ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l206_20661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tree_crossing_time_specific_l206_20629

/-- The time it takes for a train to cross a tree -/
noncomputable def train_tree_crossing_time (train_length platform_length platform_crossing_time : ℝ) : ℝ :=
  let train_speed := (train_length + platform_length) / platform_crossing_time
  train_length / train_speed

/-- Theorem: A 1200 m long train that takes 210 sec to pass a 900 m platform will take 120 sec to cross a tree -/
theorem train_tree_crossing_time_specific :
  train_tree_crossing_time 1200 900 210 = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tree_crossing_time_specific_l206_20629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_green_ball_is_17_40_l206_20610

-- Define the containers and their contents
def containerA : Fin 2 → Nat := ![4, 6]  -- [red, green]
def containerB : Fin 2 → Nat := ![7, 3]
def containerC : Fin 2 → Nat := ![7, 3]
def containerD : Fin 2 → Nat := ![5, 5]

-- Define the total number of containers
def totalContainers : Nat := 4

-- Function to calculate probability of selecting a green ball from a container
def probGreenFromContainer (container : Fin 2 → Nat) : ℚ :=
  container 1 / (container 0 + container 1)

-- Theorem to prove
theorem prob_green_ball_is_17_40 :
  let probA := (1 : ℚ) / totalContainers * probGreenFromContainer containerA
  let probB := (1 : ℚ) / totalContainers * probGreenFromContainer containerB
  let probC := (1 : ℚ) / totalContainers * probGreenFromContainer containerC
  let probD := (1 : ℚ) / totalContainers * probGreenFromContainer containerD
  probA + probB + probC + probD = 17 / 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_green_ball_is_17_40_l206_20610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_25_l206_20654

def sequence_a : ℕ → ℕ
  | 0 => 1
  | (n + 1) => sequence_a n + 2 * (n + 1) + 1

theorem a_5_equals_25 : sequence_a 5 = 25 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_25_l206_20654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_value_l206_20648

theorem min_lambda_value (x y lambda : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * Real.sqrt (2 * x * y) ≤ lambda * (x + y)) : 
  lambda ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_value_l206_20648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_set_l206_20655

theorem equation_solution_set (x y : ℝ) :
  x^3 * (x + y + 2) = y^3 * (x + y + 2) ↔ (y = x ∨ y = -x - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_set_l206_20655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_max_S_l206_20640

-- Define the inequality
noncomputable def inequality (x : ℝ) : Prop := |x^2 - 3*x - 4| < 2*x + 2

-- Define the solution set
def solution_set (a b : ℝ) : Set ℝ := {x | a < x ∧ x < b}

-- Define S
noncomputable def S (a b m n : ℝ) : ℝ := a / (m^2 - 1) + b / (3 * (n^2 - 1))

-- Theorem statement
theorem inequality_solution_and_max_S :
  ∃ (a b : ℝ),
    (∀ x, x ∈ solution_set a b ↔ inequality x) ∧
    a = 2 ∧ b = 6 ∧
    (∀ m n : ℝ, m ∈ Set.Ioo (-1 : ℝ) 1 → n ∈ Set.Ioo (-1 : ℝ) 1 →
      m * n = a / b →
      S a b m n ≤ -6 ∧
      (S a b m n = -6 ↔ m = Real.sqrt 3/3 ∨ m = -Real.sqrt 3/3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_max_S_l206_20640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_both_curves_l206_20667

-- Define the tangent line
def tangent_line (k b x : ℝ) : ℝ := k * x + b

-- Define the first curve
noncomputable def curve1 (x : ℝ) : ℝ := Real.log x + 2

-- Define the second curve
noncomputable def curve2 (x : ℝ) : ℝ := Real.log (x + 1)

-- State the theorem
theorem tangent_to_both_curves (k b : ℝ) :
  (∃ x1 > 0, tangent_line k b x1 = curve1 x1 ∧ k = (deriv curve1) x1) →
  (∃ x2 > -1, tangent_line k b x2 = curve2 x2 ∧ k = (deriv curve2) x2) →
  b = 1 - Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_both_curves_l206_20667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AOB_isosceles_right_triangle_l206_20681

def complex_to_vector (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def point_A : ℝ × ℝ := complex_to_vector (-3 + (1 : ℂ))
def point_B : ℝ × ℝ := complex_to_vector (-1 - 3*(1 : ℂ))
def point_O : ℝ × ℝ := (0, 0)

def vector_OA : ℝ × ℝ := point_A
def vector_OB : ℝ × ℝ := point_B

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem AOB_isosceles_right_triangle :
  magnitude vector_OA = magnitude vector_OB ∧ 
  dot_product vector_OA vector_OB = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AOB_isosceles_right_triangle_l206_20681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_implies_a_range_l206_20662

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem monotonic_increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 < a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_implies_a_range_l206_20662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_divisible_by_27_but_number_not_l206_20659

noncomputable def sum_of_digits : ℕ → ℕ
| 0 => 0
| n + 1 => (n + 1) % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_divisible_by_27_but_number_not : ∃ n : ℕ, 
  (∃ k : ℕ, sum_of_digits n = 27 * k) ∧ ¬(∃ m : ℕ, n = 27 * m) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_divisible_by_27_but_number_not_l206_20659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_l206_20606

theorem cos_2beta (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan α = 1/7 →
  Real.cos (α + β) = 2*Real.sqrt 5/5 →
  Real.cos (2*β) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_l206_20606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seashell_ratio_l206_20628

theorem seashell_ratio (initial : ℕ) (given_friends : ℕ) (given_brothers : ℕ) (left_after_selling : ℕ)
  (h1 : initial = 180)
  (h2 : given_friends = 40)
  (h3 : given_brothers = 30)
  (h4 : left_after_selling = 55) :
  (initial - (given_friends + given_brothers) - left_after_selling : ℚ) / 
  (initial - (given_friends + given_brothers)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seashell_ratio_l206_20628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_wedge_volume_l206_20645

/-- Given a sphere with circumference 18π inches cut into six congruent wedges,
    prove that the volume of one wedge is 162π cubic inches. -/
theorem sphere_wedge_volume (circumference : ℝ) (num_wedges : ℕ) :
  circumference = 18 * Real.pi →
  num_wedges = 6 →
  let radius := circumference / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * radius^3
  let wedge_volume := sphere_volume / (num_wedges : ℝ)
  wedge_volume = 162 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_wedge_volume_l206_20645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_equals_one_sixteenth_l206_20624

theorem cos_product_equals_one_sixteenth : 
  Real.cos (π / 9) * Real.cos (2 * π / 9) * Real.cos (-23 * π / 9) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_equals_one_sixteenth_l206_20624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_miles_after_modification_l206_20625

/-- Calculates the additional miles a car can travel after modification -/
theorem additional_miles_after_modification
  (current_efficiency : ℝ)
  (tank_capacity : ℝ)
  (fuel_reduction_factor : ℝ)
  (h1 : current_efficiency = 24)
  (h2 : tank_capacity = 12)
  (h3 : fuel_reduction_factor = 0.75)
  : ℝ := by
  let new_efficiency := current_efficiency / fuel_reduction_factor
  let current_range := current_efficiency * tank_capacity
  let new_range := new_efficiency * tank_capacity
  have h4 : new_range - current_range = 72 := by sorry
  exact 72


end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_miles_after_modification_l206_20625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_three_l206_20621

/-- The line defined by the equation 3x - 4y + 2 = 0 -/
def line : Set (ℝ × ℝ) := {p | 3 * p.1 - 4 * p.2 + 2 = 0}

/-- The fixed point (3, -1) -/
def fixed_point : ℝ × ℝ := (3, -1)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The minimum distance from any point on the line to the fixed point is 3 -/
theorem min_distance_is_three :
  ∀ p ∈ line, distance p fixed_point ≥ 3 ∧ ∃ r ∈ line, distance r fixed_point = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_three_l206_20621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l206_20657

-- Define the function f
variable (f : ℝ → ℝ) (hf : Differentiable ℝ f)

-- Define the condition that xf'(x) > -f(x) for all x in ℝ
variable (h : ∀ x : ℝ, x * (deriv f x) > -f x)

-- Define a and b, with a > b
variable (a b : ℝ)
variable (hab : a > b)

-- State the theorem
theorem function_inequality : a * f a > b * f b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l206_20657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_l206_20608

open Real

-- Define the hyperbolic functions
noncomputable def sh (x : ℝ) : ℝ := (exp x - exp (-x)) / 2
noncomputable def ch (x : ℝ) : ℝ := (exp x + exp (-x)) / 2
noncomputable def cth (x : ℝ) : ℝ := ch x / sh x

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := (2/3) * cth x - ch x / (3 * sh x ^ 3)

-- State the theorem
theorem derivative_f (x : ℝ) (h : x ≠ 0) : 
  deriv f x = 1 / (sh x)^4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_l206_20608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_asymptote_l206_20682

/-- The rational function f(x) with parameter c -/
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (x^2 - x + c) / (x^2 - 6*x + 8)

/-- Predicate for f having exactly one vertical asymptote -/
def has_exactly_one_vertical_asymptote (c : ℝ) : Prop :=
  (∃! x : ℝ, x^2 - 6*x + 8 = 0 ∧ x^2 - x + c ≠ 0)

/-- Theorem stating the condition for f to have exactly one vertical asymptote -/
theorem f_one_asymptote (c : ℝ) :
  has_exactly_one_vertical_asymptote c ↔ c = -2 ∨ c = -12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_asymptote_l206_20682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_unit_circle_l206_20626

/-- Given a curve C with polar equation ρ = 1, prove that its Cartesian coordinate equation is x^2 + y^2 = 1 -/
theorem polar_to_cartesian_unit_circle (x y : ℝ) :
  (∃ θ : ℝ, x = Real.cos θ ∧ y = Real.sin θ) ↔ x^2 + y^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_unit_circle_l206_20626
