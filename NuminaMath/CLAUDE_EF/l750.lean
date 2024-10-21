import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l750_75013

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The coordinates of the right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola) : ℝ × ℝ :=
  (Real.sqrt (h.a^2 + h.b^2), 0)

/-- The coordinates of the left vertex of a hyperbola -/
def left_vertex (h : Hyperbola) : ℝ × ℝ :=
  (-h.a, 0)

/-- Point B on the hyperbola -/
noncomputable def point_B (h : Hyperbola) : ℝ × ℝ :=
  (0, Real.sqrt 15 / 3 * h.b)

/-- Theorem: If the perpendicular bisector of AB passes through the right focus,
    then the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity_is_two (h : Hyperbola) :
  let A := left_vertex h
  let B := point_B h
  let F := right_focus h
  (∀ P : ℝ × ℝ, P.1 = (A.1 + B.1) / 2 ∧ (P.2 - A.2) * (B.1 - A.1) = (P.1 - A.1) * (B.2 - A.2) →
    (P.1 - F.1)^2 + (P.2 - F.2)^2 = 0) →
  eccentricity h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l750_75013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_abs_values_l750_75089

theorem min_sum_abs_values (p q r s : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 → s ≠ 0 →
  (p * p + q * r = 10) →
  (p * q + q * s = 0) →
  (r * p + r * s = 0) →
  (q * r + s * s = 10) →
  (∃ (a b c d : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (a * a + b * c = 10) ∧
    (a * b + b * d = 0) ∧
    (c * a + c * d = 0) ∧
    (b * c + d * d = 10) ∧
    (abs a + abs b + abs c + abs d < abs p + abs q + abs r + abs s)) →
  abs p + abs q + abs r + abs s ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_abs_values_l750_75089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l750_75074

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Define the foci
def foci (F1 F2 : ℝ × ℝ) : Prop :=
  let c := (F1.1 - F2.1) / 2
  c^2 = 25 - 9

-- Define a point on the ellipse
variable (P : ℝ × ℝ)

-- State that P is on the ellipse
axiom P_on_ellipse : is_on_ellipse P.1 P.2

-- Define the angle F1PF2
def angle_F1PF2 (F1 F2 P : ℝ × ℝ) : ℝ := sorry

-- State that the angle F1PF2 is 60°
axiom angle_is_60 (F1 F2 : ℝ × ℝ) : angle_F1PF2 F1 F2 P = 60 * Real.pi / 180

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- The theorem to prove
theorem area_of_triangle (F1 F2 : ℝ × ℝ) 
  (h_foci : foci F1 F2) 
  (h_angle : angle_F1PF2 F1 F2 P = 60 * Real.pi / 180) :
  triangle_area F1 P F2 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l750_75074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₂_value_l750_75080

-- Define the complex number z₁
noncomputable def z₁ : ℂ := 2 * Complex.I + 1

-- Define the properties for z₂
def is_valid_z₂ (z₂ : ℂ) : Prop :=
  (z₁ + z₂).im = 0 ∧ (z₁ * z₂).re = 0 ∧ (z₁ * z₂).im ≠ 0

-- Theorem statement
theorem z₂_value : ∃ (z₂ : ℂ), is_valid_z₂ z₂ ∧ z₂ = -4 - 2 * Complex.I := by
  -- Construct z₂
  let z₂ : ℂ := -4 - 2 * Complex.I
  
  -- Prove that z₂ satisfies the conditions
  have h1 : (z₁ + z₂).im = 0 := by sorry
  have h2 : (z₁ * z₂).re = 0 := by sorry
  have h3 : (z₁ * z₂).im ≠ 0 := by sorry
  
  -- Conclude the proof
  exact ⟨z₂, ⟨⟨h1, h2, h3⟩, rfl⟩⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₂_value_l750_75080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_two_zeros_l750_75081

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - a else -x + a

theorem sufficient_condition_for_two_zeros (a : ℝ) (h : 1 < a ∧ a < 2) :
  ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_two_zeros_l750_75081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l750_75000

/-- The length of the shorter side of each smaller rectangle -/
noncomputable def short_side : ℝ := 6

/-- The ratio of the longer side to the shorter side of each smaller rectangle -/
noncomputable def side_ratio : ℝ := 3 / 2

/-- The length of the longer side of each smaller rectangle -/
noncomputable def long_side : ℝ := short_side * side_ratio

/-- The width of the larger rectangle ABCD -/
noncomputable def width : ℝ := 2 * short_side

/-- The length of the larger rectangle ABCD -/
noncomputable def length : ℝ := long_side

/-- The area of the larger rectangle ABCD -/
noncomputable def area : ℝ := width * length

theorem rectangle_area : area = 108 := by
  -- Unfold definitions
  unfold area width length long_side short_side side_ratio
  -- Simplify the expression
  simp [mul_assoc, mul_comm]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l750_75000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l750_75082

noncomputable def f (x : ℝ) : ℝ := (4^x + 1) / 2^x

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∀ x ≥ 0, deriv f x > 0) :=
by
  constructor
  · intro x
    simp [f]
    ring_nf
    -- The proof for symmetry would go here
    sorry
  · intro x hx
    simp [f, deriv]
    -- The proof for the derivative being positive would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l750_75082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_poly2_factorable_by_common_factors_l750_75036

-- Define the polynomials
def poly1 (n : ℤ) : ℤ := 2*n - 5
def poly2 (a b c : ℤ) : ℤ := a*b + a*c
def poly3 (x : ℤ) : ℤ := x^2 - 4
def poly4 (x y : ℤ) : ℤ := x^2 - 2*x*y + y^2

-- Define what it means for a polynomial to be factorable by common factors
def is_factorable_by_common_factors {α : Type*} [CommRing α] (p : α → α) : Prop :=
  ∃ (q r : α → α), ∀ x, p x = q x * r x ∧ (q x ≠ 1 ∧ q x ≠ -1)

-- State the theorem
theorem only_poly2_factorable_by_common_factors :
  ¬is_factorable_by_common_factors (fun n : ℤ ↦ poly1 n) ∧
  is_factorable_by_common_factors (fun (a : ℤ) ↦ poly2 a 1 1) ∧
  ¬is_factorable_by_common_factors (fun x : ℤ ↦ poly3 x) ∧
  ¬is_factorable_by_common_factors (fun x : ℤ ↦ poly4 x 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_poly2_factorable_by_common_factors_l750_75036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l750_75064

-- Define the function f(x) = ln x - x
noncomputable def f (x : ℝ) : ℝ := Real.log x - x

-- Define the theorem
theorem inequality_holds_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → a * f x ≥ x - (1/2) * x^2) ↔ a ≤ -(1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l750_75064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_distance_theorem_l750_75019

-- Define the type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the circumcenter function
noncomputable def circumcenter (p1 p2 p3 : Point) : Point :=
  sorry -- Implementation not required for the statement

-- Define the set of all points that can be drawn
def drawable_points (A B C : Point) : Set Point :=
  sorry -- Implementation not required for the statement

-- Theorem statement
theorem circumcenter_distance_theorem 
  (A B C : Point) 
  (h_equilateral : distance A B = 6 ∧ distance B C = 6 ∧ distance C A = 6) :
  ∃ (P : Point), P ∈ drawable_points A B C ∧ 
    (∀ (Q : Point), Q ∈ drawable_points A B C → Q ≠ P → distance P Q > 7) ∧
    (∃ (P' : Point), P' ∈ drawable_points A B C ∧ 
      ∀ (Q : Point), Q ∈ drawable_points A B C → Q ≠ P' → distance P' Q > 2019) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_distance_theorem_l750_75019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_and_max_area_l750_75012

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def GivenTriangle (t : Triangle) : Prop :=
  t.b = 4 ∧ (Real.cos t.B / Real.cos t.C = 4 / (2 * t.a - t.c))

/-- Area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := 
  1 / 2 * t.a * t.c * Real.sin t.B

theorem angle_B_measure_and_max_area (t : Triangle) (h : GivenTriangle t) :
  t.B = Real.pi / 3 ∧ 
  ∀ (s : Triangle), GivenTriangle s → area s ≤ 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_and_max_area_l750_75012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_equals_864_sqrt_5_l750_75034

-- Define the given parameters
noncomputable def sector_angle : ℝ := 240
noncomputable def circle_radius : ℝ := 18

-- Define the volume of the cone divided by π
noncomputable def cone_volume_over_pi : ℝ :=
  let base_radius : ℝ := circle_radius * (sector_angle / 360)
  let cone_height : ℝ := Real.sqrt (circle_radius ^ 2 - base_radius ^ 2)
  (1 / 3) * base_radius ^ 2 * cone_height

-- Theorem statement
theorem cone_volume_equals_864_sqrt_5 :
  cone_volume_over_pi = 864 * Real.sqrt 5 := by
  -- Expand the definition of cone_volume_over_pi
  unfold cone_volume_over_pi
  -- Simplify the expression
  simp [sector_angle, circle_radius]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_equals_864_sqrt_5_l750_75034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l750_75062

/-- The function f(x) defined on the positive real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2 + (a + 1) * x + 1

/-- Theorem stating the properties of the function f(x). -/
theorem f_properties :
  -- Part 1: When a = -1, f(x) is monotonically increasing on (1, +∞)
  (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → f (-1) x₁ < f (-1) x₂) ∧
  -- Part 2: f(x) is increasing on (0, +∞) if and only if a ∈ [0, +∞)
  (∀ a : ℝ, (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ↔ 0 ≤ a) ∧
  -- Part 3: When a > 0 and |f(x₁) - f(x₂)| > 2|x₁ - x₂| for any x₁, x₂ ∈ (0, +∞) with x₁ ≠ x₂,
  --         the minimum value of a is 3 - 2√2
  (∀ a : ℝ, a > 0 ∧
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → |f a x₁ - f a x₂| > 2 * |x₁ - x₂|) →
    a ≥ 3 - 2 * Real.sqrt 2) ∧
  (∃ a : ℝ, a > 0 ∧ a = 3 - 2 * Real.sqrt 2 ∧
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → |f a x₁ - f a x₂| > 2 * |x₁ - x₂|)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l750_75062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_numbers_l750_75018

def is_good (n : ℕ) : Prop :=
  1 < n ∧ n < 1979 ∧
  ∀ m : ℕ, 1 < m → m < n → Nat.Coprime m n → Nat.Prime m

theorem good_numbers :
  {n : ℕ | is_good n} = {2, 3, 4, 6, 12, 18, 24, 30} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_numbers_l750_75018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l750_75031

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infiniteGeometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Given an infinite geometric series with common ratio 1/6 and sum 54, its first term is 45 -/
theorem first_term_of_geometric_series :
  ∃ (a : ℝ), infiniteGeometricSum a (1/6) = 54 ∧ a = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l750_75031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l750_75090

open Set
open Real

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_of_inequality (h1 : f 0 = 1) (h2 : ∀ x, 3 * f x = f' x - 3) :
  {x : ℝ | 4 * f x > f' x} = Ioi (log 2 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l750_75090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l750_75041

theorem tan_pi_4_minus_alpha (α : ℝ) 
  (h1 : Real.cos (π + α) = 3/5) 
  (h2 : α > π/2) 
  (h3 : α < π) : 
  Real.tan (π/4 - α) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l750_75041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_ordering_l750_75093

-- Define the function
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := -(k^2 + 10) / x

-- Define the theorem
theorem point_ordering (k : ℝ) (x₁ x₂ x₃ : ℝ) :
  f k x₁ = -3 ∧ f k x₂ = -2 ∧ f k x₃ = 1 → x₃ < x₁ ∧ x₁ < x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_ordering_l750_75093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l750_75046

theorem remainder_theorem (x : ℕ) (h : (5 * x) % 9 = 7) : x % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l750_75046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_h_is_6_l750_75067

noncomputable def sequenceA (n : ℕ) : ℤ :=
  match n with
  | 0 => 20
  | 1 => 100
  | n+2 => 4 * sequenceA (n+1) + 5 * sequenceA n + 20

def divisible_by_1998 (h : ℕ) : Prop :=
  ∀ n, (1998 : ℤ) ∣ (sequenceA (n + h) - sequenceA n)

theorem smallest_h_is_6 :
  ∃ h : ℕ, h > 0 ∧ divisible_by_1998 h ∧ ∀ k, 0 < k → divisible_by_1998 k → h ≤ k :=
sorry

#check smallest_h_is_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_h_is_6_l750_75067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_simplification_l750_75021

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the expression
noncomputable def complex_expression : ℂ := (1 - i)^2 / (1 + i)

-- Theorem statement
theorem complex_simplification : complex_expression = -1 - i := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_simplification_l750_75021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_integers_sum_zero_l750_75009

theorem three_integers_sum_zero (n : ℕ) (S : Finset ℤ) : 
  S.card = 2 * n + 1 → 
  (∀ x ∈ S, |x| ≤ 2 * n - 1) → 
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_integers_sum_zero_l750_75009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_for_g_iteration_l750_75022

def g (x : ℕ) : ℕ :=
  if x % 5 = 0 ∧ x % 6 = 0 then x / 30
  else if x % 6 = 0 then 5 * x
  else if x % 5 = 0 then 6 * x
  else x + 5

def g_iterate : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => g (g_iterate n x)

theorem smallest_b_for_g_iteration :
  ∀ b : ℕ, b > 1 → (g_iterate b 4 = g 4 ↔ b ≥ 7) :=
by
  sorry

#eval g 4
#eval g_iterate 7 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_for_g_iteration_l750_75022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instagram_photo_distribution_l750_75055

/-- Represents the number of photos posted by students in each grade -/
def PhotoDistribution : Type := Fin 5 → ℕ

/-- The problem statement -/
theorem instagram_photo_distribution 
  (photo_dist : PhotoDistribution)
  (h_total_students : (Finset.sum Finset.univ photo_dist) = 50)
  (h_total_photos : (Finset.sum Finset.univ (λ i ↦ photo_dist i * i.val.succ)) = 60)
  (h_min_one_photo : ∀ i, photo_dist i > 0)
  (h_different_grades : ∀ i j, i ≠ j → photo_dist i ≠ photo_dist j) :
  (Finset.filter (λ i ↦ photo_dist i * i.val.succ = photo_dist i) Finset.univ).card = 46 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_instagram_photo_distribution_l750_75055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tree_height_l750_75048

/-- Represents a tree with its height and number of branches -/
structure MyTree where
  height : ℝ
  branches : ℕ

/-- The average number of branches per foot for the trees -/
def averageBranchesPerFoot : ℝ := 4

/-- The four trees Daisy climbed -/
def trees : Fin 4 → MyTree
| 0 => { height := 0, branches := 200 }  -- height is unknown, set to 0
| 1 => { height := 40, branches := 180 }
| 2 => { height := 60, branches := 180 }
| 3 => { height := 34, branches := 153 }

/-- Theorem stating the height of the first tree -/
theorem first_tree_height :
  (trees 0).height = (trees 0).branches / averageBranchesPerFoot := by
  sorry

#check first_tree_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tree_height_l750_75048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_at_45_degree_l750_75006

noncomputable def A : ℝ × ℝ := (6, 4)
noncomputable def B : ℝ × ℝ := (3, 0)
noncomputable def C (k : ℝ) : ℝ × ℝ := (0, k)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def total_distance (k : ℝ) : ℝ :=
  distance A (C k) + distance B (C k)

noncomputable def angle_with_x_axis (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.arctan ((p2.2 - p1.2) / (p2.1 - p1.1))

theorem minimum_distance_at_45_degree :
  ∃ k : ℝ, angle_with_x_axis B (C k) = π/4 ∧
    ∀ k' : ℝ, angle_with_x_axis B (C k') = π/4 →
      total_distance k ≤ total_distance k' ∧
      k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_at_45_degree_l750_75006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l750_75044

theorem angle_in_third_quadrant (θ : Real) :
  Real.tan θ > 0 → Real.sin θ < 0 → ∃ (x y : Real), x < 0 ∧ y < 0 ∧ Real.cos θ = x ∧ Real.sin θ = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l750_75044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dimensions_l750_75098

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The distance between the foci of the ellipse -/
noncomputable def Ellipse.focalDistance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- Theorem: Given an ellipse where a perpendicular line to the x-axis through
    the right focus intersects the ellipse at points A and B, and triangle ABF₁
    is equilateral with perimeter 4√3, then a² = 3 and b² = 2 -/
theorem ellipse_dimensions (e : Ellipse) :
  (∃ (A B : ℝ × ℝ), 
    e.equation A.1 A.2 ∧ 
    e.equation B.1 B.2 ∧
    A.1 = B.1 ∧ 
    A.1 = e.focalDistance / 2 ∧
    (let F₁ : ℝ × ℝ := (-e.focalDistance / 2, 0);
     let perim := Real.sqrt ((A.1 - F₁.1)^2 + A.2^2) +
                   Real.sqrt ((B.1 - F₁.1)^2 + B.2^2) +
                   Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2);
     perim = 4 * Real.sqrt 3 ∧
     (A.1 - F₁.1)^2 + A.2^2 = (B.1 - F₁.1)^2 + B.2^2 ∧
     (A.1 - F₁.1)^2 + A.2^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2)) →
  e.a^2 = 3 ∧ e.b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dimensions_l750_75098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_star_l750_75087

/-- A star formed by overlapping two identical equilateral triangles -/
structure StarShape :=
  (area : ℝ)
  (is_formed_by_equilateral_triangles : Bool)

/-- The shaded region of the star -/
structure ShadedRegion :=
  (num_parts : ℕ)
  (total_parts : ℕ)

/-- Theorem stating the area of the shaded region in the star -/
theorem shaded_area_of_star (s : StarShape) (r : ShadedRegion) : 
  s.area = 36 ∧ 
  s.is_formed_by_equilateral_triangles = true ∧
  r.num_parts = 9 ∧ 
  r.total_parts = 12 → 
  (r.num_parts : ℝ) / (r.total_parts : ℝ) * s.area = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_star_l750_75087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_exist_l750_75040

theorem infinite_solutions_exist :
  ∃ f : ℕ → ℕ × ℕ × ℕ,
    (∀ m : ℕ, 
      let (x, y, z) := f m
      x - y + z = 1 ∧
      x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
      x > 0 ∧ y > 0 ∧ z > 0 ∧
      (x * y) % z = 0 ∧ (y * z) % x = 0 ∧ (x * z) % y = 0) ∧
    (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
by
  -- Define the function f
  let f := λ m : ℕ => (
    m * (m^2 + m - 1),
    (m + 1) * (m^2 + m - 1),
    m * (m + 1)
  )

  -- Prove existence of f
  use f

  -- Split the goal into two parts
  constructor

  -- Part 1: Prove that f satisfies all conditions for each m
  · intro m
    let x := m * (m^2 + m - 1)
    let y := (m + 1) * (m^2 + m - 1)
    let z := m * (m + 1)
    
    constructor
    · -- Prove x - y + z = 1
      sorry
    
    constructor
    · -- Prove x ≠ y ∧ y ≠ z ∧ x ≠ z
      sorry
    
    constructor
    · -- Prove x > 0 ∧ y > 0 ∧ z > 0
      sorry
    
    -- Prove divisibility conditions
    sorry

  -- Part 2: Prove that f is injective
  · intros m n h
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_exist_l750_75040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_measure_of_C_max_perimeter_max_perimeter_achieved_l750_75003

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  (2 * t.a + t.b) * Real.sin t.A + (2 * t.b + t.a) * Real.sin t.B = 2 * t.c * Real.sin t.C

-- Theorem 1: Measure of angle C
theorem measure_of_C (t : Triangle) (h : given_condition t) : t.C = 2 * Real.pi / 3 := by
  sorry

-- Theorem 2: Maximum perimeter when c = √3
theorem max_perimeter (t : Triangle) (h1 : given_condition t) (h2 : t.c = Real.sqrt 3) :
  t.a + t.b + t.c ≤ 2 + Real.sqrt 3 := by
  sorry

-- The equality case for the maximum perimeter
theorem max_perimeter_achieved : ∃ (t : Triangle), 
  given_condition t ∧ t.c = Real.sqrt 3 ∧ t.a + t.b + t.c = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_measure_of_C_max_perimeter_max_perimeter_achieved_l750_75003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_shift_sum_l750_75097

/-- Given a quadratic function f(x) = 3x^2 - 2x + 5, when shifted 6 units to the left,
    the resulting function g(x) = ax^2 + bx + c has coefficients that sum to 138. -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, 3 * (x + 6)^2 - 2 * (x + 6) + 5 = a * x^2 + b * x + c) → 
  a + b + c = 138 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_shift_sum_l750_75097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_shift_l750_75038

/-- The horizontal shift between two sine functions -/
noncomputable def horizontal_shift (f g : ℝ → ℝ) : ℝ :=
  let f' x := 2 * Real.sin (2 * x - Real.pi / 3)
  let g' x := 2 * Real.sin (2 * x + Real.pi / 6)
  (Real.pi / 12) - (-Real.pi / 6)

/-- Theorem stating the horizontal shift between the given sine functions -/
theorem sine_shift : 
  horizontal_shift (λ x => 2 * Real.sin (2 * x - Real.pi / 3)) (λ x => 2 * Real.sin (2 * x + Real.pi / 6)) = Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_shift_l750_75038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_equality_l750_75050

variable {U : Type*}
variable (P S T : Set U)

theorem subset_equality (h : P ∪ (Set.univ \ T) = (Set.univ \ T) ∪ S) :
  (P ∩ T) ∪ S = S := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_equality_l750_75050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_weight_calculation_l750_75025

-- Define the weights of items in pounds
noncomputable def brie_weight : ℚ := 8 / 16  -- 8 ounces converted to pounds
def bread_weight : ℚ := 1
def tomatoes_weight : ℚ := 1
def zucchini_weight : ℚ := 2
noncomputable def raspberries_weight : ℚ := 8 / 16  -- 8 ounces converted to pounds
noncomputable def blueberries_weight : ℚ := 8 / 16  -- 8 ounces converted to pounds
def total_weight : ℚ := 7

-- Theorem to prove
theorem chicken_weight_calculation :
  let other_items_weight := brie_weight + bread_weight + tomatoes_weight + 
                            zucchini_weight + raspberries_weight + blueberries_weight
  let chicken_weight := total_weight - other_items_weight
  chicken_weight = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_weight_calculation_l750_75025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l750_75096

theorem inequality_theorem (f : ℝ → ℝ) (f' : ℝ → ℝ) (a b : ℝ) :
  (∀ x > 0, f x ≥ 0) →
  (∀ x > 0, HasDerivAt f (f' x) x) →
  (∀ x > 0, x * f' x + f x ≤ 0) →
  a > 0 →
  b > 0 →
  a < b →
  a * f b ≤ b * f a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l750_75096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_PM_MQ_l750_75057

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents the square ABCD -/
structure Square where
  sideLength : ℝ
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents the configuration described in the problem -/
structure Configuration where
  square : Square
  E : Point
  M : Point
  P : Point
  Q : Point

/-- The main theorem to be proved -/
theorem ratio_PM_MQ (config : Configuration) : 
  config.square.sideLength = 15 →
  config.E.x = 9 ∧ config.E.y = 0 →
  config.M.x = 9/2 ∧ config.M.y = 15/2 →
  distance config.P config.M / distance config.M config.Q = 53/41 := by
  sorry

/-- Auxiliary lemma: E is 3/5 of the way from D to C -/
lemma E_position (config : Configuration) :
  config.square.sideLength = 15 →
  config.E.x = 9 ∧ config.E.y = 0 := by
  sorry

/-- Auxiliary lemma: M is the midpoint of AE -/
lemma M_midpoint (config : Configuration) :
  config.square.sideLength = 15 →
  config.E.x = 9 ∧ config.E.y = 0 →
  config.M.x = 9/2 ∧ config.M.y = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_PM_MQ_l750_75057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_length_8820_l750_75032

def sequence_terms (n : ℕ) : ℕ :=
  let rec divide_by_two (m : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then m
    else if m % 2 = 0 then divide_by_two (m / 2) (fuel - 1) else m
  let rec divide_by_five (m : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then m
    else if m % 5 = 0 then divide_by_five (m / 5) (fuel - 1) else m
  let after_two := divide_by_two n n
  let final := divide_by_five after_two n
  if final = after_two then
    if n = after_two then 1 else 2
  else
    if n = after_two then 2 else 3

theorem sequence_length_8820 :
  sequence_terms 8820 + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_length_8820_l750_75032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_closed_form_l750_75037

/-- A function f that takes two natural numbers as input -/
def f : ℕ → ℕ → ℕ := sorry

/-- The first condition: f(2, d) ≤ 2^d + 1 -/
axiom f_base_case (d : ℕ) : f 2 d ≤ 2^d + 1

/-- The second condition: For a > 1, f(2^a, d) ≤ f(2, d) + 2(f(2^(a-1), d) - 1) -/
axiom f_recursive_bound (a d : ℕ) (h : a > 1) :
  f (2^a) d ≤ f 2 d + 2 * (f (2^(a-1)) d - 1)

/-- The main theorem to prove -/
theorem f_closed_form (a d : ℕ) (h : a ≥ 1) :
  f (2^a) d = (2^a - 1) * 2^d + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_closed_form_l750_75037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_speed_increase_to_overtake_l750_75095

/-- Represents a runner on the track -/
structure Runner where
  speed : ℝ
  position : ℝ

/-- Represents the track configuration -/
structure Track where
  circumference : ℝ
  straightSection : ℝ

/-- Calculates the time for one runner to overtake another -/
noncomputable def timeToOvertake (chaser : Runner) (chased : Runner) (distance : ℝ) : ℝ :=
  distance / (chaser.speed - chased.speed)

/-- Calculates the time for two runners to meet head-on -/
noncomputable def timeToMeet (runner1 : Runner) (runner2 : Runner) (distance : ℝ) : ℝ :=
  distance / (runner1.speed + runner2.speed)

/-- The main theorem stating the minimum speed increase required -/
theorem minimum_speed_increase_to_overtake 
  (track : Track)
  (runnerA runnerB runnerC : Runner)
  (initialDistanceAB : ℝ)
  (h_track_circumference : track.circumference = 400)
  (h_track_straight : track.straightSection = 200)
  (h_speedB : runnerB.speed = 6)
  (h_speedC : runnerC.speed = 7)
  (h_initial_speedA : runnerA.speed = 8)
  (h_initial_distanceAB : initialDistanceAB = 50)
  : ∃ (speedIncrease : ℝ), 
    speedIncrease = 1.75 ∧ 
    (∀ (ε : ℝ), ε > 0 → 
      timeToOvertake {speed := runnerA.speed + speedIncrease + ε, position := runnerA.position} runnerB initialDistanceAB < 
      timeToMeet {speed := runnerA.speed + speedIncrease + ε, position := runnerA.position} runnerC track.straightSection) ∧
    (∀ (δ : ℝ), δ > 0 → 
      timeToOvertake {speed := runnerA.speed + speedIncrease - δ, position := runnerA.position} runnerB initialDistanceAB > 
      timeToMeet {speed := runnerA.speed + speedIncrease - δ, position := runnerA.position} runnerC track.straightSection) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_speed_increase_to_overtake_l750_75095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_tangent_to_circle_l750_75039

/-- The circle centered at (1, 2) with radius 3 -/
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 9

/-- The parabola x^2 = 4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- The directrix of the parabola -/
def directrix (y : ℝ) : Prop := y = -1

/-- Tangent line touches the circle at exactly one point -/
def is_tangent (y : ℝ) : Prop := ∃! x, my_circle x y

theorem parabola_directrix_tangent_to_circle :
  (∀ x y, parabola x y → directrix y) →
  is_tangent (-1) :=
by
  intro h
  -- The proof goes here
  sorry

#check parabola_directrix_tangent_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_tangent_to_circle_l750_75039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l750_75007

theorem log_inequality (n m : ℝ) (hn : 2 ≤ n) (hm : n < m) (hm3 : m ≤ 3) :
  Real.log (Real.sin (1/m)) / Real.log (Real.cos (1/n)) > 
  Real.log (Real.cos (1/n)) / Real.log (Real.sin (1/m)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l750_75007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_l750_75069

/-- Reflects a point over the y-axis -/
def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Calculates the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem reflection_distance :
  let B : ℝ × ℝ := (1, -5)
  let B' : ℝ × ℝ := reflect_over_y_axis B
  distance B B' = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_l750_75069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_after_five_years_l750_75051

noncomputable def tree_height (n : ℕ) : ℝ :=
  64 / 2^(8 - n)

theorem tree_height_after_five_years :
  tree_height 5 = 8 := by
  -- Unfold the definition of tree_height
  unfold tree_height
  -- Simplify the expression
  simp [pow_sub]
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_height_after_five_years_l750_75051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_extreme_points_l750_75088

open Real

/-- The function f(x) = ln x + x^2 - ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - a*x

/-- Theorem stating the minimum value of f(x1) - f(x2) -/
theorem min_difference_extreme_points (a : ℝ) :
  ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ioc 0 1 → 
  (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → x = x₁ ∨ x = x₂) →
  f a x₁ - f a x₂ ≥ -3/4 + Real.log 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_extreme_points_l750_75088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_theorem_l750_75017

/-- Represents the price of an ice cream cone -/
def Price : Type := Float

/-- Represents the number of cones -/
def Quantity : Type := Nat

/-- Calculates the cost of vanilla cones with the given discount -/
def vanillaCost (price : Float) (quantity : Nat) : Float :=
  let fullPriceCount := quantity - quantity / 3
  let discountedCount := quantity / 3
  (fullPriceCount.toFloat * price) + (discountedCount.toFloat * price / 2)

/-- Calculates the cost of chocolate cones with the given discount -/
def chocolateCost (price : Float) (quantity : Nat) : Float :=
  let paidCount := quantity - quantity / 4
  paidCount.toFloat * price

/-- Calculates the cost of strawberry cones with the given discount -/
def strawberryCost (price : Float) (quantity : Nat) : Float :=
  let fullPriceCount := quantity - quantity / 3
  let discountedCount := quantity / 3
  (fullPriceCount.toFloat * price) + (discountedCount.toFloat * price * 0.75)

/-- The main theorem stating the total cost of Mrs. Hilt's purchase -/
theorem total_cost_theorem (vanillaPrice chocolatePrice strawberryPrice : Float)
    (vanillaQuantity chocolateQuantity strawberryQuantity : Nat) :
    vanillaPrice = 0.99 →
    chocolatePrice = 1.29 →
    strawberryPrice = 1.49 →
    vanillaQuantity = 6 →
    chocolateQuantity = 8 →
    strawberryQuantity = 6 →
    vanillaCost vanillaPrice vanillaQuantity +
    chocolateCost chocolatePrice chocolateQuantity +
    strawberryCost strawberryPrice strawberryQuantity = 20.885 := by
  sorry

#eval vanillaCost 0.99 6 + chocolateCost 1.29 8 + strawberryCost 1.49 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_theorem_l750_75017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_divisor_count_is_ten_l750_75005

/-- The number of divisors of n -/
def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

/-- Whether n has an odd number of divisors -/
def has_odd_divisors (n : ℕ) : Prop := divisor_count n % 2 = 1

/-- The count of numbers up to 100 with an odd number of divisors -/
def odd_divisor_count_up_to_100 : ℕ :=
  Finset.filter (fun n => divisor_count n % 2 = 1) (Finset.range 101) |>.card

theorem odd_divisor_count_is_ten : odd_divisor_count_up_to_100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_divisor_count_is_ten_l750_75005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_side_length_l750_75079

/-- The length of a side of the equilateral triangle -/
noncomputable def s : ℝ := Real.sqrt 2

/-- The area of the equilateral triangle -/
noncomputable def area_equilateral : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The area of one isosceles triangle -/
noncomputable def area_isosceles : ℝ := (1/6) * area_equilateral

/-- The base of an isosceles triangle (equal to the side of the equilateral triangle) -/
noncomputable def base_isosceles : ℝ := s

/-- The height of an isosceles triangle -/
noncomputable def height_isosceles : ℝ := 2 * area_isosceles / base_isosceles

/-- The theorem to be proved -/
theorem isosceles_side_length :
  let side := Real.sqrt ((base_isosceles / 2)^2 + height_isosceles^2)
  side = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_side_length_l750_75079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_satisfies_conditions_l750_75072

def given_plane : (ℝ → ℝ → ℝ → Prop) := λ x y z ↦ 3*x - 2*y + 4*z = 10

def parallel_plane : (ℝ → ℝ → ℝ → Prop) := λ x y z ↦ 3*x - 2*y + 4*z - 32 = 0

def point : ℝ × ℝ × ℝ := (2, -3, 5)

theorem plane_satisfies_conditions :
  (parallel_plane point.1 point.2.1 point.2.2) ∧
  (∀ (x y z : ℝ), given_plane x y z ↔ ∃ (k : ℝ), parallel_plane x y z ∧ k ≠ 0) ∧
  (3 > 0) ∧
  (Nat.gcd (Nat.gcd (Nat.gcd 3 2) 4) 32 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_satisfies_conditions_l750_75072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_p_plus_q_terms_l750_75029

noncomputable section

-- Define an arithmetic sequence
def ArithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Define the sum of the first n terms of an arithmetic sequence
def SumArithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

-- Theorem statement
theorem sum_p_plus_q_terms
  (p q : ℕ) (a₁ d : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p ≠ q)
  (h1 : SumArithmeticSequence a₁ d p = q)
  (h2 : SumArithmeticSequence a₁ d q = p) :
  SumArithmeticSequence a₁ d (p + q) = -(p + q : ℝ) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_p_plus_q_terms_l750_75029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_shipped_theorem_l750_75092

/-- The percentage of units produced that are defective -/
noncomputable def defective_percentage : ℝ := 9

/-- The percentage of defective units that are shipped for sale -/
noncomputable def shipped_percentage : ℝ := 4

/-- The percentage of units produced that are defective units shipped for sale -/
noncomputable def defective_shipped_percentage : ℝ := defective_percentage * shipped_percentage / 100

theorem defective_shipped_theorem : defective_shipped_percentage = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_shipped_theorem_l750_75092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_area_similar_triangle_l750_75066

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumradius of a triangle
noncomputable def circumradius (t : Triangle) : ℝ := sorry

-- Define similarity between two triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ := sorry

-- Define a function to get the maximum perpendicular distance from a point to the sides of a triangle
noncomputable def maxPerpendicularDistance (p : ℝ × ℝ) (t : Triangle) : ℝ := sorry

-- Main theorem
theorem minimum_area_similar_triangle (ABC : Triangle) :
  ∃ (A'B'C' : Triangle),
    (similar ABC A'B'C') ∧
    (∀ (XYZ : Triangle), similar ABC XYZ → area A'B'C' ≤ area XYZ) ∧
    (circumcenter ABC = circumcenter A'B'C') ∧
    (circumradius A'B'C' = maxPerpendicularDistance (circumcenter ABC) ABC) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_area_similar_triangle_l750_75066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_slope_conditions_l750_75083

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.exp x + a * x^2 + b * x

-- Define the derivative of f
noncomputable def f_deriv (a b x : ℝ) : ℝ := Real.exp x + 2 * a * x + b

theorem tangent_line_and_slope_conditions (a b : ℝ) :
  -- Condition: Tangent line at x = 0 passes through (-1, -1)
  (f a b 0 - (-1)) / (0 - (-1)) = f_deriv a b 0 →
  -- Condition: f(0) = 1 (implicit from the problem)
  f a b 0 = 1 →
  -- Prove: b = 1
  b = 1 ∧
  -- Prove: a = -1/2 is the only value that ensures the slope is not less than 2
  (∀ x : ℝ, f_deriv a b x ≥ 2) ↔ a = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_slope_conditions_l750_75083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l750_75043

theorem complex_power_sum (z : ℂ) (h : z + z⁻¹ = 1) : z^1000 + (z^1000)⁻¹ = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l750_75043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_CaH2_molecular_weight_l750_75035

/-- The molecular weight of a compound in grams per mole. -/
noncomputable def molecular_weight (compound : Type) : ℝ := sorry

/-- The number of moles of a compound. -/
noncomputable def moles (compound : Type) : ℝ := sorry

/-- Calcium hydride compound. -/
def CaH2 : Type := Unit

/-- The total weight of a given number of moles of a compound. -/
noncomputable def total_weight (compound : Type) (m : ℝ) : ℝ :=
  m * molecular_weight compound

theorem CaH2_molecular_weight :
  total_weight CaH2 10 = 420 →
  molecular_weight CaH2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_CaH2_molecular_weight_l750_75035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_inequality_l750_75052

theorem increasing_function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f)
  (h : ∀ x ∈ Set.Icc 0 1, (deriv f x) > 0) : 
  f 1 > f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_inequality_l750_75052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l750_75077

def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 6) (m^2 - 3*m + 2)

theorem complex_number_properties (m : ℝ) :
  ((z m).re = 0 → m = -3) ∧
  ((z m).re < 0 ∧ (z m).im > 0 → -3 < m ∧ m < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l750_75077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l750_75001

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the transformed function g
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 2) - 3

-- Theorem statement
theorem graph_translation (f : ℝ → ℝ) (x y : ℝ) :
  y = g f x ↔ y + 3 = f (x + 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l750_75001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_of_C_sum_inverse_squared_distances_l750_75008

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ^2 = 9 / (Real.cos θ^2 + 9 * Real.sin θ^2)

-- Define the transformation from polar to Cartesian coordinates
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Theorem 1: Cartesian equation of curve C
theorem cartesian_equation_of_C :
  ∀ x y : ℝ, (∃ ρ θ : ℝ, C ρ θ ∧ polar_to_cartesian ρ θ = (x, y)) ↔ x^2/9 + y^2 = 1 :=
by sorry

-- Define perpendicularity in polar coordinates
def perpendicular (θ₁ θ₂ : ℝ) : Prop := ∃ k : ℤ, θ₂ = θ₁ + k * Real.pi / 2

-- Theorem 2: Sum of inverse squared distances for perpendicular points
theorem sum_inverse_squared_distances :
  ∀ ρ₁ ρ₂ θ₁ θ₂ : ℝ, 
    C ρ₁ θ₁ → C ρ₂ θ₂ → perpendicular θ₁ θ₂ →
    1 / ρ₁^2 + 1 / ρ₂^2 = 10/9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_of_C_sum_inverse_squared_distances_l750_75008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_range_l750_75061

noncomputable def a_n (a n : ℝ) : ℝ :=
  if n ≤ 5 then (a - 1) ^ (n - 4) else (7 - a) * n - 1

def is_increasing (f : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → f n < f m

theorem increasing_sequence_range (a : ℝ) :
  is_increasing (λ n => a_n a n) → 2 < a ∧ a < 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_range_l750_75061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_integer_lengths_l750_75026

-- Define the triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0

-- Define points D and E
noncomputable def D (triangle : RightTriangle) : ℝ × ℝ := sorry

noncomputable def E (triangle : RightTriangle) : ℝ × ℝ := sorry

-- Define angle bisector property
def is_angle_bisector (A B C D : ℝ × ℝ) : Prop := sorry

-- Define point I as the intersection of BD and CE
noncomputable def I (triangle : RightTriangle) : ℝ × ℝ := sorry

-- Define the length of a segment
noncomputable def segment_length (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem not_all_integer_lengths (triangle : RightTriangle) : 
  is_angle_bisector triangle.B triangle.A triangle.C (D triangle) →
  is_angle_bisector triangle.C triangle.A triangle.B (E triangle) →
  ¬(∃ (n₁ n₂ n₃ n₄ n₅ n₆ : ℕ),
    segment_length triangle.A triangle.B = n₁ ∧
    segment_length triangle.A triangle.C = n₂ ∧
    segment_length triangle.B (I triangle) = n₃ ∧
    segment_length (I triangle) (D triangle) = n₄ ∧
    segment_length triangle.C (I triangle) = n₅ ∧
    segment_length (I triangle) (E triangle) = n₆) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_integer_lengths_l750_75026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_is_factorization_l750_75010

/-- Represents a factorization from left to right -/
def is_factorization (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x ∧ ∃ (h k : ℝ → ℝ), g x = h x * k x

/-- The given expression -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- The factored form -/
def g (x : ℝ) : ℝ := (x - 2)^2

/-- Theorem stating that the given expression represents factorization from left to right -/
theorem expression_is_factorization : is_factorization f g := by
  intro x
  apply And.intro
  · -- Prove f x = g x
    simp [f, g]
    ring
  · -- Prove ∃ (h k : ℝ → ℝ), g x = h x * k x
    use (λ y => y - 2), (λ y => y - 2)
    simp [g]
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_is_factorization_l750_75010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_for_given_tax_l750_75094

/-- Calculates the tax for a given income in country x -/
noncomputable def calculateTax (income : ℝ) : ℝ :=
  if income ≤ 40000 then
    0.14 * income
  else
    0.14 * 40000 + 0.20 * (income - 40000)

/-- Theorem: If the total tax paid is $8,000, then the income is $52,000 -/
theorem income_for_given_tax : 
  ∀ income : ℝ, calculateTax income = 8000 → income = 52000 := by
  intro income
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_for_given_tax_l750_75094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_min_max_cubic_quartic_l750_75042

theorem sum_min_max_cubic_quartic (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) :
  let f := λ (x y z w : ℝ) ↦ 4 * (x^3 + y^3 + z^3 + w^3) - (x^4 + y^4 + z^4 + w^4)
  ∃ (m M : ℝ), (∀ (x y z w : ℝ), x + y + z + w = 6 → x^2 + y^2 + z^2 + w^2 = 12 → 
    m ≤ f x y z w ∧ f x y z w ≤ M) ∧ m + M = 84 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_min_max_cubic_quartic_l750_75042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_even_factors_720_l750_75058

theorem sum_of_even_factors_720 :
  ∃ (n : ℕ) (prime_factorization : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) (sum_even_factors : ℕ),
    n = 720 ∧
    prime_factorization = (2, 4, 3, 2, 5, 1) ∧
    sum_even_factors = (2^1 + 2^2 + 2^3 + 2^4) * (3^0 + 3^1 + 3^2) * (5^0 + 5^1) ∧
    n = 2^4 * 3^2 * 5^1 ∧
    sum_even_factors = 2340 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_even_factors_720_l750_75058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lorin_has_four_black_marbles_l750_75099

/-- The number of black marbles Lorin has -/
def lorins_black_marbles : ℕ := sorry

/-- The number of yellow marbles Jimmy has -/
def jimmys_yellow_marbles : ℕ := 22

/-- The total number of marbles Alex has -/
def alexs_total_marbles : ℕ := 19

theorem lorin_has_four_black_marbles :
  (2 * lorins_black_marbles + jimmys_yellow_marbles / 2 = alexs_total_marbles) →
  lorins_black_marbles = 4 := by
  intro h
  sorry

#check lorin_has_four_black_marbles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lorin_has_four_black_marbles_l750_75099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l750_75065

open Real

/-- The horizontal shift required to transform the graph of y = 3sin(2x + π/5) 
    into the graph of y = 3sin(2x - π/5) -/
noncomputable def horizontal_shift : ℝ := π / 5

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x + π / 5)

/-- The transformed function -/
noncomputable def g (x : ℝ) : ℝ := 3 * sin (2 * x - π / 5)

theorem graph_transformation (x : ℝ) : 
  f (x - horizontal_shift) = g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l750_75065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l750_75030

def is_valid_subset (T : Finset ℕ) : Prop :=
  ∀ x ∈ T, ∀ y ∈ T, x ≠ y → ¬(11 ∣ (x + y))

theorem max_subset_size :
  ∃ (T : Finset ℕ), T ⊆ Finset.range 100 ∧ is_valid_subset T ∧ T.card = 60 ∧
    ∀ (S : Finset ℕ), S ⊆ Finset.range 100 → is_valid_subset S → S.card ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l750_75030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l750_75045

/-- The number of circular arcs in the curve -/
def num_arcs : ℕ := 10

/-- The length of each circular arc -/
noncomputable def arc_length : ℝ := Real.pi / 2

/-- The side length of the regular pentagon -/
def pentagon_side : ℝ := 1

/-- The area of a regular pentagon with side length s -/
noncomputable def pentagon_area (s : ℝ) : ℝ := 
  (1 / 4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) * s^2

/-- The theorem statement -/
theorem enclosed_area_theorem :
  let r := arc_length / Real.pi  -- radius of the circular arcs
  let sector_area := num_arcs * (Real.pi * r^2 * (arc_length / (2 * Real.pi)))
  let total_area := pentagon_area pentagon_side + sector_area
  total_area = (1 / 4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) + 5 * Real.pi^2 / 4 := by
  sorry

#check enclosed_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l750_75045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_2_proposition_4_l750_75056

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (belongs_to : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Proposition (2)
theorem proposition_2 
  (α β γ : Plane) (m n : Line) :
  parallel_plane α β →
  intersect α γ m →
  intersect β γ n →
  parallel m n :=
sorry

-- Proposition (4)
theorem proposition_4
  (α β : Plane) (m n : Line) :
  intersect α β m →
  parallel n m →
  ¬ belongs_to n α →
  ¬ belongs_to n β →
  perpendicular n α ∧ perpendicular n β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_2_proposition_4_l750_75056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l750_75063

-- Define the revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then
    10.8 - (1/30) * x^2
  else if x > 10 then
    108/x - 1000/(3*x^2)
  else
    0

-- Define the profit function
noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then
    x * R x - (10 + 2.7 * x)
  else if x > 10 then
    x * R x - (10 + 2.7 * x)
  else
    0

-- Theorem statement
theorem max_profit_at_nine :
  ∃ (max_profit : ℝ), max_profit = 38.6 ∧
  ∀ (x : ℝ), W x ≤ max_profit ∧
  W 9 = max_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l750_75063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_problem_l750_75059

/-- The probability of getting heads in a single flip of the biased coin -/
noncomputable def h : ℝ := 3 / 8

/-- The probability of getting exactly k heads in n flips -/
noncomputable def prob_k_heads (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * h^k * (1 - h)^(n - k)

theorem biased_coin_problem :
  prob_k_heads 7 2 = prob_k_heads 7 3 ∧
  prob_k_heads 7 4 = 27 / 160 ∧
  (let p := 27
   let q := 160
   p + q = 187) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_problem_l750_75059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vector_l750_75033

noncomputable def vector_a : ℝ × ℝ := (1, -Real.sqrt 3)
noncomputable def vector_b : ℝ × ℝ := (Real.sqrt 3 / 2, 1 / 2)

theorem perpendicular_unit_vector :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 0) ∧
  (vector_b.1^2 + vector_b.2^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vector_l750_75033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_regular_pyramid_l750_75070

/-- The radius of a sphere touching all edges of a regular quadrilateral pyramid -/
noncomputable def sphere_radius (a b : ℝ) : ℝ :=
  a * (2 * b - a) / (2 * Real.sqrt (2 * b^2 - a^2))

/-- Theorem: The radius of a sphere touching all edges of a regular quadrilateral pyramid 
    with base side length a and slant edge b is given by a(2b - a) / (2 √(2b² - a²)) -/
theorem sphere_radius_regular_pyramid (a b : ℝ) (ha : a > 0) (hb : b > a / Real.sqrt 2) :
  sphere_radius a b = a * (2 * b - a) / (2 * Real.sqrt (2 * b^2 - a^2)) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_regular_pyramid_l750_75070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_drawing_A_l750_75086

def factory_ratio : ℚ := 2 / 3
def sample_size_A : ℕ := 10
def draw_size : ℕ := 2

theorem probability_of_drawing_A :
  let sample_size_B : ℕ := (sample_size_A * 3 / 2 : ℕ)
  let total_sample_size : ℕ := sample_size_A + sample_size_B
  let prob_A : ℚ := (Nat.choose sample_size_A 2 + sample_size_A * sample_size_B : ℚ) / Nat.choose total_sample_size 2
  prob_A = 39 / 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_drawing_A_l750_75086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_bound_l750_75024

/-- A structure representing a tetrahedron with its circumscribed sphere -/
structure TetrahedronWithCircumscribedSphere where
  volume : ℝ
  circumscribedSphereRadius : ℝ
  volume_pos : volume > 0
  radius_pos : circumscribedSphereRadius > 0

/-- The volume of a tetrahedron is bounded by a function of its circumscribed sphere's radius -/
theorem tetrahedron_volume_bound (V R : ℝ) 
  (hV : V > 0) -- V is positive (volume)
  (hR : R > 0) -- R is positive (radius)
  (h : ∃ (t : TetrahedronWithCircumscribedSphere), 
    t.volume = V ∧ t.circumscribedSphereRadius = R) : 
  V ≤ (8 / (9 * Real.sqrt 3)) * R^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_bound_l750_75024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l750_75085

/-- The number of bounces needed for a ball to reach a maximum height less than 2 feet,
    given an initial height of 45 feet and a bounce ratio of 1/3. -/
theorem ball_bounce_count : ∃ k : ℕ, 
  (∀ n : ℕ, n < k → 45 * (1/3 : ℝ)^n ≥ 2) ∧ 
  (45 * (1/3 : ℝ)^k < 2) ∧
  k = 4 := by
  sorry

#check ball_bounce_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l750_75085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l750_75023

theorem inequality_range (a : ℝ) : 
  (∀ n : ℕ+, (-2:ℝ)^(n:ℕ) * a - 3^((n:ℕ)-1) - (-2:ℝ)^(n:ℕ) < 0) → 
  (1/2 < a ∧ a < 7/4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l750_75023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_range_l750_75076

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

theorem vector_sum_magnitude_range :
  ∀ θ : ℝ, 0 ≤ (a.1 + 2 * (b θ).1)^2 + (a.2 + 2 * (b θ).2)^2 ∧
           (a.1 + 2 * (b θ).1)^2 + (a.2 + 2 * (b θ).2)^2 ≤ 16 :=
by
  intro θ
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_range_l750_75076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_origin_l750_75071

noncomputable section

-- Define the parabolas and line
def parabola1 (x : ℝ) : ℝ := x^3
def parabola2 (x : ℝ) : ℝ := x^2
def line (a b c x : ℝ) : ℝ := -(a * x + c) / b

-- Define the intersection points
def intersection_points (a b c : ℝ) : Set ℝ :=
  {x : ℝ | parabola1 x = line a b c x}

-- Define the projection function
def project (x : ℝ) : ℝ × ℝ := (x, parabola2 x)

-- Define the circle passing through three points
def circle_through_points (p1 p2 p3 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (A B : ℝ), p.1^2 + p.2^2 + A * p.1 + B * p.2 = 0 ∧
                             p1.1^2 + p1.2^2 + A * p1.1 + B * p1.2 = 0 ∧
                             p2.1^2 + p2.2^2 + A * p2.1 + B * p2.2 = 0 ∧
                             p3.1^2 + p3.2^2 + A * p3.1 + B * p3.2 = 0}

theorem circle_through_origin (a b c : ℝ) :
  ∀ (x1 x2 x3 : ℝ), x1 ∈ intersection_points a b c →
                     x2 ∈ intersection_points a b c →
                     x3 ∈ intersection_points a b c →
                     x1 ≠ x2 → x2 ≠ x3 → x3 ≠ x1 →
                     (0, 0) ∈ circle_through_points (project x1) (project x2) (project x3) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_origin_l750_75071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_thirds_pi_plus_two_alpha_l750_75054

theorem cos_two_thirds_pi_plus_two_alpha (α : ℝ) :
  Real.sin (π / 6 - α) = 1 / 3 → Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_thirds_pi_plus_two_alpha_l750_75054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocer_decaf_percentage_l750_75084

/-- Represents the coffee stock of a grocer -/
structure CoffeeStock where
  initial : ℝ
  initialDecaf : ℝ
  additional : ℝ
  additionalDecaf : ℝ

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
noncomputable def decafPercentage (stock : CoffeeStock) : ℝ :=
  ((stock.initialDecaf * stock.initial + stock.additionalDecaf * stock.additional) /
   (stock.initial + stock.additional)) * 100

/-- Theorem stating that given the initial conditions, the percentage of decaffeinated coffee is 30% -/
theorem grocer_decaf_percentage (stock : CoffeeStock)
  (h1 : stock.initial = 400)
  (h2 : stock.initialDecaf = 0.20)
  (h3 : stock.additional = 100)
  (h4 : stock.additionalDecaf = 0.70) :
  decafPercentage stock = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocer_decaf_percentage_l750_75084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_proof_l750_75011

theorem triangle_cosine_proof (a b c : ℝ) (A B C : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Condition: 8b = 5c
  8 * b = 5 * c →
  -- Condition: C = 2B
  C = 2 * B →
  -- Conclusion: cos C = 7/25
  Real.cos C = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_proof_l750_75011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l750_75020

/-- The number of ones in the original expression -/
def num_ones : ℕ := 2013

/-- The optimal group size for maximizing the expression -/
def optimal_group_size : ℕ := 3

/-- The number of optimal groups -/
def num_groups : ℕ := num_ones / optimal_group_size

/-- The maximum value achievable by grouping the ones -/
def max_value : ℕ := optimal_group_size ^ num_groups

theorem max_expression_value :
  ∀ (grouping : List ℕ),
    (grouping.sum = num_ones) →
    (grouping.prod ≤ max_value) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l750_75020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_comparison_l750_75016

-- Define the integrals
noncomputable def S₁ : ℝ := ∫ x in (1:ℝ)..2, x^2

noncomputable def S₂ : ℝ := ∫ x in (1:ℝ)..2, 1/x

noncomputable def S₃ : ℝ := ∫ x in (1:ℝ)..2, Real.exp x

-- State the theorem
theorem integral_comparison : S₂ < S₁ ∧ S₁ < S₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_comparison_l750_75016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_sphere_radius_in_tetrahedron_l750_75075

/-- Volume of the part of a regular tetrahedron with edge length a that is outside a sphere
    with radius r centered at the center of the tetrahedron. -/
def volume_tetrahedron_outside_sphere (a r : ℝ) : ℝ :=
  sorry

/-- Volume of the part of a sphere with radius r that is outside a regular tetrahedron
    with edge length a, where the sphere is centered at the center of the tetrahedron. -/
def volume_sphere_outside_tetrahedron (a r : ℝ) : ℝ :=
  sorry

/-- Given a regular tetrahedron with edge length a, the radius of a sphere centered at the center
of the tetrahedron that minimizes the total volume of the part of the tetrahedron outside the sphere
and the part of the sphere outside the tetrahedron is equal to a * (1/3) * √(2/3). -/
theorem optimal_sphere_radius_in_tetrahedron (a : ℝ) (ha : a > 0) :
  ∃ (r : ℝ), r = a * (1/3) * Real.sqrt (2/3) ∧
  r > 0 ∧
  ∀ (x : ℝ), x > 0 →
    (volume_tetrahedron_outside_sphere a x + volume_sphere_outside_tetrahedron a x) ≥
    (volume_tetrahedron_outside_sphere a r + volume_sphere_outside_tetrahedron a r) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_sphere_radius_in_tetrahedron_l750_75075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_sales_theorem_l750_75068

structure BreadSales where
  cost : ℝ
  original_price : ℝ
  promo_prices : Fin 4 → ℝ
  standard_sales : ℕ
  sales_changes : Fin 4 → ℤ

def BreadSales.actual_sales (bs : BreadSales) (week : Fin 4) : ℤ :=
  (bs.standard_sales : ℤ) + bs.sales_changes week

def BreadSales.weekly_profit (bs : BreadSales) (week : Fin 4) : ℝ :=
  (bs.actual_sales week : ℝ) * (bs.promo_prices week - bs.cost)

def BreadSales.total_profit (bs : BreadSales) : ℝ :=
  (Finset.sum Finset.univ fun w => bs.weekly_profit w)

def SchemeOneProfit (bs : BreadSales) (pieces : ℕ) : ℝ :=
  (pieces : ℝ) * (bs.original_price - bs.cost - 0.3)

def SchemeTwoProfit (bs : BreadSales) (pieces : ℕ) : ℝ :=
  let discounted := max (pieces - 3) 0
  3 * bs.original_price + (discounted : ℝ) * (bs.original_price * 0.9) - (pieces : ℝ) * bs.cost

theorem bread_sales_theorem (bs : BreadSales) 
    (h_cost : bs.cost = 3.5)
    (h_original : bs.original_price = 6)
    (h_promo : bs.promo_prices = ![4.5, 5, 5.5, 6])
    (h_standard : bs.standard_sales = 200)
    (h_changes : bs.sales_changes = ![28, 16, -6, -12]) :
    (∀ w : Fin 4, bs.sales_changes w ≥ bs.sales_changes 3) ∧ 
    (bs.actual_sales 2 : ℝ) * bs.promo_prices 2 = 1067 ∧
    bs.total_profit = 1410 ∧
    SchemeOneProfit bs 7 > SchemeTwoProfit bs 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_sales_theorem_l750_75068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_scientists_theorem_l750_75078

/-- Represents the maximum number of scientists that could work in a research institute
    given the cafeteria visiting constraints. -/
noncomputable def max_scientists (x : ℝ) : ℕ :=
  (2 * ⌊x / (2 * x - 8)⌋).toNat

/-- Theorem stating the maximum number of scientists satisfying the cafeteria constraint. -/
theorem max_scientists_theorem (x : ℝ) (hx : x > 4) :
  ∀ n : ℕ,
  (∀ i j : Fin n, i ≠ j →
    ∃ t : ℝ → Fin n → Bool,
    (∀ s, (t s i ≠ t s j) → ∫ s in (0 : ℝ)..8, (if t s i ≠ t s j then 1 else 0) ≥ x)) →
  n ≤ max_scientists x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_scientists_theorem_l750_75078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l750_75028

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / a - 1 / x

-- State the theorem
theorem function_properties (a : ℝ) (h_a : a > 0) :
  -- Part 1: f is increasing on (0, +∞)
  (∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  -- Part 2: If f(1/2) = 1/2 and f(2) = 2, then a = 2/5
  (f a (1/2) = 1/2 ∧ f a 2 = 2 → a = 2/5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l750_75028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l750_75053

/-- The maximum area of an equilateral triangle inscribed in a 12 by 15 rectangle -/
theorem max_equilateral_triangle_area_in_rectangle : 
  ∃ (A : ℝ), 
    (∀ (a : ℝ), 
      (∃ (x y : ℝ), 
        0 ≤ x ∧ x ≤ 12 ∧ 
        0 ≤ y ∧ y ≤ 15 ∧
        a = (Real.sqrt 3 / 4) * (x^2 + y^2)) → 
      a ≤ A) ∧ 
    A = 261 * Real.sqrt 3 - 540 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l750_75053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l750_75002

/-- Given an increasing function f: ℝ → ℝ, we define g(x, y) as specified. -/
noncomputable def g (f : ℝ → ℝ) (x y : ℝ) : ℝ := (f (x + y) - f x) / (f x - f (x - y))

/-- Main theorem: If f is increasing and g satisfies certain bounds for specific x and y,
    then g satisfies wider bounds for all x and y > 0. -/
theorem g_bounds (f : ℝ → ℝ) (h_incr : Monotone f) 
    (h_bound : ∀ x y, (x = 0 ∧ y > 0) ∨ (x ≠ 0 ∧ 0 < y ∧ y ≤ |x|) → 
               1/2 < g f x y ∧ g f x y < 2) :
    ∀ x y, y > 0 → 1/14 < g f x y ∧ g f x y < 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l750_75002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rational_roots_65_power_l750_75027

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- Converts a natural number to its decimal representation as a polynomial -/
def toDecimalPolynomial (m : ℕ) : Σ n, IntPolynomial n :=
  sorry

/-- Checks if a polynomial has rational roots -/
def hasRationalRoot (p : Σ n, IntPolynomial n) : Prop :=
  sorry

/-- The main theorem: The polynomial formed by the digits of 65^k (k ≥ 2) has no rational roots -/
theorem no_rational_roots_65_power (k : ℕ) (h : k ≥ 2) :
  ¬ hasRationalRoot (toDecimalPolynomial (65^k)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rational_roots_65_power_l750_75027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_yield_is_eight_percent_l750_75004

-- Define the stock parameters
noncomputable def stock_quote : ℝ := 250
noncomputable def stock_percentage : ℝ := 20
noncomputable def face_value : ℝ := 100

-- Define the annual dividend
noncomputable def annual_dividend : ℝ := stock_percentage / 100 * face_value

-- Define the yield percentage calculation
noncomputable def yield_percentage : ℝ := annual_dividend / stock_quote * 100

-- Theorem statement
theorem stock_yield_is_eight_percent : yield_percentage = 8 := by
  -- Unfold definitions
  unfold yield_percentage annual_dividend stock_quote stock_percentage face_value
  -- Simplify the expression
  simp [mul_div_assoc, mul_comm, mul_assoc]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_yield_is_eight_percent_l750_75004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_three_most_economical_l750_75047

theorem base_three_most_economical (d : ℕ) (m : ℕ) (h : 3 ∣ m ∧ d ∣ m) (hd : d > 0) :
  (3 : ℝ) ^ (m / 3 : ℝ) ≥ (d : ℝ) ^ (m / d : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_three_most_economical_l750_75047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l750_75015

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 12x -/
def isOnParabola (p : Point) : Prop :=
  p.y^2 = 12 * p.x

/-- The focus of the parabola y² = 12x -/
def focus : Point :=
  { x := 3, y := 0 }

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Main theorem -/
theorem parabola_chord_length (A B : Point) :
  isOnParabola A → isOnParabola B →
  collinear A B focus →
  A.x + B.x = 6 →
  distance A B = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l750_75015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l750_75049

-- Define the given constants
def train_speed : ℝ := 102  -- km/h
def crossing_time : ℝ := 50  -- seconds
def platform_length : ℝ := 396.78  -- meters

-- Define the theorem
theorem train_length_calculation :
  let speed_ms : ℝ := train_speed * 1000 / 3600  -- Convert km/h to m/s
  let total_distance : ℝ := speed_ms * crossing_time
  let train_length : ℝ := total_distance - platform_length
  ∃ ε > 0, |train_length - 1019.89| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l750_75049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_sqrt_5_l750_75073

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a parabola of the form y = x^2 + 1 -/
def Parabola := Unit

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- Theorem stating that under the given conditions, the hyperbola's eccentricity is √5 -/
theorem hyperbola_eccentricity_is_sqrt_5 (h : Hyperbola) (p : Parabola) :
  (∃ (x y : ℝ), x^2 / h.a^2 - y^2 / h.b^2 = 1 ∧ y = x^2 + 1) →
  (∃! (x : ℝ), ∃ (y : ℝ), y = h.b / h.a * x ∧ y = x^2 + 1) →
  eccentricity h = Real.sqrt 5 := by
  sorry

#check hyperbola_eccentricity_is_sqrt_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_sqrt_5_l750_75073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_home_electronics_allocation_l750_75060

/-- Represents the budget allocation for Megatech Corporation's research and development --/
structure BudgetAllocation where
  microphotonics : ℚ
  food_additives : ℚ
  genetically_modified_microorganisms : ℚ
  industrial_lubricants : ℚ
  basic_astrophysics_degrees : ℚ

/-- The total percentage of a circle --/
def total_percentage : ℚ := 100

/-- The total degrees in a circle --/
def total_degrees : ℚ := 360

/-- Calculates the percentage of the budget allocated to home electronics --/
def home_electronics_percentage (b : BudgetAllocation) : ℚ :=
  total_percentage - (b.microphotonics + b.food_additives + b.genetically_modified_microorganisms + b.industrial_lubricants + (b.basic_astrophysics_degrees / total_degrees * total_percentage))

/-- Theorem stating that the percentage allocated to home electronics is 19% --/
theorem home_electronics_allocation (b : BudgetAllocation) 
  (h1 : b.microphotonics = 14)
  (h2 : b.food_additives = 10)
  (h3 : b.genetically_modified_microorganisms = 24)
  (h4 : b.industrial_lubricants = 8)
  (h5 : b.basic_astrophysics_degrees = 90) :
  home_electronics_percentage b = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_home_electronics_allocation_l750_75060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_composite_l750_75014

/-- A function that replaces three adjacent digits in a natural number -/
def replaceThreeDigits (n : ℕ) (start : ℕ) (replacement : ℕ) : ℕ :=
  sorry

/-- Predicate to check if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ a b, a > 1 ∧ b > 1 ∧ a * b = n

/-- The main theorem statement -/
theorem exists_special_composite : ∃ n : ℕ, 
  (Nat.digits 10 n).length = 2021 ∧ 
  isComposite n ∧ 
  (∀ start replacement, start + 3 ≤ 2021 → 
    isComposite (replaceThreeDigits n start replacement)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_composite_l750_75014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_paradox_possible_l750_75091

/-- Represents the possible orderings of three runners in a race -/
inductive RaceOutcome
  | ABC
  | ACB
  | BAC
  | BCA
  | CAB
  | CBA

/-- Checks if runner1 beats runner2 in the given race outcome -/
def beats (runner1 runner2 : Fin 3) (outcome : RaceOutcome) : Bool :=
  match outcome, runner1, runner2 with
  | RaceOutcome.ABC, 0, 1 => true
  | RaceOutcome.ABC, 0, 2 => true
  | RaceOutcome.ABC, 1, 2 => true
  | RaceOutcome.ACB, 0, 1 => true
  | RaceOutcome.ACB, 0, 2 => true
  | RaceOutcome.ACB, 2, 1 => true
  | RaceOutcome.BAC, 1, 0 => true
  | RaceOutcome.BAC, 1, 2 => true
  | RaceOutcome.BAC, 0, 2 => true
  | RaceOutcome.BCA, 1, 0 => true
  | RaceOutcome.BCA, 1, 2 => true
  | RaceOutcome.BCA, 2, 0 => true
  | RaceOutcome.CAB, 2, 0 => true
  | RaceOutcome.CAB, 2, 1 => true
  | RaceOutcome.CAB, 0, 1 => true
  | RaceOutcome.CBA, 2, 0 => true
  | RaceOutcome.CBA, 2, 1 => true
  | RaceOutcome.CBA, 1, 0 => true
  | _, _, _ => false

/-- Counts how many times runner1 beats runner2 in a list of race outcomes -/
def countBeats (runner1 runner2 : Fin 3) (races : List RaceOutcome) : Nat :=
  races.filter (beats runner1 runner2) |>.length

/-- Theorem: There exists a set of race outcomes where each runner outperforms another in more than half of the races -/
theorem runners_paradox_possible : ∃ (races : List RaceOutcome),
  (races.length > 0) ∧
  (countBeats 0 1 races > races.length / 2) ∧
  (countBeats 1 2 races > races.length / 2) ∧
  (countBeats 2 0 races > races.length / 2) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_paradox_possible_l750_75091
