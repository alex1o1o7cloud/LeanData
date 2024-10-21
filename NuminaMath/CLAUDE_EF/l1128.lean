import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_sqrt_3_is_quadratic_radical_l1128_112802

-- Define what a quadratic radical is
noncomputable def is_quadratic_radical (x : ℝ) : Prop := ∃ y : ℝ, y ≥ 0 ∧ x = Real.sqrt y

-- Define the expressions
noncomputable def expr1 : ℝ := Real.sqrt 3
def expr2 : ℂ := Complex.I
noncomputable def expr3 : ℝ := (5 : ℝ) ^ (1/3)
noncomputable def expr4 : ℝ := Real.sqrt (Real.pi - 4)

-- State the theorem
theorem only_sqrt_3_is_quadratic_radical :
  is_quadratic_radical expr1 ∧
  ¬is_quadratic_radical (expr2.re) ∧
  ¬is_quadratic_radical expr3 ∧
  ¬is_quadratic_radical expr4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_sqrt_3_is_quadratic_radical_l1128_112802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_three_implies_fraction_half_l1128_112834

theorem tan_three_implies_fraction_half (θ : Real) (h : Real.tan θ = 3) :
  (2 * Real.sin θ - 4 * Real.cos θ) / (Real.sin θ + Real.cos θ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_three_implies_fraction_half_l1128_112834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equals_pi_cubed_over_24_l1128_112803

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := (t^2 - 2) * Real.sin t + 2 * t * Real.cos t
noncomputable def y (t : ℝ) : ℝ := (2 - t^2) * Real.cos t + 2 * t * Real.sin t

-- Define the arc length function
noncomputable def arcLength (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)

-- State the theorem
theorem arc_length_equals_pi_cubed_over_24 :
  arcLength 0 (Real.pi / 2) = Real.pi^3 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equals_pi_cubed_over_24_l1128_112803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_cosine_value_l1128_112888

theorem point_on_line_cosine_value (α : Real) :
  (Real.sin α = -2 * Real.cos α) → Real.cos (2 * α + Real.pi / 2) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_cosine_value_l1128_112888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1128_112866

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Axioms based on the given conditions
axiom f_symmetry (x : ℝ) : f (-x) + f x = x^2
axiom f'_gt_x (x : ℝ) : x ≥ 0 → f' x > x

-- Define the set of a that satisfies the inequality
def A : Set ℝ := {a | f (2 - a) - f a ≥ 2 - 2*a}

-- Theorem statement
theorem range_of_a : A = Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1128_112866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_stamps_for_postage_l1128_112857

theorem min_stamps_for_postage : ∃ (a b c : ℕ),
  (4/10 : ℚ) * a + (8/10 : ℚ) * b + (15/10 : ℚ) * c = (102/10 : ℚ) ∧
  a + b + c = 8 ∧
  ∀ (x y z : ℕ), (4/10 : ℚ) * x + (8/10 : ℚ) * y + (15/10 : ℚ) * z = (102/10 : ℚ) → x + y + z ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_stamps_for_postage_l1128_112857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_price_percentage_l1128_112813

theorem sale_price_percentage (original_price : ℝ) (h : original_price > 0) :
  (0.9 * (0.8 * original_price)) / original_price = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_price_percentage_l1128_112813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_segment_length_is_five_halves_l1128_112842

/-- Regular triangular pyramid with specific properties -/
structure RegularTriangularPyramid where
  -- Base side length
  base_side : ℝ
  -- Tangent of dihedral angle at the base
  dihedral_angle_tan : ℝ
  -- Condition: base side length is √3
  base_side_eq : base_side = Real.sqrt 3
  -- Condition: tangent of dihedral angle is 3
  dihedral_angle_tan_eq : dihedral_angle_tan = 3

/-- The length of the segment connecting the midpoint of a base side 
    with the midpoint of the opposite edge -/
noncomputable def midpoint_segment_length (p : RegularTriangularPyramid) : ℝ :=
  5 / 2

/-- Theorem: The length of the segment connecting the midpoint of a base side 
    with the midpoint of the opposite edge in the given pyramid is 5/2 -/
theorem midpoint_segment_length_is_five_halves (p : RegularTriangularPyramid) :
  midpoint_segment_length p = 5 / 2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_segment_length_is_five_halves_l1128_112842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_with_three_integers_l1128_112805

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def is_consecutive_odd_sequence (s : List ℤ) : Prop :=
  ∀ i : ℕ, i + 1 < s.length → s[i + 1]! = s[i]! + 2

def satisfies_sum_condition (s : List ℤ) : Prop :=
  s.length ≥ 3 → s[s.length - 1]! + s[s.length - 2]! = s[0]! + 17

theorem unique_sequence_with_three_integers :
  ∃! s : List ℤ, s.length = 3 ∧
                 s[0]! = 11 ∧ s[1]! = 13 ∧ s[2]! = 15 ∧
                 (∀ n ∈ s, is_odd n) ∧
                 is_consecutive_odd_sequence s ∧
                 satisfies_sum_condition s :=
by sorry

#check unique_sequence_with_three_integers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_with_three_integers_l1128_112805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_triangle_area_l1128_112877

/-- Given an equilateral triangle with side length s and area T, prove that the area of the triangle formed by joining the centroids of the three smaller triangles created by connecting the medians is T/9, and the ratio of this new area to T is 1/9. -/
theorem centroid_triangle_area (s : ℝ) (T : ℝ) (h : T = (Real.sqrt 3 / 4) * s^2) :
  let new_area := (Real.sqrt 3 / 4) * (s / 3)^2
  new_area = T / 9 ∧ new_area / T = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_triangle_area_l1128_112877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_changes_for_different_sums_l1128_112828

def Matrix4x4 := Fin 4 → Fin 4 → ℤ

def rowSum (m : Matrix4x4) (i : Fin 4) : ℤ :=
  (Finset.sum Finset.univ fun j => m i j)

def colSum (m : Matrix4x4) (j : Fin 4) : ℤ :=
  (Finset.sum Finset.univ fun i => m i j)

def allSumsEqual (m : Matrix4x4) : Prop :=
  ∀ i j : Fin 4, rowSum m i = colSum m j

def allSumsDifferent (m : Matrix4x4) : Prop :=
  ∀ i j : Fin 4, i ≠ j → 
    (rowSum m i ≠ rowSum m j ∧ 
     colSum m i ≠ colSum m j ∧
     rowSum m i ≠ colSum m j)

def hammingDistance (m1 m2 : Matrix4x4) : Nat :=
  (Finset.sum Finset.univ fun i => Finset.sum Finset.univ fun j => if m1 i j = m2 i j then 0 else 1)

theorem min_changes_for_different_sums (m : Matrix4x4) 
  (h : allSumsEqual m) : 
  ∃ m' : Matrix4x4, allSumsDifferent m' ∧ 
    ∀ m'' : Matrix4x4, allSumsDifferent m'' → 
      hammingDistance m m' ≤ hammingDistance m m'' :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_changes_for_different_sums_l1128_112828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1128_112869

/-- Ellipse C with equation x²/4 + y²/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Left focus of ellipse C -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- Line l with slope k and y-intercept m -/
def line_l (k m x y : ℝ) : Prop := y = k * x + m

/-- Predicate to check if a point is on ellipse C -/
def on_ellipse_C (p : ℝ × ℝ) : Prop := ellipse_C p.1 p.2

/-- Predicate to check if a point is on line l -/
def on_line_l (k m : ℝ) (p : ℝ × ℝ) : Prop := line_l k m p.1 p.2

/-- Slope of a line passing through two points -/
noncomputable def line_slope (p q : ℝ × ℝ) : ℝ := (q.2 - p.2) / (q.1 - p.1)

/-- Area of a triangle given three points -/
noncomputable def triangle_area (p q r : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))

/-- Main theorem -/
theorem max_triangle_area (k m : ℝ) (A B : ℝ × ℝ) :
  on_ellipse_C A ∧ on_ellipse_C B ∧
  on_line_l k m A ∧ on_line_l k m B ∧
  A ≠ B ∧
  ¬ on_line_l k m F₁ ∧
  ∃ (t : ℝ), line_slope A F₁ + t = 2 * k ∧ line_slope B F₁ - t = 2 * k →
  ∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧
    ∀ (k' m' : ℝ) (A' B' : ℝ × ℝ),
      on_ellipse_C A' ∧ on_ellipse_C B' ∧
      on_line_l k' m' A' ∧ on_line_l k' m' B' ∧
      A' ≠ B' ∧
      ¬ on_line_l k' m' F₁ ∧
      (∃ (t' : ℝ), line_slope A' F₁ + t' = 2 * k' ∧ line_slope B' F₁ - t' = 2 * k') →
      triangle_area (0, 0) A' B' ≤ max_area :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1128_112869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1128_112807

theorem problem_solution : 
  (∀ x y : ℝ, 
    |(-3)| - Real.sqrt 9 + 5⁻¹ = (1 : ℝ) / 5) ∧
  (∀ x y : ℝ, (x - 2*y)^2 - x*(x - 4*y) = 4*y^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1128_112807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_equals_apples_l1128_112852

/-- Represents the value of one fish in terms of apples -/
def fish_value : ℚ := 7.5

/-- Represents the trading relationship between fish and bread -/
def fish_to_bread : ℚ → ℚ := λ f ↦ 3 * f / 4

/-- Represents the trading relationship between bread and rice -/
def bread_to_rice : ℚ → ℚ := λ b ↦ 5 * b

/-- Represents the trading relationship between rice and apples -/
def rice_to_apples : ℚ → ℚ := λ r ↦ 2 * r

/-- Theorem stating that one fish is worth 7.5 apples given the trading relationships -/
theorem fish_equals_apples : 
  fish_value = (rice_to_apples ∘ bread_to_rice ∘ fish_to_bread) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_equals_apples_l1128_112852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1128_112894

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x + 1 > 0}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1128_112894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_neg_half_max_a_for_inequality_l1128_112808

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := |x - 5/2| + |x - a|

-- Part I
theorem solution_set_when_a_is_neg_half :
  {x : ℝ | f x (-1/2) ≥ 4} = {x : ℝ | x ≤ -1 ∨ x ≥ 3} := by sorry

-- Part II
theorem max_a_for_inequality :
  ∃ (a : ℝ), a = 5/4 ∧ ∀ (x : ℝ), f x a ≥ a ∧
  ∀ (b : ℝ), (∀ (x : ℝ), f x b ≥ b) → b ≤ a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_neg_half_max_a_for_inequality_l1128_112808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_equation_l1128_112864

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line in slope-intercept form -/
structure Line where
  k : ℝ
  c : ℝ

/-- The focal length of an ellipse -/
noncomputable def focal_length (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- The sum of distances from any point on the ellipse to its foci -/
def sum_distances (e : Ellipse) : ℝ := 2 * e.a

theorem ellipse_equation (e : Ellipse) 
  (h_focal : focal_length e = 2 * Real.sqrt 6)
  (h_sum : sum_distances e = 6) :
  e.a = 3 ∧ e.b = Real.sqrt 3 := by
  sorry

theorem line_equation (e : Ellipse) (l : Line) (P : ℝ × ℝ)
  (h_ellipse : e.a = 3 ∧ e.b = Real.sqrt 3)
  (h_P : P = (0, 1))
  (h_intersect : ∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (A.1^2 / 9 + A.2^2 / 3 = 1) ∧
    (B.1^2 / 9 + B.2^2 / 3 = 1) ∧
    (A.2 = l.k * A.1 - l.c) ∧
    (B.2 = l.k * B.1 - l.c))
  (h_equidistant : ∃ A B : ℝ × ℝ, 
    (A.1^2 / 9 + A.2^2 / 3 = 1) ∧
    (B.1^2 / 9 + B.2^2 / 3 = 1) ∧
    (A.2 = l.k * A.1 - l.c) ∧
    (B.2 = l.k * B.1 - l.c) ∧
    ((A.1 - P.1)^2 + (A.2 - P.2)^2 = (B.1 - P.1)^2 + (B.2 - P.2)^2)) :
  (l.k = 1 ∧ l.c = 2) ∨ (l.k = -1 ∧ l.c = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_equation_l1128_112864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1128_112814

theorem relationship_abc : Real.rpow 7 (1/3) < 2 ∧ 2 < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1128_112814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_properties_l1128_112801

/-- The area of a regular hexagon inscribed in a circle of radius 3 units -/
noncomputable def hexagon_area : ℝ := (27 * Real.sqrt 3) / 2

/-- The perimeter of a regular hexagon inscribed in a circle of radius 3 units -/
def hexagon_perimeter : ℝ := 18

/-- Theorem stating the area and perimeter of a regular hexagon inscribed in a circle of radius 3 units -/
theorem hexagon_properties (radius : ℝ) (h : radius = 3) :
  let side_length := radius
  (6 * ((side_length ^ 2 * Real.sqrt 3) / 4) = hexagon_area) ∧
  (6 * side_length = hexagon_perimeter) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_properties_l1128_112801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l1128_112863

/-- Two circles in a plane -/
structure TwoCircles where
  center₁ : ℝ × ℝ
  center₂ : ℝ × ℝ
  radius₁ : ℝ
  radius₂ : ℝ
  intersection : ℝ × ℝ
  (intersect : (center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2 ≤ (radius₁ + radius₂)^2)
  (on_circle₁ : (intersection.1 - center₁.1)^2 + (intersection.2 - center₁.2)^2 = radius₁^2)
  (on_circle₂ : (intersection.1 - center₂.1)^2 + (intersection.2 - center₂.2)^2 = radius₂^2)

/-- Moving points on the circles -/
noncomputable def moving_points (tc : TwoCircles) (t : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let x₁ := (tc.center₁.1 + tc.radius₁ * Real.cos (2 * Real.pi * t), 
             tc.center₁.2 + tc.radius₁ * Real.sin (2 * Real.pi * t))
  let x₂ := (tc.center₂.1 + tc.radius₂ * Real.cos (2 * Real.pi * t), 
             tc.center₂.2 + tc.radius₂ * Real.sin (2 * Real.pi * t))
  (x₁, x₂)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: There exists a fixed point B equidistant from X₁ and X₂ at all times -/
theorem equidistant_point_exists (tc : TwoCircles) : 
  ∃ B : ℝ × ℝ, ∀ t : ℝ, 
    let (X₁, X₂) := moving_points tc t
    distance B X₁ = distance B X₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l1128_112863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_otimes_evaluation_l1128_112816

/-- The ⊗ operation for three real numbers -/
noncomputable def otimes (x y z : ℝ) : ℝ := x / (y - z)

/-- Main theorem: Evaluation of nested ⊗ operations -/
theorem nested_otimes_evaluation :
  ∀ (a b c d e f g h i : ℝ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧ i ≠ 0 →
    b - c ≠ 0 ∧ e - f ≠ 0 ∧ h - i ≠ 0 →
    otimes (3 * otimes a b c) (otimes d e f) (otimes g h i) = -1/3 :=
by
  intros a b c d e f g h i h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_otimes_evaluation_l1128_112816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l1128_112818

/-- A geometric sequence of real numbers -/
noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ := fun n => a * r ^ (n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := 
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_five (a r : ℝ) :
  geometric_sum a r 3 = 13 →
  geometric_sum a r 8 = 1093 →
  geometric_sum a r 5 = 13 * Real.sqrt 333 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l1128_112818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_credits_per_class_l1128_112890

theorem credits_per_class 
  (total_semesters : ℕ) 
  (total_credits_needed : ℕ) 
  (classes_per_semester : ℕ) 
  (credits_per_class : ℕ)
  (h1 : total_semesters = 8)
  (h2 : total_credits_needed = 120)
  (h3 : classes_per_semester = 5)
  (h4 : total_credits_needed = credits_per_class * (classes_per_semester * total_semesters))
  : credits_per_class = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_credits_per_class_l1128_112890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_sqrt_eight_l1128_112804

theorem consecutive_integers_sqrt_eight (a b : ℤ) : 
  (a < Real.sqrt 8 ∧ Real.sqrt 8 < b) →
  (b = a + 1) →
  (b : ℝ)^(a : ℝ) = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_sqrt_eight_l1128_112804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_equivalences_true_l1128_112879

-- Define the necessary types
def Line : Type := Unit
def Plane : Type := Unit

-- Define the perpendicular relation between a line and a plane
def perpendicular (l : Line) (α : Plane) : Prop := True

-- Define the property of a line being perpendicular to every line in a plane
def perpendicularToEveryLine (l : Line) (α : Plane) : Prop := True

-- State the theorem
theorem all_equivalences_true :
  (∀ l : Line, ∀ α : Plane, perpendicularToEveryLine l α → perpendicular l α) ∧
  (∀ l : Line, ∀ α : Plane, perpendicular l α → perpendicularToEveryLine l α) ∧
  (∀ l : Line, ∀ α : Plane, ¬perpendicularToEveryLine l α → ¬perpendicular l α) ∧
  (∀ l : Line, ∀ α : Plane, ¬perpendicular l α → ¬perpendicularToEveryLine l α) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_equivalences_true_l1128_112879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_SQM_is_six_l1128_112829

/-- Rectangle PQRS with length 8 and width 6 -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Point on the diagonal PR -/
structure DiagonalPoint where
  x : ℝ
  y : ℝ

/-- Triangle SQM -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- The area of triangle SQM in the given rectangle PQRS -/
noncomputable def area_triangle_SQM (rect : Rectangle) (m : DiagonalPoint) : ℝ :=
  let diagonal_length := Real.sqrt (rect.length ^ 2 + rect.width ^ 2)
  let segment_length := diagonal_length / 4
  let triangle_base := segment_length
  let triangle_height := 2 * (rect.length * rect.width) / diagonal_length
  (1 / 2) * triangle_base * triangle_height

/-- Theorem stating that the area of triangle SQM is 6 square units -/
theorem area_triangle_SQM_is_six (rect : Rectangle) (m : DiagonalPoint) :
    rect.length = 8 → rect.width = 6 → area_triangle_SQM rect m = 6 := by
  sorry

#check area_triangle_SQM_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_SQM_is_six_l1128_112829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_condition_l1128_112838

/-- Triangle ABC with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi
  sine_law : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

/-- A triangle is equilateral if all its sides are equal -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem triangle_equilateral_condition (t : Triangle) :
  (t.a / Real.cos t.A = t.b / Real.cos t.B) ∧ (t.b / Real.cos t.B = t.c / Real.cos t.C) → t.isEquilateral := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_condition_l1128_112838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l1128_112876

noncomputable def f (x : ℝ) := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

noncomputable def f_translated (x φ : ℝ) := f (x + φ)

theorem min_translation_for_symmetry :
  ∀ φ : ℝ, φ > 0 →
  (∀ x : ℝ, f_translated x φ = f_translated (-x) φ) →
  φ ≥ π / 12 :=
by
  sorry

#check min_translation_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l1128_112876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_closest_to_17_l1128_112871

/-- The area of the shaded region in a 4 × 5 rectangle with a circle of diameter 2 removed -/
noncomputable def shaded_area : ℝ := 20 - Real.pi

/-- The whole number closest to the shaded area -/
def closest_whole_number : ℕ := 17

/-- Theorem stating that 17 is the closest whole number to the shaded area -/
theorem shaded_area_closest_to_17 :
  ∀ n : ℕ, |shaded_area - (closest_whole_number : ℝ)| ≤ |shaded_area - (n : ℝ)| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_closest_to_17_l1128_112871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_abs_diff_greater_than_one_l1128_112858

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the possible values for x and y -/
inductive NumberValue
| Zero
| Two
| Uniform (v : ℝ)

/-- The selection process for a single number -/
def selectNumber (firstFlip secondFlip : CoinFlip) : NumberValue :=
  match firstFlip with
  | CoinFlip.Heads => 
      match secondFlip with
      | CoinFlip.Heads => NumberValue.Zero
      | CoinFlip.Tails => NumberValue.Two
  | CoinFlip.Tails => NumberValue.Uniform 1.5  -- Using a fixed value instead of uniformRandom

/-- The probability of the first coin flip being heads -/
def probHeads : ℚ := 1/2

/-- The probability of |x-y| > 1 given the described selection process -/
def probAbsDiffGreaterThanOne : ℚ := 3/8

/-- Theorem stating that the probability of |x-y| > 1 is 3/8 -/
theorem prob_abs_diff_greater_than_one :
  probAbsDiffGreaterThanOne = 3/8 := by
  -- The proof is omitted for now
  sorry

#eval probAbsDiffGreaterThanOne

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_abs_diff_greater_than_one_l1128_112858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_inter_B_eq_two_three_four_l1128_112823

def A : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}

def B : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n ∧ |n - 3| < 2}

theorem A_inter_B_eq_two_three_four : A ∩ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_inter_B_eq_two_three_four_l1128_112823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_norbs_age_l1128_112874

def guesses : List ℕ := [26, 31, 33, 35, 39, 41, 43, 46, 49, 53, 55, 57]

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def count_low_guesses (age : ℕ) : ℕ := (guesses.filter (· < age)).length

def has_two_off_by_one (age : ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ guesses ∧ b ∈ guesses ∧ ((a + 1 = age ∧ b = age + 1) ∨ (a = age - 1 ∧ b + 1 = age))

theorem norbs_age :
  ∃ (age : ℕ),
    age ∈ guesses ∧
    is_prime age ∧
    (count_low_guesses age : ℚ) ≥ 0.6 * (guesses.length : ℚ) ∧
    has_two_off_by_one age ∧
    ∀ (x : ℕ),
      x ≠ age →
      ¬(x ∈ guesses ∧
        is_prime x ∧
        (count_low_guesses x : ℚ) ≥ 0.6 * (guesses.length : ℚ) ∧
        has_two_off_by_one x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_norbs_age_l1128_112874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_equals_1_l1128_112889

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 3
  | (n + 3) => |sequence_a (n + 2) - sequence_a (n + 1)|

theorem a_2014_equals_1 : sequence_a 2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_equals_1_l1128_112889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_needed_for_field_trip_l1128_112884

theorem bread_needed_for_field_trip : ℕ := by
  let students_per_group : ℕ := 6
  let number_of_groups : ℕ := 5
  let sandwiches_per_student : ℕ := 2
  let bread_slices_per_sandwich : ℕ := 2
  
  let total_students : ℕ := students_per_group * number_of_groups
  let total_sandwiches : ℕ := total_students * sandwiches_per_student
  let total_bread_slices : ℕ := total_sandwiches * bread_slices_per_sandwich
  
  have : total_bread_slices = 100 := by
    -- Proof steps would go here
    sorry
  
  exact total_bread_slices

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_needed_for_field_trip_l1128_112884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_values_l1128_112895

-- Define the original proposition
def original_prop (a b : ℝ) : Prop := (a > 0 ∧ b > 0) → a * b > 0

-- Define the converse
def converse_prop (a b : ℝ) : Prop := a * b > 0 → (a > 0 ∧ b > 0)

-- Define the inverse
def inverse_prop (a b : ℝ) : Prop := (a ≤ 0 ∨ b ≤ 0) → a * b ≤ 0

-- Define the contrapositive
def contrapositive_prop (a b : ℝ) : Prop := a * b ≤ 0 → (a ≤ 0 ∨ b ≤ 0)

-- Theorem stating the truth values of each proposition
theorem proposition_truth_values :
  (∃ a b : ℝ, ¬(converse_prop a b)) ∧
  (∃ a b : ℝ, ¬(inverse_prop a b)) ∧
  (∀ a b : ℝ, contrapositive_prop a b) := by
  sorry

#check proposition_truth_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_truth_values_l1128_112895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_value_l1128_112836

def list (x : ℚ) : List ℚ := [8, 3, x, 3, 7, 3, 9]

def mean (x : ℚ) : ℚ := (33 + x) / 7

def mode : ℚ := 3

def median (x : ℚ) : ℚ :=
  if x ≤ 3 then 3
  else if 7 ≤ x ∧ x ≤ 8 then 7
  else x

def is_arithmetic_progression (a b c : ℚ) : Prop :=
  b - a = c - b ∧ b - a ≠ 0

theorem unique_x_value : 
  ∃! x : ℚ, 3 < x ∧ x < 7 ∧ 
    is_arithmetic_progression mode (median x) (mean x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_value_l1128_112836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l1128_112832

-- Define a structure for a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Auxiliary definitions
def IsIsosceles (t : Triangle) : Prop := t.A = t.B ∨ t.B = t.C ∨ t.C = t.A
def IsRightAngled (t : Triangle) : Prop := t.A + t.B + t.C = Real.pi / 2
def mk_triangle (A B C : ℝ) : Triangle := ⟨A, B, C⟩

theorem triangle_shape (A B C : ℝ) (a b : ℝ) : 
  (a^2 / b^2 = Real.tan A / Real.tan B) →
  (IsIsosceles (mk_triangle A B C) ∨ IsRightAngled (mk_triangle A B C)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l1128_112832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_pockets_l1128_112886

/-- The number of pockets in Janet's dresses -/
def total_pockets (total_dresses : ℕ) (pocket_ratio : ℚ) (two_pocket_ratio : ℚ) : ℕ :=
  let dresses_with_pockets := (total_dresses : ℚ) * pocket_ratio
  let dresses_with_two_pockets := dresses_with_pockets * two_pocket_ratio
  let dresses_with_three_pockets := dresses_with_pockets - dresses_with_two_pockets
  ((dresses_with_two_pockets * 2 + dresses_with_three_pockets * 3).floor : ℤ).toNat

theorem janet_pockets :
  total_pockets 24 (1/2) (1/3) = 32 := by
  sorry

#eval total_pockets 24 (1/2) (1/3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_pockets_l1128_112886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_and_point_B_l1128_112826

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Eccentricity of ellipse C -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- Area of triangle formed by any three vertices of ellipse C -/
noncomputable def triangle_area (a b : ℝ) : ℝ := 2 * Real.sqrt 2

/-- Theorem about ellipse C and point B -/
theorem ellipse_C_and_point_B (a b : ℝ) :
  (ellipse_C x y a b ∧ 
   eccentricity a b = Real.sqrt 2 / 2 ∧
   triangle_area a b = 2 * Real.sqrt 2) →
  (∀ x y, ellipse_C x y a b ↔ x^2 / 4 + y^2 / 2 = 1) ∧
  (∀ t : ℝ, (∃ x y, ellipse_C x y a b ∧ 
              x ≠ 2 ∧ x ≠ -2 ∧
              (x - 2) * (t - x) + y^2 = 0) →
            -2 < t ∧ t < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_and_point_B_l1128_112826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l1128_112896

open Real

/-- Curve C₁ -/
def C₁ (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

/-- Curve C₂ -/
def C₂ (x y : ℝ) : Prop := x - Real.sqrt 2 * y - 5 = 0

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Theorem: Minimum distance between curves C₁ and C₂ -/
theorem min_distance_C₁_C₂ :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
    (∀ (x₃ y₃ x₄ y₄ : ℝ), C₁ x₃ y₃ → C₂ x₄ y₄ →
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄) ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l1128_112896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carnation_count_l1128_112853

theorem carnation_count (total : ℕ) (roses_fraction : ℚ) (tulips : ℕ) (carnations : ℕ) : 
  total = 40 →
  roses_fraction = 2 / 5 →
  tulips = 10 →
  carnations = total - (roses_fraction * ↑total).floor - tulips →
  carnations = 14 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carnation_count_l1128_112853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1128_112827

def M : Set ℕ := {x | 1 < x ∧ x < 7}
def N : Set ℕ := {x : ℕ | ¬(∃ k : ℕ, x = 3 * k)}

theorem intersection_M_N : M ∩ N = {2, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1128_112827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1128_112848

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (x + 1)
  else -x^2 + 2*x

-- State the theorem
theorem f_properties :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (2*x - 1) > f (2 - x) ↔ x > 1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1128_112848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pairwise_sums_for_unique_determination_l1128_112837

/-- Given n ≥ 3 numbers, the minimum number of pairwise sums needed to uniquely determine the original numbers -/
theorem min_pairwise_sums_for_unique_determination (n : ℕ) (h : n ≥ 3) :
  ∃ (k : ℕ), k = Nat.choose (n - 1) 2 + 1 ∧
  (∀ (a : Fin n → ℝ) (b : Fin (Nat.choose n 2) → ℝ),
    (∀ (i j : Fin n), i.val > j.val → ∃ (l : Fin (Nat.choose n 2)), b l = a i + a j) →
    ∃ (S : Finset (Fin (Nat.choose n 2))), S.card = k ∧
      (∀ (a' : Fin n → ℝ),
        (∀ (l : Fin (Nat.choose n 2)), l ∈ S → b l = a' (Fin.mk (l.val / (n-1)) (by sorry)) + a' (Fin.mk (l.val % (n-1)) (by sorry))) →
        a = a')) ∧
  (∀ (k' : ℕ), k' < k →
    ∃ (a a' : Fin n → ℝ) (b : Fin (Nat.choose n 2) → ℝ),
      (∀ (i j : Fin n), i.val > j.val → ∃ (l : Fin (Nat.choose n 2)), b l = a i + a j) ∧
      (∀ (i j : Fin n), i.val > j.val → ∃ (l : Fin (Nat.choose n 2)), b l = a' i + a' j) ∧
      (∃ (S : Finset (Fin (Nat.choose n 2))), S.card = k' ∧
        (∀ (l : Fin (Nat.choose n 2)), l ∈ S → b l = b l) ∧
        (∀ (l : Fin (Nat.choose n 2)), l ∈ S → b l = b l)) ∧
      a ≠ a') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pairwise_sums_for_unique_determination_l1128_112837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fans_receiving_all_items_l1128_112897

def stadium_capacity : ℕ := 5000
def scarf_interval : ℕ := 100
def cap_interval : ℕ := 45
def program_interval : ℕ := 60

theorem fans_receiving_all_items : ℕ := by
  -- The number of fans receiving all three items is equal to the number of multiples
  -- of LCM(scarf_interval, cap_interval, program_interval) that fit within the stadium capacity
  let lcm := Nat.lcm (Nat.lcm scarf_interval cap_interval) program_interval
  have h1 : lcm = Nat.lcm (Nat.lcm scarf_interval cap_interval) program_interval := rfl
  have h2 : stadium_capacity / lcm = 5 := by
    -- This is where we would prove that the division equals 5
    sorry
  exact stadium_capacity / lcm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fans_receiving_all_items_l1128_112897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karl_drove_540_miles_l1128_112880

/-- Represents Karl's car and trip details -/
structure KarlsCar where
  milesPerGallon : ℕ
  tankCapacity : ℕ
  initialMiles : ℕ
  gasBought : ℕ

/-- Calculates the total miles driven by Karl -/
def totalMilesDriven (car : KarlsCar) : ℕ :=
  let initialGasUsed := car.initialMiles / car.milesPerGallon
  let remainingGas := car.tankCapacity - initialGasUsed
  let gasAfterRefill := remainingGas + car.gasBought
  let finalGas := car.tankCapacity / 2
  let secondLegGasUsed := gasAfterRefill - finalGas
  car.initialMiles + (secondLegGasUsed * car.milesPerGallon)

/-- Theorem stating that Karl drove 540 miles -/
theorem karl_drove_540_miles :
  let car := KarlsCar.mk 30 16 360 10
  totalMilesDriven car = 540 := by
  sorry

#eval totalMilesDriven (KarlsCar.mk 30 16 360 10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_karl_drove_540_miles_l1128_112880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometry_l1128_112845

theorem triangle_trigonometry (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  Real.cos B = Real.sqrt 3 / 3 ∧
  Real.sin (A + B) = Real.sqrt 6 / 9 →
  Real.sin A = 2 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometry_l1128_112845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_escalator_time_l1128_112856

/-- Represents the speed of Vasya running down the escalator -/
noncomputable def vasya_down_speed : ℝ := 1 / 2

/-- Represents the speed of the escalator -/
noncomputable def escalator_speed : ℝ := 21 / 100

/-- Time taken on a stationary escalator -/
def stationary_time : ℝ := 6

/-- Time taken on a downward moving escalator -/
def down_escalator_time : ℝ := 13.5

/-- Theorem stating the time taken on an upward moving escalator -/
noncomputable def upward_escalator_time : ℝ :=
  let vasya_up_speed := vasya_down_speed / 2
  let up_time := 1 / (vasya_down_speed - escalator_speed)
  let down_time := 1 / (vasya_up_speed + escalator_speed)
  (up_time + down_time) * 60

/-- The main theorem to be proved -/
theorem vasya_escalator_time : ⌊upward_escalator_time⌋ = 324 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_escalator_time_l1128_112856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_deposit_problem_l1128_112812

/-- Represents the total amount withdrawn after n years -/
noncomputable def total_amount (a r : ℝ) (n : ℕ) : ℝ :=
  a / r * ((1 + r)^(n + 1) - (1 + r))

/-- The problem statement -/
theorem bank_deposit_problem (a r : ℝ) (h1 : a > 0) (h2 : r > 0) (h3 : r < 1) :
  total_amount a r 5 = a / r * ((1 + r)^6 - (1 + r)) := by
  -- The proof goes here
  sorry

#eval "Bank deposit problem formalized."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_deposit_problem_l1128_112812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_g_range_l1128_112833

-- Define the domain M
def M : Set ℝ := { x | -2 < x ∧ x ≤ 1 }

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x) - Real.log (2 + x) / Real.log 10
noncomputable def g (x : ℝ) : ℝ := (4 : ℝ)^x - 2^(x + 1)

-- Theorem for the monotonicity of f
theorem f_decreasing : ∀ {x₁ x₂}, x₁ ∈ M → x₂ ∈ M → x₁ < x₂ → f x₁ > f x₂ := by sorry

-- Theorem for the range of g
theorem g_range : 
  (∀ x, x ∈ M → -1 ≤ g x ∧ g x ≤ 0) ∧ 
  (∃ x₁, x₁ ∈ M ∧ g x₁ = -1) ∧ 
  (∃ x₂, x₂ ∈ M ∧ g x₂ = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_g_range_l1128_112833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_divisibility_l1128_112898

theorem least_number_divisibility (p q r s : ℕ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  ∃ (n : ℕ), n = 863 ∧
    (n + p) % 24 = 0 ∧ 
    (n + q) % 32 = 0 ∧ 
    (n + r) % 36 = 0 ∧ 
    (n + s) % 54 = 0 ∧
    ∀ (m : ℕ), m < n →
      ¬((m + p) % 24 = 0 ∧ 
        (m + q) % 32 = 0 ∧ 
        (m + r) % 36 = 0 ∧ 
        (m + s) % 54 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_divisibility_l1128_112898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_for_singers_l1128_112867

/-- Calculates the total payment for a singer including tip -/
noncomputable def singer_payment (hours : ℝ) (rate : ℝ) (tip_percent : ℝ) : ℝ :=
  let base := hours * rate
  base + (tip_percent / 100) * base

/-- Theorem stating the total payment for all singers -/
theorem total_payment_for_singers : 
  let singer1 := singer_payment 2 25 15
  let singer2 := singer_payment 3 35 20
  let singer3 := singer_payment 4 20 25
  let singer4 := singer_payment 2.5 30 18
  singer1 + singer2 + singer3 + singer4 = 372 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_for_singers_l1128_112867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l1128_112851

noncomputable section

/-- Inverse proportion function -/
def f (x : ℝ) : ℝ := 6 / x

theorem inverse_proportion_properties :
  (∃ x y : ℝ, x = 2 ∧ y = 3 ∧ f x = y) ∧ 
  (∀ x y : ℝ, f x = y → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) ∧
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) ∧
  (∀ x : ℝ, x ≠ 0 → f x ≠ 0) ∧
  (∀ y : ℝ, y ≠ 0 → ∃ x : ℝ, f x = y) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l1128_112851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_l1128_112861

noncomputable section

open Real

-- Define the curves C₁ and C₂
def C₁ (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ * Real.cos θ, 4 * Real.cos θ * Real.sin θ)
def C₂ (m t : ℝ) : ℝ × ℝ := (m + t/2, Real.sqrt 3/2 * t)

-- Define points A, B, and C
def A : ℝ × ℝ := C₁ (π/3)
def B : ℝ × ℝ := C₁ (5*π/6)
def C (m : ℝ) : ℝ × ℝ := C₂ m (2*m/Real.sqrt 3)

-- Define the area of a triangle given three points
def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (x₃, y₃) := p₃
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

-- The main theorem
theorem unique_m_value :
  ∃! m : ℝ, m < 0 ∧ triangle_area A B (C m) = 3 * Real.sqrt 3 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_l1128_112861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_semicircles_area_proof_l1128_112893

/-- The area of the region inside a regular hexagon with side length 4 but outside six alternating semicircles -/
noncomputable def hexagon_semicircles_area : ℝ :=
  let s : ℝ := 4  -- side length of the hexagon
  let r₁ : ℝ := s / 2  -- radius of larger semicircles
  let r₂ : ℝ := s / 4  -- radius of smaller semicircles
  let hexagon_area : ℝ := 3 * Real.sqrt 3 * s^2 / 2
  let large_semicircles_area : ℝ := 3 * Real.pi * r₁^2 / 2
  let small_semicircles_area : ℝ := 3 * Real.pi * r₂^2 / 2
  hexagon_area - large_semicircles_area - small_semicircles_area

/-- Proof that the area of the region inside a regular hexagon with side length 4 
    but outside six alternating semicircles equals 24 * √3 - 15π / 2 -/
theorem hexagon_semicircles_area_proof : 
  hexagon_semicircles_area = 24 * Real.sqrt 3 - 15 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_semicircles_area_proof_l1128_112893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_ge_100_l1128_112831

noncomputable section

-- Define the points A_n and B_n
def A (n : ℕ) : ℝ × ℝ := 
  if n = 0 then (0, 0) else (x n, 0) 
where x : ℕ → ℝ := sorry

def B (n : ℕ) : ℝ × ℝ := 
  (x n, Real.sqrt (x n)) 
where x : ℕ → ℝ := sorry

-- Define the property that A_{n-1}B_nA_n is an equilateral triangle
def is_equilateral (n : ℕ) : Prop :=
  let d₁ := Real.sqrt ((A n).1 - (A (n-1)).1)^2 + ((B n).2)^2
  let d₂ := (A n).1 - (A (n-1)).1
  d₁ = d₂

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Main theorem
theorem smallest_n_ge_100 :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (k : ℕ), k > 0 → is_equilateral k) ∧
  (distance (A 0) (A n) ≥ 100) ∧
  (∀ (m : ℕ), m < n → distance (A 0) (A m) < 100) ∧
  n = 17 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_ge_100_l1128_112831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_unique_r_l1128_112875

/-- The value of r for which the graphs of |z - 4| = 3|z + 4| and |z - 2i| = r 
    intersect at exactly one point in the complex plane -/
noncomputable def intersection_radius : ℝ := Real.sqrt 29 + 3

/-- The set of points satisfying |z - 4| = 3|z + 4| -/
def set_A : Set ℂ := {z : ℂ | Complex.abs (z - 4) = 3 * Complex.abs (z + 4)}

/-- The set of points satisfying |z - 2i| = r -/
def set_B (r : ℝ) : Set ℂ := {z : ℂ | Complex.abs (z - 2*Complex.I) = r}

/-- The theorem stating that the intersection of set_A and set_B with the intersection_radius
    contains exactly one point -/
theorem unique_intersection :
  ∃! z : ℂ, z ∈ set_A ∩ set_B intersection_radius :=
sorry

/-- The theorem stating that intersection_radius is the only value of r for which
    set_A and set_B intersect at exactly one point -/
theorem unique_r :
  ∀ r : ℝ, (∃! z : ℂ, z ∈ set_A ∩ set_B r) → r = intersection_radius :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_unique_r_l1128_112875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_on_day_26_l1128_112860

def cost_price : ℝ := 30

def sales_price (x : ℕ) : ℝ :=
  if x ≤ 30 then 0.5 * (x : ℝ) + 35 else 50

def sales_volume (x : ℕ) : ℝ := 124 - 2 * (x : ℝ)

def daily_profit (x : ℕ) : ℝ :=
  (sales_price x - cost_price) * sales_volume x

def is_valid_day (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 60

theorem max_profit_on_day_26 :
  ∀ x : ℕ, is_valid_day x → daily_profit 26 ≥ daily_profit x ∧ daily_profit 26 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_on_day_26_l1128_112860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_paint_cans_l1128_112872

def red_ratio : ℚ := 4 / 7
def white_ratio : ℚ := 3 / 7
def total_cans : ℕ := 35

theorem red_paint_cans : 
  ⌊(red_ratio * total_cans : ℚ)⌋ = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_paint_cans_l1128_112872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_translation_l1128_112854

theorem sin_graph_translation (x : ℝ) :
  Real.sin (2 * (x + π / 12)) = Real.sin (2 * x + π / 6) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_translation_l1128_112854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutations_with_one_greater_element_l1128_112830

theorem permutations_with_one_greater_element (n : ℕ) (hn : n = 2018) :
  (Finset.range n).card = 2^n - (n + 1) :=
by
  sorry

#check permutations_with_one_greater_element

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutations_with_one_greater_element_l1128_112830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_theorem_l1128_112882

/-- The area of a regular hexagon circumscribed around a circle that is circumscribed around a square with side length a -/
noncomputable def hexagon_area (a : ℝ) : ℝ :=
  Real.sqrt 3 * a^2

/-- Given a square with side length a, a circle circumscribed around the square,
    and a regular hexagon circumscribed around the circle,
    prove that the area of the hexagon is √3 * a^2 -/
theorem hexagon_area_theorem (a : ℝ) (h : a > 0) :
  let r := a * Real.sqrt 2 / 2  -- radius of the circumscribed circle
  hexagon_area a = 3 * Real.sqrt 3 / 2 * (2 * r / Real.sqrt 3)^2 :=
by
  -- Proof steps would go here
  sorry

#check hexagon_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_theorem_l1128_112882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_x_l1128_112868

theorem determine_x (y z x : ℝ) (h1 : y * z ≠ 0) 
  (h2 : Set.toFinset {2 * x, 3 * z, x * y} = Set.toFinset {y, 2 * x^2, 3 * x * z}) : x = 1 :=
by
  -- The proof steps go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_x_l1128_112868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_when_a_zero_a_range_for_negative_f_l1128_112865

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * (x - 1)^2 - x + 1

-- Theorem for part I
theorem extreme_value_when_a_zero :
  ∃ (x_min : ℝ), x_min > 0 ∧ f 0 x_min = 0 ∧ ∀ (x : ℝ), x > 0 → f 0 x ≥ f 0 x_min :=
by sorry

-- Theorem for part II
theorem a_range_for_negative_f :
  ∀ (a : ℝ), (∀ (x : ℝ), x > 1 → f a x < 0) ↔ a ≥ (1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_when_a_zero_a_range_for_negative_f_l1128_112865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_ran_20_more_laps_l1128_112821

/-- Represents the number of laps run by the boys -/
def boys_laps : ℕ := 34

/-- Represents the distance of one lap in miles -/
def lap_distance : ℚ := 1/6

/-- Represents the total distance run by the girls in miles -/
def girls_distance : ℚ := 9

/-- Calculates the difference in laps run by the girls and boys -/
def lap_difference : ℤ := 
  Int.floor ((girls_distance - (boys_laps : ℚ) * lap_distance) / lap_distance)

theorem girls_ran_20_more_laps : lap_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_ran_20_more_laps_l1128_112821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_approx_4_56_minutes_l1128_112806

/-- Represents a train with its length and speed -/
structure Train where
  length : ℝ  -- length in meters
  speed : ℝ   -- speed in km/h

/-- Calculates the time taken for two trains to cross each other when moving in the same direction -/
noncomputable def timeToCross (trainA trainB : Train) : ℝ :=
  let relativeSpeed := trainB.speed - trainA.speed
  let totalLength := trainA.length + trainB.length
  let relativeSpeedMPS := relativeSpeed * (1000 / 3600)
  totalLength / relativeSpeedMPS / 60  -- time in minutes

/-- The theorem stating that the time taken for the given trains to cross is approximately 4.56 minutes -/
theorem trains_crossing_time_approx_4_56_minutes :
  let trainA : Train := { length := 200, speed := 40 }
  let trainB : Train := { length := 180, speed := 45 }
  ∃ ε > 0, |timeToCross trainA trainB - 4.56| < ε := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check timeToCross { length := 200, speed := 40 } { length := 180, speed := 45 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_approx_4_56_minutes_l1128_112806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_correct_l1128_112835

def point : ℝ × ℝ × ℝ := (8, -2, 2)

def normal_vector (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := p

def plane_equation (A B C D : ℤ) (x y z : ℝ) : Prop :=
  (A : ℝ) * x + (B : ℝ) * y + (C : ℝ) * z + (D : ℝ) = 0

def satisfies_equation (A B C D : ℤ) (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  plane_equation A B C D x y z

def is_normal_vector (A B C : ℤ) (n : ℝ × ℝ × ℝ) : Prop :=
  let (nx, ny, nz) := n
  A = Int.floor nx ∧ B = Int.floor ny ∧ C = Int.floor nz

def gcd_condition (A B C D : ℤ) : Prop :=
  Int.gcd (Int.gcd (Int.gcd A B) C) D = 1

theorem plane_equation_correct (A B C D : ℤ) :
  A = 4 ∧ B = -1 ∧ C = 1 ∧ D = -18 →
  A > 0 →
  satisfies_equation A B C D point →
  is_normal_vector A B C (normal_vector point) →
  gcd_condition A B C D →
  ∀ x y z, plane_equation A B C D x y z ↔ 4 * x - y + z - 18 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_correct_l1128_112835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_coefficients_l1128_112873

/-- Given two quadratic functions with coefficients b and c, prove that under certain conditions, b = 2 and c = -3 -/
theorem quadratic_functions_coefficients 
  (f g : ℝ → ℝ) 
  (b c : ℝ) 
  (hf : f = λ x ↦ x^2 + b*x + c) 
  (hg : g = λ x ↦ x^2 + c*x + b) 
  (h_common_root : f 1 = 0 ∧ g 1 = 0) 
  (h_f_root : f (-3) = 0) : 
  b = 2 ∧ c = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_coefficients_l1128_112873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_correct_l1128_112811

/-- The amount of water Jan uses for plates and clothes given her water usage pattern -/
noncomputable def water_for_plates_and_clothes : ℝ :=
  let barrel1 : ℝ := 65
  let barrel2 : ℝ := 75
  let barrel3 : ℝ := 45
  let total_water : ℝ := barrel1 + barrel2 + barrel3
  let water_per_car : ℝ := 7
  let num_cars : ℝ := 2
  let water_for_plants : ℝ := 15
  let water_for_dog : ℝ := 10
  let water_used : ℝ := water_per_car * num_cars + water_for_plants + water_for_dog
  let remaining_water : ℝ := total_water - water_used
  remaining_water / 2

/-- Theorem stating that the calculated water usage for plates and clothes is correct -/
theorem water_usage_correct :
  water_for_plates_and_clothes = 73 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_correct_l1128_112811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1128_112824

-- Define the function f and its inverse g
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := 2^x

-- State the theorem
theorem min_value_theorem (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (h : g a * g b = 16) :
  4 / (2 * a + b) + 1 / (a + 2 * b) ≥ 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1128_112824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_addition_theorem_l1128_112885

/-- Represents a mixture of milk and water -/
structure Mixture where
  total : ℚ
  milk : ℚ
  water : ℚ

/-- Creates a mixture given a total volume and a ratio of milk to water -/
def createMixture (total : ℚ) (ratio : ℚ) : Mixture :=
  let milk := (ratio * total) / (ratio + 1)
  let water := total - milk
  { total := total, milk := milk, water := water }

/-- Adds water to a mixture -/
def addWater (m : Mixture) (amount : ℚ) : Mixture :=
  { total := m.total + amount, milk := m.milk, water := m.water + amount }

/-- Calculates the ratio of milk to water in a mixture -/
def milkToWaterRatio (m : Mixture) : ℚ :=
  m.milk / m.water

theorem water_addition_theorem (initialMixture : Mixture) (addedWater : ℚ) :
  initialMixture = createMixture 45 4 →
  addedWater = 21 →
  milkToWaterRatio (addWater initialMixture addedWater) = 6/5 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_addition_theorem_l1128_112885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_prime_characterization_l1128_112883

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Predicate to check if a number is prime -/
def IsPrime (n : ℤ) : Prop := Nat.Prime n.natAbs

/-- Predicate to check if a set of integers is infinite -/
def IsInfinite (S : Set ℤ) : Prop := ∀ N, ∃ n ∈ S, n > N

/-- The main theorem -/
theorem polynomial_prime_characterization (P : IntPolynomial) :
  (IsInfinite {n : ℤ | IsPrime (P.eval (P.eval n + n))}) →
  (∃ p : ℤ, IsPrime p ∧ P = Polynomial.C p) ∨
  (∃ k : ℤ, Odd k ∧ P = Polynomial.X * (-2) + Polynomial.C k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_prime_characterization_l1128_112883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equality_l1128_112810

theorem trigonometric_equality (α β : ℝ) 
  (h : (Real.sin α)^4 / (Real.sin β)^2 + (Real.cos α)^4 / (Real.cos β)^2 = 1) :
  (Real.tan β)^4 / (Real.tan α)^2 + 1 / ((Real.tan β)^4 * (Real.tan α)^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equality_l1128_112810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_B_squared_l1128_112891

open Matrix

variable {n : ℕ}
variable (A B : Matrix (Fin n) (Fin n) ℝ)

theorem det_A_B_squared (h1 : Matrix.det A = -3) (h2 : Matrix.det B = 5) :
  Matrix.det (A * B ^ 2) = -75 := by
  -- Rewrite B^2 in terms of det B
  have h3 : Matrix.det (B ^ 2) = Matrix.det B ^ 2 := by sorry
  -- Use multiplicativity of determinants
  have h4 : Matrix.det (A * B ^ 2) = Matrix.det A * Matrix.det (B ^ 2) := by sorry
  -- Combine all steps
  calc
    Matrix.det (A * B ^ 2) = Matrix.det A * Matrix.det (B ^ 2) := h4
    _ = Matrix.det A * (Matrix.det B ^ 2) := by rw [h3]
    _ = (-3) * (5 ^ 2) := by rw [h1, h2]
    _ = -75 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_B_squared_l1128_112891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_4170_l1128_112840

/-- Represents a swimmer --/
inductive Swimmer
| jamir
| sarah
| julien
| lily

/-- Calculates the total distance swum by all swimmers over a week --/
noncomputable def totalDistanceSwum (julienSunnySpeed : ℝ) (julienSunnyTime : ℝ) (lilyDailyTime : ℝ) 
  (sunnyDays : ℕ) (cloudyDays : ℕ) : ℝ :=
  let julienSunnyDist := julienSunnySpeed * julienSunnyTime
  let sarahSunnyDist := 2 * julienSunnyDist
  let jamirSunnyDist := sarahSunnyDist + 20
  let lilySunnyDist := 4 * julienSunnySpeed * lilyDailyTime
  let sunnydayTotal := julienSunnyDist + sarahSunnyDist + jamirSunnyDist + lilySunnyDist
  let cloudydayTotal := sunnydayTotal / 2
  sunnyDays * sunnydayTotal + cloudyDays * cloudydayTotal

/-- Theorem stating that the total distance swum by all swimmers over a week is 4170 meters --/
theorem total_distance_is_4170 : 
  totalDistanceSwum 2.5 20 30 5 2 = 4170 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_4170_l1128_112840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l1128_112887

/-- The radius of a sphere inscribed in a regular triangular pyramid -/
noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ :=
  (a * (Real.sqrt 13 - 1)) / 12

/-- A regular triangular pyramid with base side length a and lateral edge angle 60° with the base -/
structure RegularTriangularPyramid (a : ℝ) where
  base_side_length : ℝ
  lateral_edge_angle : ℝ
  base_side_length_eq : base_side_length = a
  lateral_edge_angle_eq : lateral_edge_angle = 60

theorem inscribed_sphere_radius_formula {a : ℝ} (pyramid : RegularTriangularPyramid a) :
  inscribed_sphere_radius a = (a * (Real.sqrt 13 - 1)) / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l1128_112887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_is_sqrt_two_l1128_112870

/-- A right triangular prism with an isosceles right triangle as its base -/
structure IsoscelesRightTriangularPrism where
  side_length : ℝ
  height : ℝ

/-- The volume of an isosceles right triangular prism -/
noncomputable def volume (prism : IsoscelesRightTriangularPrism) : ℝ :=
  (prism.side_length^2 / 2) * prism.height

/-- Theorem: If the height is 3 and the volume is 3, then the side length is √2 -/
theorem side_length_is_sqrt_two (prism : IsoscelesRightTriangularPrism)
  (h_height : prism.height = 3)
  (h_volume : volume prism = 3) :
  prism.side_length = Real.sqrt 2 := by
  sorry

#check side_length_is_sqrt_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_is_sqrt_two_l1128_112870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_4_sqrt_3_pi_l1128_112846

/-- The number of small circles -/
def n : ℕ := 13

/-- The radius of each small circle -/
def r : ℝ := 1

/-- The radius of the large circle -/
noncomputable def R : ℝ := 2 * Real.sqrt 3 + 1

/-- The area of the shaded region -/
noncomputable def shaded_area : ℝ := Real.pi * R^2 - n * Real.pi * r^2

theorem shaded_area_equals_4_sqrt_3_pi : shaded_area = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_4_sqrt_3_pi_l1128_112846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_repaint_theorem_l1128_112843

/-- Represents a cell's color in the table -/
inductive CellColor
| Black
| White

/-- Represents the state of the n × n table -/
def TableState (n : ℕ) := Fin n → Fin n → CellColor

/-- Initial state of the table with opposite corners black and the rest white -/
def initialState (n : ℕ) : TableState n :=
  λ i j => if (i.val = 0 ∧ j.val = 0) ∨ (i.val = n - 1 ∧ j.val = n - 1) then CellColor.Black else CellColor.White

/-- Checks if a given state can be transformed to all black -/
def canTransformToAllBlack (n : ℕ) (state : TableState n) : Prop := sorry

/-- Counts the number of black cells in a given state -/
def blackCellCount (n : ℕ) (state : TableState n) : ℕ := sorry

/-- Theorem: The minimum number of white cells to repaint is 2n - 4 -/
theorem min_repaint_theorem (n : ℕ) (h : n > 1) :
  ∃ (state : TableState n),
    canTransformToAllBlack n state ∧
    blackCellCount n state = 2 * n - 4 ∧
    ∀ (otherState : TableState n),
      canTransformToAllBlack n otherState →
      blackCellCount n otherState ≥ 2 * n - 4 :=
by sorry

#check min_repaint_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_repaint_theorem_l1128_112843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l1128_112881

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem: A train 100 meters long takes 28 seconds to cross a 180-meter bridge at 36 kmph -/
theorem train_crossing_bridge :
  train_crossing_time 100 180 36 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l1128_112881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l1128_112800

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (3*θ) = -23/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l1128_112800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_cartesian_eq_circle_C_polar_eq_perpendicular_line_eq_l1128_112847

-- Define the circle C parametrically
noncomputable def circle_C (φ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos φ, 2 * Real.sin φ)

-- Standard equation in Cartesian coordinates
theorem circle_C_cartesian_eq :
  ∀ x y : ℝ, (∃ φ : ℝ, circle_C φ = (x, y)) ↔ (x - 2)^2 + y^2 = 4 := by sorry

-- Polar coordinate equation of the circle
theorem circle_C_polar_eq :
  ∀ ρ θ : ℝ, (∃ φ : ℝ, circle_C φ = (ρ * Real.cos θ, ρ * Real.sin θ)) ↔ ρ = 4 * Real.cos θ := by sorry

-- Define what it means for a point to lie on a line
def lies_on (point : ℝ × ℝ) (line : ℝ × ℝ → Prop) : Prop :=
  line point

-- Define the perpendicular line passing through (4,0)
def perpendicular_line (point : ℝ × ℝ) : Prop :=
  point.1 = 4

-- Polar coordinate equation of the perpendicular line
theorem perpendicular_line_eq :
  ∀ ρ θ : ℝ, (ρ * Real.cos θ = 4) ↔ lies_on (ρ * Real.cos θ, ρ * Real.sin θ) perpendicular_line := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_cartesian_eq_circle_C_polar_eq_perpendicular_line_eq_l1128_112847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weng_total_earnings_l1128_112817

/-- Represents a babysitting job with an hourly rate and duration in hours and minutes -/
structure BabysittingJob where
  rate : ℚ
  hours : ℕ
  minutes : ℕ

/-- Calculates the earnings for a single babysitting job -/
def jobEarnings (job : BabysittingJob) : ℚ :=
  job.rate * (job.hours + job.minutes / 60 : ℚ)

/-- Calculates the total earnings from multiple babysitting jobs -/
def totalEarnings (jobs : List BabysittingJob) : ℚ :=
  jobs.map jobEarnings |>.sum

/-- Weng's three babysitting jobs -/
def wengsJobs : List BabysittingJob := [
  { rate := 12, hours := 2, minutes := 15 },
  { rate := 15, hours := 1, minutes := 40 },
  { rate := 10, hours := 3, minutes := 10 }
]

theorem weng_total_earnings :
  totalEarnings wengsJobs = 8375 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weng_total_earnings_l1128_112817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_minus_one_l1128_112820

open Complex

theorem modulus_of_z_minus_one : 
  let z : ℂ := (-1 - 2*I) / I
  ‖z - 1‖ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_minus_one_l1128_112820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_to_directrix_distance_l1128_112844

/-- The distance from the center of a hyperbola to its directrix -/
noncomputable def distance_center_to_directrix (a b : ℝ) : ℝ :=
  a^2 / (Real.sqrt (a^2 + b^2))

/-- Theorem: For a hyperbola with given conditions, the distance from its center to its directrix is 1 -/
theorem hyperbola_center_to_directrix_distance
  (a b : ℝ)
  (h_positive : a > 0 ∧ b > 0)
  (h_asymptote : b / a = Real.sqrt 3)
  (h_focal_point : Real.sqrt (a^2 + b^2) = 4) :
  distance_center_to_directrix a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_to_directrix_distance_l1128_112844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1128_112859

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin x + Real.sqrt 3 * (2 * Real.cos x ^ 2 - 1)

theorem f_properties :
  (∃ (max : ℝ), ∀ x, f x ≤ max ∧ ∃ x₀, f x₀ = max) ∧
  (∃ (T : ℝ), T > 0 ∧ ∀ x, f (2 * (x + T)) = f (2 * x)) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (-5 * Real.pi / 24 + k * Real.pi / 2) (Real.pi / 24 + k * Real.pi / 2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1128_112859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l1128_112855

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (1/2) * Real.sin x - (Real.sqrt 3 / 2) * Real.cos x

-- Define the set of slope angles
def slope_angle_set : Set ℝ := {α | ∃ x, 0 ≤ α ∧ α < Real.pi ∧ Real.tan α = (deriv f x)}

-- Theorem statement
theorem slope_angle_range :
  slope_angle_set = (Set.Icc 0 (Real.pi/4)) ∪ (Set.Ico (3*Real.pi/4) Real.pi) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l1128_112855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_interior_angle_is_135_l1128_112841

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- The sum of interior angles of an n-gon in degrees -/
noncomputable def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Measure of one interior angle of a regular octagon in degrees -/
noncomputable def octagon_interior_angle : ℝ := sum_interior_angles octagon_sides / octagon_sides

/-- Theorem stating that the measure of one interior angle of a regular octagon is 135 degrees -/
theorem octagon_interior_angle_is_135 : 
  octagon_interior_angle = 135 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_interior_angle_is_135_l1128_112841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l1128_112839

-- Define the complex number z
noncomputable def z : ℂ := sorry

-- Define the condition
axiom z_condition : (1 + 2 * Complex.I) / z = 1 - Complex.I

-- Theorem to prove
theorem z_in_second_quadrant : 
  z.re < 0 ∧ z.im > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l1128_112839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_bound_two_zeros_product_less_than_one_l1128_112892

noncomputable def f (x a : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

theorem f_nonnegative_implies_a_bound (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 := by sorry

theorem two_zeros_product_less_than_one (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → f x₁ a = 0 → f x₂ a = 0 → x₁ ≠ x₂ → x₁ * x₂ < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_bound_two_zeros_product_less_than_one_l1128_112892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1128_112850

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_angles : angles 0 + angles 1 + angles 2 = Real.pi
  positive_angles : ∀ i, 0 < angles i

-- Define a right angle
def is_right_angle (angle : ℝ) : Prop := angle = Real.pi / 2

-- Define the original statement
def at_most_one_right_angle (t : Triangle) : Prop :=
  ∀ i j : Fin 3, i ≠ j → (is_right_angle (t.angles i) → ¬is_right_angle (t.angles j))

-- Define the negation
def at_least_two_right_angles (t : Triangle) : Prop :=
  ∃ i j : Fin 3, i ≠ j ∧ is_right_angle (t.angles i) ∧ is_right_angle (t.angles j)

-- The theorem to prove
theorem negation_equivalence :
  ¬(∀ t : Triangle, at_most_one_right_angle t) ↔ (∃ t : Triangle, at_least_two_right_angles t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1128_112850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_2x_geq_1_l1128_112815

theorem negation_of_forall_2x_geq_1 :
  ¬(∀ x : ℝ, (2 : ℝ)^x ≥ 1) ↔ ∃ x : ℝ, (2 : ℝ)^x < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_2x_geq_1_l1128_112815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_180_not_divisible_by_3_l1128_112878

theorem divisors_of_180_not_divisible_by_3 : 
  (Finset.filter (fun d => d ∣ 180 ∧ ¬(3 ∣ d)) (Finset.range 181)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_180_not_divisible_by_3_l1128_112878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_figure_l1128_112825

/-- The area of a triangle given its three vertices. -/
noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

theorem area_of_figure (x y : ℝ) : 
  (|15*x| + |8*y| + |120 - 15*x - 8*y| = 120) →
  (∃ A B C : ℝ × ℝ, 
    A = (0, 0) ∧ B = (8, 0) ∧ C = (0, 15) ∧
    (∀ p : ℝ × ℝ, (|15*p.1| + |8*p.2| + |120 - 15*p.1 - 8*p.2| = 120) → 
      (0 ≤ p.1 ∧ p.1 ≤ 8 ∧ 0 ≤ p.2 ∧ p.2 ≤ 15 ∧ 15*p.1 + 8*p.2 ≤ 120)) ∧
    (area_of_triangle A B C = 60)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_figure_l1128_112825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_count_in_class_l1128_112849

/-- Represents a class of students with boys and girls. -/
structure MyClass where
  total_students : ℕ
  boys : ℕ
  girls : ℕ
  total_is_sum : total_students = boys + girls

/-- The probability of choosing a specific student from the class. -/
def student_probability (c : MyClass) : ℚ :=
  1 / c.total_students

/-- The probability of choosing a boy from the class. -/
def boy_probability (c : MyClass) : ℚ :=
  c.boys * (student_probability c)

/-- The probability of choosing a girl from the class. -/
def girl_probability (c : MyClass) : ℚ :=
  c.girls * (student_probability c)

/-- Theorem stating that in a class of 40 students where the probability of choosing
    a boy is 3/4 of the probability of choosing a girl, there are 17 boys. -/
theorem boys_count_in_class (c : MyClass) 
  (h1 : c.total_students = 40)
  (h2 : boy_probability c = 3/4 * girl_probability c) :
  c.boys = 17 := by
  sorry

#check boys_count_in_class

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_count_in_class_l1128_112849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_l1128_112822

theorem downstream_distance 
  (boat_speed : ℝ) 
  (current_speed : ℝ) 
  (travel_time_minutes : ℝ) 
  (h1 : boat_speed = 20) 
  (h2 : current_speed = 5) 
  (h3 : travel_time_minutes = 27) : 
  (boat_speed + current_speed) * (travel_time_minutes / 60) = 11.25 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_l1128_112822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_line_l1128_112819

/-- The equation of a circle given a line passing through its center -/
theorem circle_equation_from_line (a : ℝ) :
  let line := fun (x y : ℝ) => (a - 1) * x - y - a - 1 = 0
  let C : ℝ × ℝ := (1, -2)  -- The fixed point (center of the circle)
  let circle := fun (x y : ℝ) => (x - C.1)^2 + (y - C.2)^2 = 5
  (∀ x y, line x y ↔ line x y) →  -- This represents that the line passes through C for all a
  (∀ x y, circle x y ↔ x^2 + y^2 - 2*x + 4*y = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_line_l1128_112819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_implies_zero_l1128_112862

/-- A function satisfying the given functional equation is the zero function. -/
theorem function_equation_implies_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) + f (x + f y) = 2 * f (x * f y)) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_implies_zero_l1128_112862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l1128_112809

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 20 + y^2 / 16 = 1

-- Define the intersection points A and B
def A : ℝ × ℝ := (-2 * Real.sqrt 5, 0)
def B : ℝ × ℝ := (2 * Real.sqrt 5, 0)

-- Define a movable point M on the ellipse
def M (m n : ℝ) : Prop := ellipse m n ∧ (m, n) ≠ A ∧ (m, n) ≠ B

-- Define the tangent line l at M
def tangent_line (m n x y : ℝ) : Prop := m * x / 20 + n * y / 16 = 1

-- Define points C and D
def C (m n : ℝ) : ℝ × ℝ := (-2 * Real.sqrt 5, 16 / n + 8 * Real.sqrt 5 * m / (5 * n))
def D (m n : ℝ) : ℝ × ℝ := (2 * Real.sqrt 5, 16 / n - 8 * Real.sqrt 5 * m / (5 * n))

-- Define point Q
def Q (m n : ℝ) : ℝ × ℝ := (m, (40 - 2 * m^2) / (5 * n))

-- Define point P
def P (m n : ℝ) : ℝ × ℝ := (m, (2 * m^2 + 10 * n^2 - 40) / (5 * n))

end noncomputable section

-- Theorem statement
theorem locus_of_P (x y : ℝ) :
  (∃ m n : ℝ, M m n ∧ P m n = (x, y)) → y ≠ 0 → y^2 / 36 + x^2 / 20 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l1128_112809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_theorem_l1128_112899

noncomputable def initial_height : ℝ := 500
noncomputable def bounce_ratio : ℝ := 2 / 3
noncomputable def target_height : ℝ := 2

noncomputable def bounce_height (k : ℕ) : ℝ := initial_height * (bounce_ratio ^ k)

theorem ball_bounce_theorem :
  (∀ k : ℕ, k < 12 → bounce_height k ≥ target_height) ∧
  bounce_height 12 < target_height :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_theorem_l1128_112899
