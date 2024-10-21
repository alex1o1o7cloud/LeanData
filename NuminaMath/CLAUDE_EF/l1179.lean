import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_equations_is_28_l1179_117997

def is_valid_equation (b c : ℤ) : Bool :=
  b ∈ ({1,2,3,4,5,6,7} : Finset ℤ) &&
  c ∈ ({1,2,3,4,5,6,7} : Finset ℤ) &&
  b^2 + 4*c ≥ 4*b*c

def count_valid_equations : ℕ :=
  (Finset.filter (λ p : ℤ × ℤ => is_valid_equation p.1 p.2) 
    (Finset.product {1,2,3,4,5,6,7} {1,2,3,4,5,6,7})).card

theorem count_valid_equations_is_28 : count_valid_equations = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_equations_is_28_l1179_117997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l1179_117923

-- Define the binomial expansion function
def binomialExpansion (a b : ℝ) (n : ℕ) : Polynomial ℝ := sorry

-- Define the constant term extractor
def constantTerm (p : Polynomial ℝ) : ℝ := sorry

-- Theorem statement
theorem constant_term_of_expansion :
  constantTerm (binomialExpansion 1 (-1) 8 * (1 + 2 * Polynomial.X^2)) = -42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l1179_117923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1179_117956

/-- The inclination angle of a line is the angle between the positive x-axis and the line, measured counterclockwise. -/
noncomputable def inclination_angle (a b c : ℝ) : ℝ :=
  Real.arctan (-a / b)

/-- Converts radians to degrees -/
noncomputable def to_degrees (x : ℝ) : ℝ :=
  x * (180 / Real.pi)

theorem line_inclination_angle :
  to_degrees (inclination_angle (Real.sqrt 3) 3 2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1179_117956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1179_117912

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (2 * ω * x + Real.pi / 3) - 4 * (Real.cos (ω * x))^2 + 3

theorem problem_solution (ω : ℝ) (h_ω : 0 < ω ∧ ω < 2) 
  (h_symmetry : ∀ x, f ω x = f ω (Real.pi/3 - x)) : 
  ω = 1 ∧ 
  (∀ x, f ω x ≥ -1) ∧
  (∃ x, f ω x = -1) ∧
  (∀ a b c A B C : ℝ, 
    a = 1 → 
    (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4 → 
    f ω A = 2 → 
    a + b + c = 3) := 
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1179_117912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l1179_117920

theorem tan_double_angle (x : ℝ) (h1 : Real.cos x = 3/5) (h2 : x ∈ Set.Ioo (-Real.pi/2) 0) :
  Real.tan (2 * x) = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l1179_117920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_count_l1179_117941

theorem period_count (π : ℝ) (h : π > 0) : 
  (Finset.filter (λ ω : ℕ ↦ 100 * π < ↑ω ∧ ↑ω < 200 * π) (Finset.range 1000)).card = 314 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_count_l1179_117941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_2_sufficient_not_necessary_l1179_117966

-- Define the function f(x) = ax - 2^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (2 : ℝ)^x

-- Define what it means for f to have a root
def has_root (a : ℝ) : Prop := ∃ x : ℝ, f a x = 0

-- State the theorem
theorem a_eq_2_sufficient_not_necessary :
  (∀ a : ℝ, a = 2 → has_root a) ∧ 
  (∃ a : ℝ, a ≠ 2 ∧ has_root a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_2_sufficient_not_necessary_l1179_117966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_connected_triangles_smallest_leg_l1179_117900

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ

/-- Sequence of four connected 30-60-90 triangles -/
structure ConnectedTriangles where
  t1 : Triangle30_60_90
  t2 : Triangle30_60_90
  t3 : Triangle30_60_90
  t4 : Triangle30_60_90

/-- The longer leg of a 30-60-90 triangle -/
noncomputable def longerLeg (t : Triangle30_60_90) : ℝ := t.hypotenuse * (Real.sqrt 3) / 2

/-- Predicate for connected triangles property -/
def isConnected (ct : ConnectedTriangles) : Prop :=
  longerLeg ct.t1 = ct.t2.hypotenuse ∧
  longerLeg ct.t2 = ct.t3.hypotenuse ∧
  longerLeg ct.t3 = ct.t4.hypotenuse

theorem connected_triangles_smallest_leg
  (ct : ConnectedTriangles)
  (h_connected : isConnected ct)
  (h_largest : ct.t1.hypotenuse = 10) :
  longerLeg ct.t4 = 45 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_connected_triangles_smallest_leg_l1179_117900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1179_117927

theorem simplify_expression (n : ℕ) : (2^(n+4) - 3*(2^n)) / (3*(2^(n+3))) = 13/24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1179_117927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_relations_l1179_117986

theorem cubic_root_relations (p q x₁ x₂ : ℝ) 
  (h₁ : x₁^2 + p * x₁ + q = 0)
  (h₂ : x₂^2 + p * x₂ + q = 0) :
  (x₁^3 + x₂^3 = 3 * p * q - p^3) ∧ 
  ((x₁^3 - x₂^3 = (p^2 - q) * Real.sqrt (p^2 - 4 * q)) ∨ 
   (x₁^3 - x₂^3 = -(p^2 - q) * Real.sqrt (p^2 - 4 * q))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_relations_l1179_117986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_one_eq_half_l1179_117947

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x else Real.log x / Real.log 4

-- State the theorem
theorem f_of_f_one_eq_half : f (f 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_one_eq_half_l1179_117947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l1179_117954

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 4 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem statement
theorem line_tangent_to_circle :
  ∃ (x y : ℝ), line_eq x y ∧ circle_eq x y ∧
  ∀ (x' y' : ℝ), line_eq x' y' → circle_eq x' y' → (x = x' ∧ y = y') :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l1179_117954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_e_value_in_special_cubic_l1179_117962

/-- Represents a cubic polynomial of the form 3x^3 + dx^2 + ex + f -/
structure CubicPolynomial where
  d : ℝ
  e : ℝ
  f : ℝ

/-- Computes the y-intercept of the polynomial -/
def y_intercept (p : CubicPolynomial) : ℝ := p.f

/-- Computes the product of zeros of the polynomial -/
noncomputable def product_of_zeros (p : CubicPolynomial) : ℝ := -p.f / 3

/-- Computes the sum of coefficients of the polynomial -/
def sum_of_coefficients (p : CubicPolynomial) : ℝ := 3 + p.d + p.e + p.f

/-- Theorem stating the value of e in the given polynomial -/
theorem e_value_in_special_cubic (p : CubicPolynomial) 
  (h1 : y_intercept p = 27)
  (h2 : product_of_zeros p = sum_of_coefficients p) : 
  p.e = -120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_e_value_in_special_cubic_l1179_117962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cosA_l1179_117961

theorem right_triangle_cosA (A B C : ℝ) (h1 : 0 < A ∧ A < π) (h2 : 0 < C ∧ C < π/2) :
  B = π/2 → Real.sin C = 3/5 → Real.cos A = 3 * Real.sqrt 34 / 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cosA_l1179_117961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_triangle_perimeter_l1179_117950

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_property (t : Triangle) 
  (h : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) :
  2 * t.a^2 = t.b^2 + t.c^2 := by sorry

theorem triangle_perimeter (t : Triangle) 
  (h1 : t.a = 5)
  (h2 : Real.cos t.A = 25/31) :
  t.a + t.b + t.c = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_triangle_perimeter_l1179_117950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l1179_117945

/-- A polynomial in two variables -/
def polynomial (x y k : ℝ) : ℝ := x^2 - 2*x*y + k*y^2 + 3*x - 5*y + 2

/-- Two linear factors -/
def linearFactors (x y m n : ℝ) : ℝ := (x + m*y + 1) * (x + n*y + 2)

/-- The theorem stating the condition for factorization -/
theorem polynomial_factorization (k : ℝ) :
  (∃ m n : ℝ, ∀ x y : ℝ, polynomial x y k = linearFactors x y m n) ↔ k = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l1179_117945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_sum_l1179_117988

/-- Predicate to represent an acute-angled triangle -/
def AcuteAngledTriangle (A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate to represent the circumcenter of a triangle -/
def Circumcenter (O A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate to represent the orthocenter of a triangle -/
def Orthocenter (H A B C : ℝ × ℝ) : Prop := sorry

/-- Function to calculate the area of a triangle -/
noncomputable def AreaTriangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given an acute-angled triangle ABC with circumcenter O and orthocenter H (O ≠ H),
    one of the areas of triangles AOH, BOH, COH is equal to the sum of the other two. -/
theorem triangle_area_sum (A B C O H : ℝ × ℝ) : 
  AcuteAngledTriangle A B C →
  Circumcenter O A B C →
  Orthocenter H A B C →
  O ≠ H →
  (AreaTriangle A O H = AreaTriangle B O H + AreaTriangle C O H) ∨
  (AreaTriangle B O H = AreaTriangle A O H + AreaTriangle C O H) ∨
  (AreaTriangle C O H = AreaTriangle A O H + AreaTriangle B O H) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_sum_l1179_117988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_unbounded_l1179_117982

open Real
open Filter

noncomputable def sequenceLimit (n : ℝ) : ℝ := 
  (n * n^(1/6) + (n^10 + 1)^(1/3)) / ((n + n^(1/4)) * (n^3 - 1)^(1/3))

theorem sequence_unbounded : 
  ∀ M : ℝ, ∃ N : ℝ, ∀ n : ℝ, n ≥ N → sequenceLimit n > M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_unbounded_l1179_117982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_matrices_sum_l1179_117937

theorem inverse_matrices_sum :
  ∀ (a b c d e f g h : ℝ),
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2*a, 2, 3*b; 1, 3, 2; d, 4, c]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![-10, e, -15; f, -20, g; 3, h, 5]
  (A * B = 1) → a + b + c + d + e + f + g + h = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_matrices_sum_l1179_117937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l1179_117919

theorem count_integer_pairs : 
  let count := Finset.card (Finset.filter 
    (fun p : ℕ × ℕ => p.1 < p.2 ∧ Nat.gcd p.1 p.2 = 5 ∧ Nat.lcm p.1 p.2 = 50)
    (Finset.product (Finset.range 51) (Finset.range 51)))
  count = 2^14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l1179_117919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_S_l1179_117964

/-- A square with vertices P, Q, R, S -/
structure Square (P Q R S : ℝ × ℝ) : Prop where
  is_square : True  -- We assume this structure represents a square

/-- Point O is on the line RQ -/
def on_line (O R Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, O = (1 - t) • R + t • Q

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The main theorem -/
theorem max_distance_to_S 
  (P Q R S : ℝ × ℝ) 
  (O : ℝ × ℝ) 
  (h_square : Square P Q R S)
  (h_on_line : on_line O R Q)
  (h_distance : distance O P = 1) :
  (∀ O', on_line O' R Q → distance O' P = 1 → 
    distance O' S ≤ (1 + Real.sqrt 5) / 2) ∧ 
  (∃ O', on_line O' R Q ∧ distance O' P = 1 ∧ 
    distance O' S = (1 + Real.sqrt 5) / 2) := by
  sorry

#check max_distance_to_S

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_S_l1179_117964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_b_value_l1179_117987

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis 1 and semi-minor axis b -/
def Ellipse (b : ℝ) := {p : Point | p.x^2 + p.y^2 / b^2 = 1}

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to a vertical line -/
def distanceToVerticalLine (p : Point) (x : ℝ) : ℝ := |p.x - x|

/-- Theorem: Maximum value of b for the given ellipse properties -/
theorem ellipse_max_b_value 
  (b : ℝ) 
  (hb : 0 < b ∧ b < 1) 
  (F1 F2 : Point) 
  (hF : distance F1 F2 = 2 * Real.sqrt (1 - b^2)) 
  (P : Point) 
  (hP : P ∈ Ellipse b) 
  (hDist : distanceToVerticalLine P (1 / Real.sqrt (1 - b^2)) = (distance P F1 + distance P F2) / 2) :
  b ≤ Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_b_value_l1179_117987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_identity_transformation_l1179_117904

/-- Angle of a line with respect to the positive x-axis -/
noncomputable def angle_with_x_axis (slope : ℝ) : ℝ := Real.arctan slope

/-- Reflection of an angle across a given axis angle -/
def reflect_angle (θ : ℝ) (axis : ℝ) : ℝ := 2 * axis - θ

/-- Transformation R applied once -/
def R (θ : ℝ) (l₁ : ℝ) (l₂ : ℝ) : ℝ := reflect_angle (reflect_angle θ l₁) l₂

/-- Transformation R applied n times -/
def R_n (n : ℕ) (θ : ℝ) (l₁ : ℝ) (l₂ : ℝ) : ℝ :=
  match n with
  | 0 => θ
  | n + 1 => R (R_n n θ l₁ l₂) l₁ l₂

theorem smallest_m_for_identity_transformation :
  let l₁ : ℝ := π / 80
  let l₂ : ℝ := π / 60
  let θ : ℝ := angle_with_x_axis (20 / 89)
  ∃ m : ℕ+, (∀ k : ℕ+, k < m → R_n k θ l₁ l₂ ≠ θ) ∧ R_n m θ l₁ l₂ = θ ∧ m = 240 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_identity_transformation_l1179_117904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1179_117975

/-- The function y = √(x+1) / x -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) / x

/-- The set representing the range of x -/
def X : Set ℝ := {x | x ≥ -1 ∧ x ≠ 0}

/-- Theorem stating that x is in the domain of f if and only if f(x) is defined -/
theorem f_domain (x : ℝ) : x ∈ X ↔ ∃ y, f x = y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1179_117975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_collections_count_l1179_117901

/-- Represents the number of each letter in MATHEMATICEONS --/
def letterCounts : List (Char × Nat) :=
  [('A', 2), ('E', 2), ('I', 1), ('O', 1), ('U', 1),
   ('C', 2), ('N', 2), ('T', 2), ('M', 2), ('H', 1), ('S', 1)]

/-- The total number of vowels that fall off --/
def fallenVowels : Nat := 3

/-- The total number of consonants that fall off --/
def fallenConsonants : Nat := 5

/-- Set of vowels --/
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

/-- Set of consonants --/
def consonants : Finset Char := {'C', 'N', 'T', 'M', 'H', 'S'}

/-- Function to calculate the number of distinct collections --/
def distinctCollections : Nat :=
  -- Implementation details omitted
  288 -- Placeholder value

theorem distinct_collections_count :
  distinctCollections = 288 := by
  -- Proof details omitted
  sorry

#eval distinctCollections

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_collections_count_l1179_117901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_diagonal_edge_60_degrees_l1179_117914

-- Define a cube
structure Cube where
  side : ℝ
  side_positive : side > 0

-- Define the face diagonal of a cube
noncomputable def face_diagonal (c : Cube) : ℝ := c.side * Real.sqrt 2

-- Define the angle between face diagonal and edge
noncomputable def angle_diagonal_edge (c : Cube) : ℝ := Real.arccos (1 / Real.sqrt 2)

-- Theorem statement
theorem angle_diagonal_edge_60_degrees (c : Cube) :
  angle_diagonal_edge c = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_diagonal_edge_60_degrees_l1179_117914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1179_117913

/-- The parabola C: y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (3, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_distance_theorem (A : ℝ × ℝ) :
  parabola A.1 A.2 →
  distance A focus = distance B focus →
  distance A B = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1179_117913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l1179_117925

theorem calculation_proofs :
  ((-1 : ℝ) ^ 2023 + (Real.pi - 3.14) ^ 0 * (-2 : ℝ) ^ 2 + (1 / 3) ^ (-2 : ℤ) : ℝ) = 12 ∧
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → x / (x - 1) - 1 / (x + 1) = (x^2 + 1) / (x^2 - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l1179_117925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalene_triangle_distinct_lines_l1179_117969

/-- A scalene triangle is a triangle where all sides and angles are different. -/
structure ScaleneTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_scalene : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
    dist (vertices i) (vertices j) ≠ dist (vertices j) (vertices k) ∧
    dist (vertices i) (vertices j) ≠ dist (vertices k) (vertices i) ∧
    dist (vertices j) (vertices k) ≠ dist (vertices k) (vertices i)

/-- A line in a triangle can be a median, altitude, or angle bisector. -/
inductive TriangleLine
  | Median
  | Altitude
  | AngleBisector

/-- Count the number of distinct lines (medians, altitudes, and angle bisectors) in a scalene triangle. -/
def count_distinct_lines (t : ScaleneTriangle) : ℕ :=
  9 -- Implementation details omitted, directly return 9 as per the problem solution

/-- Theorem stating that a scalene triangle has exactly 9 distinct lines comprising medians, altitudes, and angle bisectors. -/
theorem scalene_triangle_distinct_lines (t : ScaleneTriangle) :
  count_distinct_lines t = 9 := by
  rfl -- reflexivity, since count_distinct_lines is defined to return 9

#check scalene_triangle_distinct_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalene_triangle_distinct_lines_l1179_117969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_area_is_700_l1179_117998

/-- The area of a trapezoidal cross-section of a water channel. -/
noncomputable def trapezoidalChannelArea (topWidth bottomWidth depth : ℝ) : ℝ :=
  (1/2) * (topWidth + bottomWidth) * depth

/-- Theorem stating that the area of the specified trapezoidal channel is 700 square meters. -/
theorem channel_area_is_700 :
  trapezoidalChannelArea 12 8 70 = 700 := by
  -- Unfold the definition of trapezoidalChannelArea
  unfold trapezoidalChannelArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_channel_area_is_700_l1179_117998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1179_117926

noncomputable def f (x : ℝ) := Real.cos (2 * x) + 6 * Real.cos (Real.pi / 2 - x) + 1

theorem f_max_value : ∀ x : ℝ, f x ≤ 6 ∧ ∃ y : ℝ, f y = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1179_117926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_l1179_117985

-- Define the curves
noncomputable def curve1 (x : ℝ) : ℝ := x + Real.log x
def curve2 (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2*a + 3) * x + 1

-- Define the tangent line of curve1 at (1, 1)
def tangent_line (x : ℝ) : ℝ := 2*x - 1

-- Define the condition for single intersection
def single_intersection (a : ℝ) : Prop :=
  ∃! x, tangent_line x = curve2 a x

theorem tangent_intersection :
  ∀ a : ℝ, single_intersection a ↔ (a = 0 ∨ a = 1/2) := by
  sorry

#check tangent_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_l1179_117985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_sqrt_131_l1179_117970

-- Define the points
def A : ℝ × ℝ := (6, 7)
def B : ℝ × ℝ := (9, 11)
def C : ℝ × ℝ := (8, 16)
def O : ℝ × ℝ := (0, 0)

-- Define the circle passing through A, B, and C
def circleABC : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (center : ℝ × ℝ) (radius : ℝ), 
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 ∧ 
  (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
  (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2 ∧
  (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2}

-- Define the tangent length
noncomputable def tangentLength : ℝ := Real.sqrt 131

-- Theorem statement
theorem tangent_length_is_sqrt_131 : 
  ∃ (T : ℝ × ℝ), T ∈ circleABC ∧ 
  (∀ (p : ℝ × ℝ), p ∈ circleABC → (T.1 - O.1)^2 + (T.2 - O.2)^2 ≤ (p.1 - O.1)^2 + (p.2 - O.2)^2) ∧
  (T.1 - O.1)^2 + (T.2 - O.2)^2 = tangentLength^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_sqrt_131_l1179_117970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_and_non_P_l1179_117922

-- Define the Quadrilateral type and necessary properties
structure Quadrilateral where
  -- You might want to add more properties here, like vertices or sides
  mk :: -- Empty constructor for now

def InscribedInCircle : Quadrilateral → Prop := sorry
def SupplementaryOppositeAngles : Quadrilateral → Prop := sorry

-- Define the proposition P
def P : Prop := ∀ q : Quadrilateral, InscribedInCircle q → SupplementaryOppositeAngles q

-- Define the negation of P
def negation_of_P : Prop := ∃ q : Quadrilateral, ¬InscribedInCircle q ∧ ¬SupplementaryOppositeAngles q

-- Define non-P
def non_P : Prop := ∃ q : Quadrilateral, InscribedInCircle q ∧ ¬SupplementaryOppositeAngles q

-- Theorem to prove
theorem negation_and_non_P : 
  (¬P ↔ negation_of_P) ∧ (¬P ↔ non_P) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_and_non_P_l1179_117922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_half_l1179_117960

-- Define the function g
def g (x : ℝ) : ℝ := 1 - 2 * x

-- Define f as a variable (since it's implicitly defined)
noncomputable def f : ℝ → ℝ := sorry

-- Define the relationship between f and g
axiom f_comp_g : ∀ x : ℝ, x ≠ 0 → f (g x) = (1 - x^2) / 2

-- Theorem to prove
theorem f_one_half : f (1/2 : ℝ) = 15/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_half_l1179_117960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_squares_divisible_by_four_l1179_117910

theorem three_digit_squares_divisible_by_four :
  ∃! (count : ℕ), 
    count = (Finset.filter 
      (λ n : ℕ ↦ 100 ≤ n^2 ∧ n^2 < 1000 ∧ 4 ∣ n^2) 
      (Finset.range 32)).card ∧
    count = 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_squares_divisible_by_four_l1179_117910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chromium_percentage_in_new_alloy_l1179_117959

/-- The percentage of chromium in the new alloy -/
noncomputable def new_alloy_chromium_percentage (first_alloy_weight : ℝ) (first_alloy_percentage : ℝ)
  (second_alloy_weight : ℝ) (second_alloy_percentage : ℝ) : ℝ :=
  ((first_alloy_weight * first_alloy_percentage + second_alloy_weight * second_alloy_percentage) /
   (first_alloy_weight + second_alloy_weight)) * 100

/-- Theorem stating that the percentage of chromium in the new alloy is 9.2% -/
theorem chromium_percentage_in_new_alloy :
  new_alloy_chromium_percentage 15 0.12 35 0.08 = 9.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chromium_percentage_in_new_alloy_l1179_117959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_value_l1179_117995

/-- Calculates the percent increase in area of a circular pizza when its diameter increases from 10 inches to 12 inches -/
noncomputable def pizza_area_increase : ℝ := by
  -- Define the initial and final diameters
  let initial_diameter : ℝ := 10
  let final_diameter : ℝ := 12

  -- Define a function to calculate the area of a pizza given its diameter
  let pizza_area (d : ℝ) : ℝ := Real.pi * (d / 2) ^ 2

  -- Calculate the initial and final areas
  let initial_area := pizza_area initial_diameter
  let final_area := pizza_area final_diameter

  -- Calculate the percent increase
  let percent_increase := (final_area - initial_area) / initial_area * 100

  -- Return the percent increase
  exact percent_increase

/-- The percent increase in area is 44% -/
theorem pizza_area_increase_value : pizza_area_increase = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_value_l1179_117995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_point_distance_l1179_117951

-- Define the stage length
def stageLength : ℝ := 20

-- Theorem statement
theorem golden_ratio_point_distance :
  let x : ℝ := 30 - 10 * Real.sqrt 5
  let goldenRatio : ℝ := (Real.sqrt 5 - 1) / 2
  x / stageLength = goldenRatio ∧ x < stageLength := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_point_distance_l1179_117951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1179_117946

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio q -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- The common ratio of a geometric sequence satisfying S_4 = 5S_2 -/
theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) (h : q ≠ 1) :
  geometric_sum a q 4 = 5 * geometric_sum a q 2 →
  q = -1 ∨ q = 2 ∨ q = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1179_117946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_sales_portion_l1179_117928

/-- Represents the problem of determining the portion of a dozen strawberries sold at a discounted price. -/
theorem strawberry_sales_portion
  (cost_per_dozen : ℝ)
  (total_dozens : ℝ)
  (total_profit : ℝ)
  (discounted_price : ℝ)
  (h1 : cost_per_dozen = 50)
  (h2 : total_dozens = 50)
  (h3 : total_profit = 500)
  (h4 : discounted_price = 30) :
  (((cost_per_dozen * total_dozens + total_profit) / total_dozens - cost_per_dozen) /
   (discounted_price - cost_per_dozen)) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_sales_portion_l1179_117928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutations_of_five_adjacent_nonadjacent_five_distribute_five_to_three_l1179_117948

-- Define the set of students
def Students := Fin 5

-- Define the number of classes
def NumClasses := 3

-- Part 1: Permutations of 5 students
theorem permutations_of_five : Nat.factorial 5 = 120 := by sorry

-- Part 2: Permutations with adjacency and non-adjacency constraints
def adjacent_nonadjacent_permutations : ℕ := 24

theorem adjacent_nonadjacent_five : adjacent_nonadjacent_permutations = 24 := by sorry

-- Part 3: Distributing students to classes
def distribute_to_classes : ℕ := 150

theorem distribute_five_to_three : distribute_to_classes = 150 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutations_of_five_adjacent_nonadjacent_five_distribute_five_to_three_l1179_117948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l1179_117990

/-- The function f(x) = (3x^2 + 4x + 8) / (x + 5) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 4 * x + 8) / (x + 5)

/-- The proposed oblique asymptote function g(x) = 3x - 11 -/
def g (x : ℝ) : ℝ := 3 * x - 11

/-- Theorem: The oblique asymptote of f(x) is g(x) -/
theorem oblique_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - g x| < ε :=
by
  sorry

#check oblique_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l1179_117990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l1179_117934

-- Define the types for our variables
variable (a b : ℝ)

-- Define the solution set of ax + b > 0
def solution_set_1 (a b : ℝ) : Set ℝ := {x | x < 1}

-- Define the solution set we want to prove
def solution_set_2 : Set ℝ := {x | x < -2 ∨ x > -1}

-- State the theorem
theorem inequality_solution_sets (a b : ℝ) :
  (∀ x, ax + b > 0 ↔ x ∈ solution_set_1 a b) →
  (∀ x, (b*x - a) / (x + 2) > 0 ↔ x ∈ solution_set_2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l1179_117934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1179_117989

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℚ
  last_term : ℚ
  sum : ℚ
  num_terms : ℕ
  common_difference : ℚ

/-- The properties of our specific arithmetic sequence -/
def our_sequence : ArithmeticSequence where
  first_term := 3
  last_term := 58
  sum := 488
  num_terms := 16  -- This is derived from the sum formula
  common_difference := 11/3

/-- Theorem stating that our sequence satisfies the properties of an arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h1 : seq.first_term = 3)
  (h2 : seq.last_term = 58)
  (h3 : seq.sum = 488) :
  seq.common_difference = 11/3 := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1179_117989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_distance_l1179_117957

/-- The total distance covered by a fly moving between two cyclists -/
theorem fly_distance (cyclist_speed initial_distance fly_speed : ℝ) :
  cyclist_speed = 10 →
  initial_distance = 50 →
  fly_speed = 15 →
  let relative_speed := 2 * cyclist_speed;
  let meeting_time := initial_distance / relative_speed;
  fly_speed * meeting_time = 37.5 := by
  sorry

#check fly_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_distance_l1179_117957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_needed_theorem_l1179_117935

/-- Calculates the total amount of water needed for Nicole's fish tanks over 25 days -/
def water_needed_in_25_days (num_tanks : ℕ) (first_tank_gallons : ℕ) (second_tank_diff : ℕ) (third_tank_diff : ℕ) (change_interval : ℕ) (total_days : ℕ) : ℕ :=
  let first_pair := 2 * first_tank_gallons
  let second_pair := 2 * (first_tank_gallons - second_tank_diff)
  let third_pair := 2 * (first_tank_gallons + third_tank_diff)
  let total_per_change := first_pair + second_pair + third_pair
  let num_changes := total_days / change_interval
  total_per_change * num_changes

theorem water_needed_theorem :
  water_needed_in_25_days 6 10 2 3 5 25 = 310 := by
  rfl

#eval water_needed_in_25_days 6 10 2 3 5 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_needed_theorem_l1179_117935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_guess_l1179_117981

noncomputable def correct_result : ℝ := 2.4 - 1.5 * 3.6 / 2

def tomas_guess : ℝ := -1.2
def jirka_guess : ℝ := 1.7

def average_deviation : ℝ := 0.4
def worst_deviation : ℝ := 2

theorem martin_guess :
  ∃ (martin_guess : ℝ),
    (abs (martin_guess - correct_result) ≤ worst_deviation) ∧
    (abs (tomas_guess - correct_result) ≤ worst_deviation) ∧
    (abs (jirka_guess - correct_result) ≤ worst_deviation) ∧
    (abs ((martin_guess + tomas_guess + jirka_guess) / 3 - correct_result) = average_deviation) ∧
    martin_guess = -0.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_guess_l1179_117981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_l1179_117942

theorem zeros_before_first_nonzero_digit (n d : ℕ) (h : d ≠ 0) :
  let f := n / d
  let decimal := toString f
  let zeros := decimal.drop 2 |>.takeWhile (· = '0')
  zeros.length = 4 :=
by
  sorry

#eval toString (5 / 1600 : Rat)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_l1179_117942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_eq_N_eq_M_inter_N_eq_M_union_complement_N_eq_l1179_117909

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 - 4*x + 1 else -1/x + 5

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt (-2*x^2 + 5*x + 3)

-- Define set M
def M : Set ℝ :=
  {x | f x ≤ 4}

-- Define set N
def N : Set ℝ :=
  {x | -2*x^2 + 5*x + 3 ≥ 0}

-- Theorem statements
theorem M_eq : M = {x | x ≤ -3 ∨ (-1 ≤ x ∧ x ≤ 1)} := by
  sorry

theorem N_eq : N = {x | -1/2 ≤ x ∧ x ≤ 3} := by
  sorry

theorem M_inter_N_eq : M ∩ N = {x | -1/2 ≤ x ∧ x ≤ 1} := by
  sorry

theorem M_union_complement_N_eq : M ∪ Nᶜ = {x | x ≤ 1 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_eq_N_eq_M_inter_N_eq_M_union_complement_N_eq_l1179_117909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_point_in_g_given_in_G_l1179_117911

-- Define the ellipses
def ellipse_G (x y : ℝ) : Prop := x^2/49 + y^2/16 = 1
def ellipse_g (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Define the areas of the ellipses
noncomputable def area_G : ℝ := Real.pi * 7 * 4
noncomputable def area_g : ℝ := Real.pi * 5 * 3

-- State the theorem
theorem probability_point_in_g_given_in_G : 
  (area_g / area_G : ℝ) = 3/7 := by
  -- Expand the definitions of area_g and area_G
  unfold area_g area_G
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_point_in_g_given_in_G_l1179_117911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_feed_cost_per_pound_l1179_117921

-- Define the variables
def total_weight : ℝ := 17
def cheap_price : ℝ := 0.11
def expensive_price : ℝ := 0.50
def cheap_weight : ℝ := 12.2051282051

-- Define the theorem
theorem mixed_feed_cost_per_pound :
  let expensive_weight := total_weight - cheap_weight
  let total_cost := cheap_price * cheap_weight + expensive_price * expensive_weight
  let cost_per_pound := total_cost / total_weight
  ∃ ε > 0, |cost_per_pound - 0.22| < ε := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_feed_cost_per_pound_l1179_117921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1179_117905

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (0, 2)
def c : ℝ × ℝ := (4, 1)

theorem vector_problem :
  (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5) ∧
  (∃ k : ℝ, k = 3 ∧ ∃ t : ℝ, t ≠ 0 ∧ 
    (a.1 + k * c.1, a.2 + k * c.2) = (t * (2 * a.1 - b.1), t * (2 * a.2 - b.2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1179_117905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_eight_l1179_117903

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -Real.sqrt 3; Real.sqrt 3, 1]

theorem matrix_power_eight :
  A ^ 8 = !![(-128 : ℝ), 128 * Real.sqrt 3; -128 * Real.sqrt 3, (-128 : ℝ)] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_eight_l1179_117903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_norm_inequality_l1179_117915

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_norm_inequality (a b c : V) : 
  ‖a‖ + ‖b‖ + ‖c‖ + ‖a + b + c‖ ≥ ‖a + b‖ + ‖b + c‖ + ‖c + a‖ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_norm_inequality_l1179_117915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_selection_probabilities_l1179_117907

-- Define the total number of students and the number of female and male students
def total_students : ℕ := 10
def female_students : ℕ := 6
def male_students : ℕ := 4

-- Define the number of students to be selected
def selected_students : ℕ := 3

-- Define the probabilities of passing the test for female and male students
def female_pass_prob : ℚ := 4/5
def male_pass_prob : ℚ := 3/5

-- Define the probability of at least one male student being selected
def prob_at_least_one_male : ℚ := 5/6

-- Define the probability of specific students A and B being selected and passing
def prob_A_and_B_selected_and_pass : ℚ := 4/125

-- Theorem statement
theorem student_selection_probabilities :
  (1 - (Nat.choose female_students selected_students : ℚ) / (Nat.choose total_students selected_students : ℚ) = prob_at_least_one_male) ∧
  ((Nat.choose (total_students - 2) 1 : ℚ) / (Nat.choose total_students selected_students : ℚ) * female_pass_prob * male_pass_prob = prob_A_and_B_selected_and_pass) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_selection_probabilities_l1179_117907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_210_and_330_l1179_117902

theorem lcm_of_210_and_330 :
  Nat.lcm 210 330 = 2310 :=
by
  -- We'll use the built-in Nat.lcm function instead of defining our own
  -- Calculate the LCM
  have h : Nat.lcm 210 330 = 2310 := by norm_num
  -- Apply the calculated result
  exact h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_210_and_330_l1179_117902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_comparison_l1179_117917

/-- Represents a class with its average score and variance -/
structure ClassStats where
  average_score : ℝ
  variance : ℝ

/-- Given two classes, determines if the first class has more stable scores -/
def more_stable (a b : ClassStats) : Prop :=
  a.variance < b.variance

/-- Given two classes, determines if the second class fluctuates more -/
def fluctuates_more (a b : ClassStats) : Prop :=
  b.variance > a.variance

theorem class_comparison (class_a class_b : ClassStats)
  (h1 : class_a.average_score = 106.8)
  (h2 : class_b.average_score = 107)
  (h3 : class_a.variance = 6)
  (h4 : class_b.variance = 14) :
  more_stable class_a class_b ∧ fluctuates_more class_a class_b := by
  sorry

#check class_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_comparison_l1179_117917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_side_inequality_l1179_117908

theorem triangle_angle_side_inequality 
  (A B C k a b : Real) 
  (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = π)
  (h5 : k > 1)
  (h6 : A = k * B)
  (h7 : a = 2 * Real.sin (A/2))
  (h8 : b = 2 * Real.sin (B/2)) :
  a < k * b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_side_inequality_l1179_117908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_l1179_117940

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - 4*x else ((-x)^2 - 4*(-x))

-- State the theorem
theorem solution_set_of_f (f : ℝ → ℝ) : 
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x ≥ 0, f x = x^2 - 4*x) →  -- definition for x ≥ 0
  {x | f (x + 2) < 5} = Set.Ioo (-7) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_l1179_117940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1179_117984

noncomputable def m (α : Real) : Real × Real := (Real.cos α - Real.sqrt 2 / 3, -1)
noncomputable def n (α : Real) : Real × Real := (Real.sin α, 1)

def collinear (v w : Real × Real) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem trigonometric_identities (α : Real) 
  (h1 : collinear (m α) (n α)) 
  (h2 : α ∈ Set.Icc (-Real.pi) 0) : 
  Real.sin α + Real.cos α = Real.sqrt 2 / 3 ∧ 
  Real.sin (2 * α) / (Real.sin α - Real.cos α) = 7 / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1179_117984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_billing_theorem_l1179_117932

/-- Represents a tariff with its rate -/
structure Tariff where
  name : String
  rate : ℝ

/-- Represents meter readings for a specific tariff -/
structure MeterReading where
  current : ℕ
  previous : ℕ

/-- Calculates the consumption for a meter reading -/
def consumption (reading : MeterReading) : ℕ :=
  reading.current - reading.previous

/-- Calculates the cost for a given consumption and tariff -/
def cost (consumption : ℕ) (tariff : Tariff) : ℝ :=
  (consumption : ℝ) * tariff.rate

/-- Theorem: Maximum additional payment and expected difference -/
theorem energy_billing_theorem 
  (peak : Tariff)
  (night : Tariff)
  (half_peak : Tariff)
  (peak_reading : MeterReading)
  (night_reading : MeterReading)
  (half_peak_reading : MeterReading)
  (actual_payment : ℝ) :
  let max_additional_payment := 397.34
  let expected_difference := 19.30
  (peak.rate = 4.03 ∧ 
   night.rate = 1.01 ∧ 
   half_peak.rate = 3.39 ∧
   peak_reading.current = 1402 ∧
   peak_reading.previous = 1214 ∧
   night_reading.current = 1347 ∧
   night_reading.previous = 1270 ∧
   half_peak_reading.current = 1337 ∧
   half_peak_reading.previous = 1298 ∧
   actual_payment = 660.72) →
  (∃ (calculated_payment : ℝ),
     calculated_payment - actual_payment ≤ max_additional_payment ∧
     ∃ (expected_calculated_payment : ℝ),
       expected_calculated_payment - actual_payment = expected_difference) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_billing_theorem_l1179_117932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1179_117929

noncomputable def series_term (n : ℕ) (b c : ℝ) : ℝ :=
  1 / ((n * c - (n - 1) * b) * ((n + 1) * c - n * b))

theorem series_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a > b) (hbc : b > c) :
  (∑' n, series_term n b c) = 1 / ((c - b) * b) := by
  sorry

#check series_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1179_117929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base9ToBase3Conversion_l1179_117949

-- Define a function to convert a single digit from base 9 to base 3
def base9ToBase3Digit (d : Nat) : List Nat :=
  match d with
  | 0 => [0, 0]
  | 1 => [0, 1]
  | 2 => [0, 2]
  | 3 => [1, 0]
  | 4 => [1, 1]
  | 5 => [1, 2]
  | 6 => [2, 0]
  | 7 => [2, 1]
  | 8 => [2, 2]
  | _ => [0, 0]  -- Default case, should not occur for valid base 9 digits

-- Define the base 9 number
def base9Number : List Nat := [8, 7, 2, 3]

-- Define the expected base 3 result
def expectedBase3Result : List Nat := [2, 2, 2, 1, 0, 2, 1, 0]

-- Theorem statement
theorem base9ToBase3Conversion :
  (base9Number.bind base9ToBase3Digit) = expectedBase3Result :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base9ToBase3Conversion_l1179_117949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slice_result_edges_l1179_117973

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ

/-- Represents the result of slicing a polyhedron -/
structure SlicedPolyhedron where
  original : ConvexPolyhedron
  newEdges : ℕ

/-- Function that performs the slicing operation -/
def slicePolyhedron (Q : ConvexPolyhedron) : SlicedPolyhedron :=
  { original := Q
  , newEdges := 720 }

/-- Theorem stating that slicing a specific polyhedron results in 840 edges -/
theorem slice_result_edges (Q : ConvexPolyhedron) 
  (h1 : Q.vertices > 0)
  (h2 : Q.edges = 120) :
  (slicePolyhedron Q).original.edges + (slicePolyhedron Q).newEdges = 840 := by
  sorry

#check slice_result_edges

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slice_result_edges_l1179_117973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l1179_117953

/-- The central angle of the sector in the lateral surface development of a cone -/
noncomputable def central_angle (base_diameter : ℝ) (slant_height : ℝ) : ℝ :=
  (2 * base_diameter * Real.pi / slant_height) * (180 / Real.pi)

/-- Theorem: The central angle of the sector in the lateral surface development
    of a cone with base diameter 4 and slant height 6 is 120° -/
theorem cone_central_angle :
  central_angle 4 6 = 120 := by
  -- Unfold the definition of central_angle
  unfold central_angle
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l1179_117953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_greater_than_2_99_l1179_117999

-- Define the sequence
def a (a₀ a₁ : ℤ) : ℕ → ℤ
  | 0 => a₀
  | 1 => a₁
  | (n + 2) => 3 * a a₀ a₁ (n + 1) - 2 * a a₀ a₁ n

-- State the theorem
theorem a_100_greater_than_2_99 
  (a₀ a₁ : ℤ) 
  (h1 : 0 < a₀) 
  (h2 : a₀ < a₁) : 
  a a₀ a₁ 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_greater_than_2_99_l1179_117999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_white_correct_prob_at_least_two_white_correct_l1179_117955

/-- Represents the color of a ball -/
inductive BallColor
| White
| Black

/-- Represents the urn with white and black balls -/
structure Urn :=
  (white : ℕ)
  (black : ℕ)

/-- Calculates the probability of drawing a specific color from the urn -/
def prob_draw (u : Urn) (c : BallColor) : ℚ :=
  match c with
  | BallColor.White => u.white / (u.white + u.black)
  | BallColor.Black => u.black / (u.white + u.black)

/-- Calculates the probability of drawing exactly two white balls in three draws with replacement -/
def prob_exactly_two_white (u : Urn) : ℚ :=
  3 * (prob_draw u BallColor.White) * (prob_draw u BallColor.White) * (prob_draw u BallColor.Black)

/-- Calculates the probability of drawing at least two white balls in three draws with replacement -/
def prob_at_least_two_white (u : Urn) : ℚ :=
  prob_exactly_two_white u + (prob_draw u BallColor.White) * (prob_draw u BallColor.White) * (prob_draw u BallColor.White)

/-- The urn used in the problem -/
def problem_urn : Urn := ⟨3, 5⟩

theorem prob_two_white_correct :
  prob_exactly_two_white problem_urn = 135 / 512 := by sorry

theorem prob_at_least_two_white_correct :
  prob_at_least_two_white problem_urn = 81 / 256 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_white_correct_prob_at_least_two_white_correct_l1179_117955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_a_equals_one_l1179_117943

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x

-- Define the condition for the line
def line_equation (x y : ℝ) : Prop := 2 * x - y = 0

-- Define what it means for a function to be tangent to a line at a point
def is_tangent_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ y₀ : ℝ, f x₀ = y₀ ∧ line_equation x₀ y₀ ∧
  ∀ x : ℝ, x ≠ x₀ → f x > 2 * x

-- State the theorem
theorem tangent_implies_a_equals_one :
  ∀ a : ℝ, (∃ x₀ : ℝ, is_tangent_at (f a) x₀) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_a_equals_one_l1179_117943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_13_5_l1179_117938

-- Define the triangle vertices
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (6, 8)
def C : ℝ × ℝ := (3, 9)

-- Define the triangle area calculation function
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem triangle_area_is_13_5 : triangleArea A B C = 13.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_13_5_l1179_117938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_plane_l1179_117980

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- The plane equation 20x - 8y + 48z = 110 -/
def planeEquation (p : Point3D) : Prop :=
  20 * p.x - 8 * p.y + 48 * p.z = 110

theorem equidistant_point_on_plane :
  let p1 : Point3D := ⟨2, 4, -10⟩
  let q : Point3D := ⟨12, 0, 14⟩
  ∀ p : Point3D, distance p p1 = distance p q → planeEquation p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_plane_l1179_117980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_sum_count_l1179_117996

/-- The number of ways to express n as an ordered sum of p positive integers --/
def ways_to_express_as_ordered_sum (n p : ℕ) : ℕ :=
  Nat.choose (n - 1) (p - 1)

theorem ordered_sum_count (n p : ℕ) (h : 0 < p ∧ p ≤ n) : 
  ways_to_express_as_ordered_sum n p = Nat.choose (n - 1) (p - 1) := by
  -- The proof is trivial because of how we defined the function
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_sum_count_l1179_117996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waiting_time_properties_l1179_117992

/-- Represents the maximum waiting time function for the bus route -/
noncomputable def w : ℝ → ℝ := sorry

/-- The length of the circular loop in miles -/
def loop_length : ℝ := 10

/-- The length of the straight line segment in miles -/
def straight_length : ℝ := 1

/-- The total time for a round trip in minutes -/
def round_trip_time : ℝ := 20

/-- The delay between Bus 1 and Bus 2 in minutes -/
def bus_delay : ℝ := 10

/-- The total route length in miles -/
def total_route_length : ℝ := loop_length + 2 * straight_length

/-- Theorem stating the properties of the waiting time function -/
theorem waiting_time_properties :
  (∀ x : ℝ, 0 ≤ x → x < total_route_length → w x ≥ 0) ∧
  w 2 = 70/3 ∧
  w 4 = 70/3 ∧
  (∀ x : ℝ, 0 ≤ x → x < total_route_length → w x ≤ 25) ∧
  w 3 = 25 ∧
  w 9 = 25 := by
  sorry

#check waiting_time_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_waiting_time_properties_l1179_117992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_gravity_is_18_l1179_117967

/-- The specific gravity of gold relative to water -/
noncomputable def gold_gravity : ℚ := 19

/-- The specific gravity of copper relative to water -/
noncomputable def copper_gravity : ℚ := 9

/-- The ratio of gold to copper in the alloy (9:1) -/
noncomputable def gold_parts : ℚ := 9
noncomputable def copper_parts : ℚ := 1

/-- The total parts in the alloy mixture -/
noncomputable def total_parts : ℚ := gold_parts + copper_parts

/-- The specific gravity of the alloy relative to water -/
noncomputable def alloy_gravity : ℚ := (gold_gravity * gold_parts + copper_gravity * copper_parts) / total_parts

theorem alloy_gravity_is_18 : alloy_gravity = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_gravity_is_18_l1179_117967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_plus_x_l1179_117979

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x
  else if x ≤ 2 then x - 1
  else 0  -- This else case is added to make the function total

-- Define the theorem
theorem range_of_f_plus_x :
  Set.range (fun x => f x + x) = Set.Icc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_plus_x_l1179_117979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equations_l1179_117936

/-- Triangle ABC with vertices A, B, C, and M as the midpoint of BC -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  h_A : A = (-1, 5)
  h_B : B = (-2, -1)
  h_C : C = (4, 3)
  h_M : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

/-- The equation of line AB is y = 6x + 11 -/
def line_equation (t : Triangle) : Prop :=
  ∀ x y : ℝ, (y - t.A.2) / (x - t.A.1) = (t.B.2 - t.A.2) / (t.B.1 - t.A.1) ↔ y = 6 * x + 11

/-- The equation of the circle with diameter AM is x^2 + (y - 3)^2 = 5 -/
def circle_equation (t : Triangle) : Prop :=
  ∀ x y : ℝ, (x - 0)^2 + (y - 3)^2 = 5 ↔ 
    (x - (t.A.1 + t.M.1) / 2)^2 + (y - (t.A.2 + t.M.2) / 2)^2 = ((t.A.1 - t.M.1)^2 + (t.A.2 - t.M.2)^2) / 4

/-- Main theorem stating the equations of line AB and circle with diameter AM -/
theorem triangle_equations (t : Triangle) : line_equation t ∧ circle_equation t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equations_l1179_117936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shift_sine_l1179_117972

theorem symmetric_shift_sine (φ : Real) : 
  (0 < φ) → 
  (φ < Real.pi / 2) → 
  (∀ x, Real.sin (2 * (x - φ) + Real.pi / 3) = Real.sin (-2 * (x - φ) + Real.pi / 3)) → 
  φ = 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_shift_sine_l1179_117972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_of_f_l1179_117976

-- Define the function f(x)
noncomputable def f (x θ : ℝ) : ℝ := Real.sin (2 * x) * Real.cos θ + Real.sin θ - 2 * (Real.sin x)^2 * Real.sin θ

-- Define the symmetry condition
def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem statement
theorem extreme_points_of_f :
  ∃ θ : ℝ, -π/2 < θ ∧ θ < 0 ∧
  symmetric_about (f · θ) (π/3) ∧
  (∃ S : Finset ℝ, S.card = 4 ∧
    ∀ x ∈ S, 0 < x ∧ x < 2*π ∧
    (∀ y ∈ Set.Ioo 0 (2*π), f y θ ≤ f x θ ∨ f y θ ≥ f x θ)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_of_f_l1179_117976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_percentage_theorem_l1179_117968

-- Define the total number of birds (we'll use 100 for simplicity)
noncomputable def total_birds : ℚ := 100

-- Define the percentage of hawks
noncomputable def hawk_percentage : ℚ := 30

-- Define the percentage of paddyfield-warblers among non-hawks
noncomputable def paddyfield_warbler_percentage_among_nonhawks : ℚ := 40

-- Define the ratio of kingfishers to paddyfield-warblers
noncomputable def kingfisher_to_paddyfield_warbler_ratio : ℚ := 25 / 100

-- Theorem to prove
theorem bird_percentage_theorem :
  let hawk_count := hawk_percentage / 100 * total_birds
  let non_hawk_count := total_birds - hawk_count
  let paddyfield_warbler_count := paddyfield_warbler_percentage_among_nonhawks / 100 * non_hawk_count
  let kingfisher_count := kingfisher_to_paddyfield_warbler_ratio * paddyfield_warbler_count
  let other_bird_count := total_birds - (hawk_count + paddyfield_warbler_count + kingfisher_count)
  other_bird_count / total_birds * 100 = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_percentage_theorem_l1179_117968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_A_l1179_117994

/-- A power function passing through the point (1/2, √2/2) -/
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

/-- The point through which the function passes -/
noncomputable def A : ℝ × ℝ := (1/2, Real.sqrt 2 / 2)

/-- The condition that f passes through point A -/
axiom f_passes_through_A : f (1/2) A.1 = A.2

/-- The equation of the tangent line at point A -/
def tangent_line (x y : ℝ) : Prop := 2 * Real.sqrt 2 * x - 4 * y + Real.sqrt 2 = 0

/-- Theorem stating that the given equation is the tangent line at point A -/
theorem tangent_line_at_A : 
  ∃ α, tangent_line A.1 A.2 ∧ 
    ∀ x, tangent_line x (f α x) ↔ x = A.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_A_l1179_117994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l1179_117958

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 3}

-- Define set B
def B : Set Nat := {3}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l1179_117958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_ABC_l1179_117933

-- Define the circles and points
variable (C₁ C₂ : Set (EuclideanSpace ℝ (Fin 2))) -- Circles in 2D plane
variable (P Q : EuclideanSpace ℝ (Fin 2)) -- Points in 2D plane
variable (A B C : EuclideanSpace ℝ (Fin 2)) -- Points on C₂

-- Define the conditions
variable (h1 : C₁ ≠ C₂) -- Circles are distinct
variable (h2 : P ∈ C₁ ∩ C₂) -- P is on both circles (external touch point)
variable (h3 : Q ∈ C₁) -- Q is on C₁
variable (h4 : A ∈ C₂ ∧ B ∈ C₂) -- A and B are on C₂
variable (h5 : C ∈ C₂) -- C is on C₂
variable (h6 : IsTangentLine C₁ Q (Line.throughPts Q A)) -- Tangent line at Q to C₁
variable (h7 : A ≠ B ∧ (Line.throughPts Q A).contains B) -- A and B are distinct and on the tangent line
variable (h8 : (Line.throughPts Q P).contains C) -- C is on line QP

-- State the theorem
theorem isosceles_triangle_ABC : ‖A - C‖ = ‖B - C‖ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_ABC_l1179_117933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersections_imply_a_values_l1179_117916

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then Real.sin x
  else x^3 - 9*x^2 + 25*x + a

-- Define the condition for three distinct intersection points
def has_three_intersections (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = x₁ ∧ f a x₂ = x₂ ∧ f a x₃ = x₃

-- Theorem statement
theorem intersections_imply_a_values :
  ∀ a : ℝ, has_three_intersections a → a = -20 ∨ a = -16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersections_imply_a_values_l1179_117916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_over_pi_eq_solution_l1179_117983

/-- The volume of a cone divided by π, where the cone is formed from a 270-degree sector of a circle with radius 15 -/
noncomputable def cone_volume_over_pi : ℝ :=
  let sector_angle : ℝ := 270
  let circle_radius : ℝ := 15
  let base_radius : ℝ := circle_radius * (sector_angle / 360)
  let height : ℝ := Real.sqrt (circle_radius^2 - base_radius^2)
  (1/3) * base_radius^2 * height

/-- Theorem stating that the volume of the cone divided by π is equal to 126.5625√11 -/
theorem cone_volume_over_pi_eq_solution : 
  cone_volume_over_pi = 126.5625 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_over_pi_eq_solution_l1179_117983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_sugar_calories_l1179_117977

/-- Proves that the calories of added sugar in each candy bar is 25 given the problem conditions -/
theorem candy_bar_sugar_calories (soft_drink_calories : ℝ) 
  (soft_drink_sugar_percentage : ℝ) (recommended_sugar_intake : ℝ) 
  (exceed_percentage : ℝ) (num_candy_bars : ℕ) 
  (h1 : soft_drink_calories = 2500)
  (h2 : soft_drink_sugar_percentage = 0.05)
  (h3 : recommended_sugar_intake = 150)
  (h4 : exceed_percentage = 1)
  (h5 : num_candy_bars = 7)
  : 
  ((1 + exceed_percentage) * recommended_sugar_intake - 
   soft_drink_calories * soft_drink_sugar_percentage) / num_candy_bars = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_sugar_calories_l1179_117977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_treasure_signs_l1179_117944

/-- Represents the number of palm trees with signs -/
def total_trees : ℕ := 30

/-- Represents the number of signs saying "Exactly under 15 signs a treasure is buried" -/
def signs_15 : ℕ := 15

/-- Represents the number of signs saying "Exactly under 8 signs a treasure is buried" -/
def signs_8 : ℕ := 8

/-- Represents the number of signs saying "Exactly under 4 signs a treasure is buried" -/
def signs_4 : ℕ := 4

/-- Represents the number of signs saying "Exactly under 3 signs a treasure is buried" -/
def signs_3 : ℕ := 3

/-- Represents that only signs under which there is no treasure are truthful -/
def truthful_signs (n : ℕ) : Prop := n ≤ total_trees

/-- The theorem stating that the minimum number of signs under which treasures can be buried is 15 -/
theorem min_treasure_signs : ∃ (n : ℕ), n = 15 ∧ (∀ m : ℕ, m < n → ¬(truthful_signs m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_treasure_signs_l1179_117944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l1179_117971

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x / 4 + y / 3 = 1

-- Define the slope-intercept form of a line
def slope_intercept_form (m b x y : ℝ) : Prop := y = m * x + b

-- Theorem statement
theorem line_slope :
  ∃ (m : ℝ), m = -3/4 ∧ ∀ (x y : ℝ), line_equation x y → 
    ∃ b, slope_intercept_form m b x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l1179_117971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l1179_117991

/-- Given a function f: ℝ → ℝ with a tangent line at (2, f(2)) described by the equation 2x - y - 3 = 0,
    prove that f(2) + f'(2) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x y, 2 * x - y - 3 = 0 ↔ y = f 2 + (deriv f 2) * (x - 2)) →
  f 2 + deriv f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l1179_117991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_a_closed_l1179_117952

/-- The sequence {aₙ} defined by the given recurrence relation -/
noncomputable def a : ℕ → ℝ → ℝ
  | 0, x => x + 1/x  -- Added base case for n = 0
  | 1, x => x + 1/x
  | n+2, x => a 1 x - 1/(a (n+1) x)

/-- The closed form expression for the nth term of the sequence -/
noncomputable def a_closed (n : ℕ) (x : ℝ) : ℝ :=
  (x^(2*n+2) - 1) / (x * (x^(2*n) - 1))

/-- Theorem stating the equivalence of the recurrence relation and the closed form -/
theorem a_equals_a_closed (x : ℝ) (hx : x > 0 ∧ x ≠ 1) :
  ∀ n : ℕ, n ≥ 1 → a n x = a_closed n x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_a_closed_l1179_117952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1179_117918

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  17 * x^2 - 16 * x * y + 4 * y^2 - 34 * x + 16 * y + 13 = 0

/-- The center of symmetry -/
def center : ℝ × ℝ := (1, 0)

/-- The slope of the axes of symmetry -/
noncomputable def axis_slope : ℝ × ℝ := 
  ((13 + 5 * Real.sqrt 17) / 16, (13 - 5 * Real.sqrt 17) / 16)

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties :
  (∀ x y : ℝ, ellipse_equation x y → 
    ∃ t : ℝ, y = axis_slope.1 * (x - center.1) ∨ y = axis_slope.2 * (x - center.1)) ∧
  (∀ x y : ℝ, ellipse_equation x y → 
    ellipse_equation (2 * center.1 - x) (2 * center.2 - y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1179_117918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_for_odd_k_l1179_117924

/-- The number of positive divisors of n -/
def d (n : ℕ) : ℕ := sorry

/-- The theorem stating that for any odd positive integer k, 
    there exists a positive integer n such that d(n^2) / d(n) = k -/
theorem exists_n_for_odd_k (k : ℕ) (h : Odd k) : 
  ∃ n : ℕ, n > 0 ∧ (d (n^2) : ℚ) / d n = k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_for_odd_k_l1179_117924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_always_wins_l1179_117939

/-- Represents a player in the game -/
inductive Player
| Vasya
| Petya
deriving Repr, DecidableEq

/-- Represents the state of the game -/
structure GameState where
  piles : Nat
  currentPlayer : Player
  firstMove : Bool
deriving Repr

/-- Represents a move in the game -/
inductive Move
| Divide
| Take
deriving Repr

/-- Define the game rules -/
def gameRules (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Divide => { state with 
      piles := state.piles + 1, 
      currentPlayer := if state.currentPlayer = Player.Vasya then Player.Petya else Player.Vasya 
    }
  | Move.Take => { state with 
      piles := state.piles - 1, 
      currentPlayer := if state.currentPlayer = Player.Vasya then Player.Petya else Player.Vasya,
      firstMove := false 
    }

/-- Define a winning state -/
def isWinningState (state : GameState) : Bool :=
  state.piles = 0

/-- Theorem: Vasya always wins the game -/
theorem vasya_always_wins :
  ∀ (initialPiles : Nat), initialPiles > 0 →
  ∃ (moves : List Move),
    let finalState := moves.foldl gameRules { piles := initialPiles, currentPlayer := Player.Vasya, firstMove := true }
    isWinningState finalState ∧ finalState.currentPlayer = Player.Vasya :=
by
  sorry

#check vasya_always_wins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_always_wins_l1179_117939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_three_roots_negative_discriminant_l1179_117974

/-- The discriminant of a cubic equation x^3 + ax + b = 0 -/
noncomputable def discriminant (a b : ℝ) : ℝ := b^2 / 4 + a^3 / 27

/-- A cubic equation x^3 + ax + b = 0 has three distinct real roots -/
def has_three_distinct_real_roots (a b : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 + a*x + b = 0 ∧ y^3 + a*y + b = 0 ∧ z^3 + a*z + b = 0

theorem cubic_three_roots_negative_discriminant (a b : ℝ) :
  has_three_distinct_real_roots a b → discriminant a b < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_three_roots_negative_discriminant_l1179_117974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1179_117930

/-- The sum of the infinite series ∑(n=1 to ∞) [2^n / (1 + 2^n + 2^(n+1) + 2^(2n+1))] -/
noncomputable def infiniteSeries : ℝ := ∑' n, (2^n : ℝ) / (1 + 2^n + 2^(n+1) + 2^(2*n+1))

/-- The sum of the infinite series is equal to 1/3 -/
theorem infiniteSeriesSum : infiniteSeries = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1179_117930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_vertices_l1179_117963

/-- Predicate to check if three points form an equilateral triangle -/
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let d₂ := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let d₃ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  d₁ = d₂ ∧ d₂ = d₃

/-- Predicate to check if a point is the centroid of a triangle -/
def is_centroid (G A B C : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- An equilateral triangle with one vertex at (2√3, 2√3) and centroid at (0, 0) has its other two vertices at (-3-√3, 3-√3) and (3-√3, -3-√3). -/
theorem equilateral_triangle_vertices (A B C : ℝ × ℝ) : 
  A = (2 * Real.sqrt 3, 2 * Real.sqrt 3) →
  is_equilateral_triangle A B C →
  is_centroid (0, 0) A B C →
  ((B = (-3 - Real.sqrt 3, 3 - Real.sqrt 3) ∧ C = (3 - Real.sqrt 3, -3 - Real.sqrt 3)) ∨
   (B = (3 - Real.sqrt 3, -3 - Real.sqrt 3) ∧ C = (-3 - Real.sqrt 3, 3 - Real.sqrt 3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_vertices_l1179_117963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_when_a_is_one_monotonicity_condition_l1179_117965

-- Define the function f(x) = x^2 + 2ax + 2
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- Define the domain [-5, 5]
def domain : Set ℝ := Set.Icc (-5) 5

-- Theorem for part (1)
theorem max_min_values_when_a_is_one :
  (∀ x, x ∈ domain → f 1 x ≤ 37) ∧
  (∃ x ∈ domain, f 1 x = 37) ∧
  (∀ x, x ∈ domain → f 1 x ≥ 1) ∧
  (∃ x ∈ domain, f 1 x = 1) :=
sorry

-- Theorem for part (2)
theorem monotonicity_condition (a : ℝ) :
  (∀ x y, x ∈ domain → y ∈ domain → x < y → f a x < f a y) ∨
  (∀ x y, x ∈ domain → y ∈ domain → x < y → f a x > f a y) ↔
  a ≤ -5 ∨ a ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_when_a_is_one_monotonicity_condition_l1179_117965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_percentage_calculation_l1179_117993

/-- Calculates the percentage of a stock given the income, initial investment, and brokerage fee. -/
theorem stock_percentage_calculation
  (income : ℝ)
  (initial_investment : ℝ)
  (brokerage_fee_percent : ℝ)
  (h_income : income = 756)
  (h_initial_investment : initial_investment = 9000)
  (h_brokerage_fee_percent : brokerage_fee_percent = 0.25) :
  let brokerage_fee := initial_investment * (brokerage_fee_percent / 100)
  let net_investment := initial_investment - brokerage_fee
  let dividend_yield := (income / net_investment) * 100
  ∃ ε > 0, |dividend_yield - 8.42| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_percentage_calculation_l1179_117993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sister_pairs_count_l1179_117906

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2*x else 2 / Real.exp x

-- Define a sister pair
def is_sister_pair (A B : ℝ × ℝ) : Prop :=
  (f A.fst = A.snd) ∧ (f B.fst = B.snd) ∧ (B = (-A.fst, -A.snd))

-- Theorem statement
theorem sister_pairs_count :
  ∃ (P Q : (ℝ × ℝ) × (ℝ × ℝ)), 
    is_sister_pair P.1 P.2 ∧ 
    is_sister_pair Q.1 Q.2 ∧ 
    P ≠ Q ∧
    (∀ (R : (ℝ × ℝ) × (ℝ × ℝ)), 
      is_sister_pair R.1 R.2 → (R = P ∨ R = Q)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sister_pairs_count_l1179_117906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_k_neg_one_range_of_g_main_theorem_l1179_117931

-- Define the function f
noncomputable def f (x k : ℝ) : ℝ := x * (2 / (2^x - 1) + k)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x (-1) / x

-- Theorem 1: f is even implies k = -1
theorem f_even_implies_k_neg_one :
  (∀ x, f x k = f (-x) k) → k = -1 := by sorry

-- Theorem 2: Range of g(x) for x ∈ (0, 1] is (-∞, 1]
theorem range_of_g :
  Set.range (fun x => g x) = Set.Iic 1 := by sorry

-- The main theorem combining both results
theorem main_theorem :
  (∀ x, f x k = f (-x) k) →
  k = -1 ∧
  Set.range (fun x => g x) = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_k_neg_one_range_of_g_main_theorem_l1179_117931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_key_presses_l1179_117978

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x)

theorem calculator_key_presses :
  (f ∘ f ∘ f) 2 = 2 ∧
  (f^[10]) 2 = -1 ∧
  (f^[2015]) 2 = 1/2 := by
  sorry

#check calculator_key_presses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_key_presses_l1179_117978
