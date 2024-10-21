import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_center_coord_sum_l1282_128255

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The center of a rectangle -/
noncomputable def Rectangle.center (r : Rectangle) : Point :=
  { x := (r.A.x + r.C.x) / 2,
    y := (r.A.y + r.C.y) / 2 }

/-- Sum of coordinates of a point -/
noncomputable def Point.coordSum (p : Point) : ℝ := p.x + p.y

/-- Main theorem -/
theorem rectangle_center_coord_sum 
  (r : Rectangle)
  (h1 : r.A.y = 0 ∧ r.B.y = 0)  -- AB is on x-axis
  (h2 : r.A.x = 6 ∧ r.B.x = 10)  -- (6,0) and (10,0) on AB
  (h3 : r.D.x = 2 ∧ r.D.y = 0)  -- (2,0) on AD
  (h4 : r.C.x = 2 ∧ r.C.y > 0)  -- (2,y) where y > 0 on CD
  : r.center.coordSum = 8 + r.C.y / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_center_coord_sum_l1282_128255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2_cos_3_tan_4_negative_l1282_128295

theorem sin_2_cos_3_tan_4_negative : Real.sin 2 * Real.cos 3 * Real.tan 4 < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2_cos_3_tan_4_negative_l1282_128295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bounds_l1282_128211

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 + 3

/-- The line function -/
def line (r : ℝ) : ℝ → ℝ := λ x ↦ r

/-- The area of the triangle formed by the vertex of the parabola and its intersections with the line y = r -/
noncomputable def triangleArea (r : ℝ) : ℝ := (r - 3) * Real.sqrt (r - 3)

theorem triangle_area_bounds (r : ℝ) :
  (12 : ℝ) ≤ triangleArea r ∧ triangleArea r ≤ 48 → r ∈ Set.Icc 15 19 := by
  sorry

#check triangle_area_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bounds_l1282_128211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_alpha_gamma_sum_exists_min_alpha_gamma_l1282_128275

noncomputable section

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function f
def f (α γ : ℂ) (z : ℂ) : ℂ := (3 - 2*i) * z^2 + α * z + γ

-- Main theorem
theorem min_alpha_gamma_sum (α γ : ℂ) : 
  (f α γ 1).im = 0 → (f α γ (-i)).im = 0 → Complex.abs α + Complex.abs γ ≥ Real.sqrt 13 :=
by sorry

-- Existence of α and γ that achieve the minimum
theorem exists_min_alpha_gamma : 
  ∃ α γ : ℂ, (f α γ 1).im = 0 ∧ (f α γ (-i)).im = 0 ∧ Complex.abs α + Complex.abs γ = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_alpha_gamma_sum_exists_min_alpha_gamma_l1282_128275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_arrangement_ends_with_B_l1282_128243

/-- Represents a piece in the number arrangement puzzle -/
inductive Piece
  | A  -- Blank piece (assumed as 0)
  | Four
  | B  -- 8
  | C  -- Blank piece (assumed as 0)
  | D  -- 59
  | E  -- 107
deriving DecidableEq

/-- Returns the numeric value of a piece -/
def pieceValue : Piece → Nat
  | Piece.A => 0
  | Piece.Four => 4
  | Piece.B => 8
  | Piece.C => 0
  | Piece.D => 59
  | Piece.E => 107

/-- Represents an arrangement of pieces -/
def Arrangement := List Piece

/-- Converts an arrangement to a natural number -/
def arrangementToNat (arr : Arrangement) : Nat :=
  arr.foldl (fun acc p => acc * 1000 + pieceValue p) 0

/-- Checks if an arrangement is valid (uses all pieces exactly once) -/
def isValidArrangement (arr : Arrangement) : Prop :=
  arr.length = 5 ∧ 
  arr.toFinset = {Piece.A, Piece.Four, Piece.B, Piece.C, Piece.D, Piece.E}

/-- The main theorem: The smallest valid arrangement ends with piece B -/
theorem smallest_arrangement_ends_with_B :
  ∃ (arr : Arrangement), isValidArrangement arr ∧
  (∀ (other : Arrangement), isValidArrangement other → 
    arrangementToNat arr ≤ arrangementToNat other) ∧
  arr.getLast? = some Piece.B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_arrangement_ends_with_B_l1282_128243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1282_128244

/-- The function f(x) = ax³ - bx² -/
def f (a b x : ℝ) : ℝ := a * x^3 - b * x^2

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 - 2 * b * x

theorem range_of_f (a b : ℝ) :
  (f_derivative a b 1 = -1) →
  (f a b 1 = 0) →
  (∀ x, x ∈ Set.Icc (-1/2 : ℝ) (3/2 : ℝ) → 
    f a b x ∈ Set.Icc (-9/8 : ℝ) (3/8 : ℝ)) ∧
  (∃ x₁ x₂, x₁ ∈ Set.Icc (-1/2 : ℝ) (3/2 : ℝ) ∧ 
    x₂ ∈ Set.Icc (-1/2 : ℝ) (3/2 : ℝ) ∧
    f a b x₁ = -9/8 ∧ f a b x₂ = 3/8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1282_128244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_theorem_trihedral_angle_l1282_128258

/-- Predicate to represent a valid trihedral angle -/
def IsTrihedralAngle (α β γ A B C : Real) : Prop :=
  0 < α ∧ α < Real.pi ∧
  0 < β ∧ β < Real.pi ∧
  0 < γ ∧ γ < Real.pi ∧
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  α + β + γ < 2 * Real.pi

/-- Sine theorem for a trihedral angle -/
theorem sine_theorem_trihedral_angle 
  (α β γ A B C : Real) 
  (h_trihedral : IsTrihedralAngle α β γ A B C) :
  (Real.sin α) / (Real.sin A) = (Real.sin β) / (Real.sin B) ∧ 
  (Real.sin β) / (Real.sin B) = (Real.sin γ) / (Real.sin C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_theorem_trihedral_angle_l1282_128258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_15_l1282_128203

/-- Represents the sum of the first n terms of a geometric sequence --/
def GeometricSum (n : ℕ) : ℝ := sorry

/-- Given a geometric sequence where the sum of the first 5 terms is 10
    and the sum of the first 10 terms is 50, prove that the sum of the
    first 15 terms is 210 --/
theorem geometric_sum_15 (h1 : GeometricSum 5 = 10) (h2 : GeometricSum 10 = 50) :
  GeometricSum 15 = 210 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_15_l1282_128203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_gt_zero_neither_sufficient_nor_necessary_l1282_128237

-- Define an arithmetic sequence
noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def S (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n - 1) / 2 : ℝ) * d

-- Theorem statement
theorem d_gt_zero_neither_sufficient_nor_necessary
  (a₁ : ℝ) (d : ℝ) :
  ¬(∀ n : ℕ, d > 0 → S a₁ d (n + 1) > S a₁ d n) ∧
  ¬(∀ n : ℕ, S a₁ d (n + 1) > S a₁ d n → d > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_gt_zero_neither_sufficient_nor_necessary_l1282_128237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_eight_l1282_128272

theorem power_of_eight (x : ℝ) : (8 : ℝ)^(3*x) = 512 → (8 : ℝ)^(-x) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_eight_l1282_128272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_to_second_layer_ratio_l1282_128235

/-- Represents the amount of sugar required for a cake layer -/
structure SugarAmount where
  amount : ℕ
  deriving Repr

/-- Represents the size of a cake layer relative to the smallest layer -/
structure LayerSize where
  size : ℕ
  deriving Repr

/-- The amount of sugar required for the smallest layer -/
def smallest_layer_sugar : SugarAmount := ⟨2⟩

/-- The size of the second layer relative to the smallest layer -/
def second_layer_size : LayerSize := ⟨2⟩

/-- The amount of sugar required for the third layer -/
def third_layer_sugar : SugarAmount := ⟨12⟩

/-- Theorem: The ratio of the size of the third layer to the second layer is 3 -/
theorem third_to_second_layer_ratio :
  (third_layer_sugar.amount / (smallest_layer_sugar.amount * second_layer_size.size) : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_to_second_layer_ratio_l1282_128235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_line_equation_l1282_128264

/-- Given a line with equation y = 3/4x + 6, we define a new line M with the following properties:
    - The slope of M is one-third of the original line's slope
    - The y-intercept of M is thrice the original line's y-intercept
    - The x-intercept of M is half the original line's x-intercept
    - M passes through the point (12, 0)
    This theorem proves that the equation of line M is y = 1/4x + 17 -/
theorem new_line_equation :
  let original_line := λ x : ℝ => (3/4) * x + 6
  let original_slope := 3/4
  let original_y_intercept := 6
  let original_x_intercept := -8
  let new_slope := (1/3) * original_slope
  let new_y_intercept := 3 * original_y_intercept
  let new_x_intercept := (1/2) * original_x_intercept
  let line_M := λ x : ℝ => new_slope * x + 17
  (∀ x : ℝ, original_line x = (3/4) * x + 6) →
  new_slope = 1/4 →
  new_y_intercept = 18 →
  new_x_intercept = -4 →
  line_M 12 = 0 →
  ∀ x : ℝ, line_M x = (1/4) * x + 17 :=
by
  intros
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_line_equation_l1282_128264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_current_rate_l1282_128294

/-- Represents the speed of a rower in still water -/
noncomputable def rower_speed : ℝ → ℝ := sorry

/-- Represents the speed of the river current -/
noncomputable def river_current : ℝ := sorry

/-- Time taken for downstream journey -/
noncomputable def time_downstream (r : ℝ) : ℝ := 20 / (rower_speed r + river_current)

/-- Time taken for upstream journey -/
noncomputable def time_upstream (r : ℝ) : ℝ := 20 / (rower_speed r - river_current)

/-- Theorem stating the rate of the river's current -/
theorem river_current_rate : 
  (∃ r : ℝ, time_upstream r = time_downstream r + 3) ∧ 
  (∃ r : ℝ, time_upstream (3 * r) = time_downstream (3 * r) + 2) → 
  river_current = 10 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_current_rate_l1282_128294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EQ_length_is_ten_l1282_128284

/-- Represents a trapezoid EFGH with a circle centered at Q on EF and tangent to FG and HE -/
structure TrapezoidWithCircle where
  -- Points of the trapezoid
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  -- Center of the circle
  Q : ℝ × ℝ
  -- EF is parallel to GH
  parallel_EF_GH : (F.1 - E.1) * (H.2 - G.2) = (F.2 - E.2) * (H.1 - G.1)
  -- Lengths of sides
  EF_length : Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) = 105
  FG_length : Real.sqrt ((G.1 - F.1)^2 + (G.2 - F.2)^2) = 57
  GH_length : Real.sqrt ((H.1 - G.1)^2 + (H.2 - G.2)^2) = 22
  HE_length : Real.sqrt ((E.1 - H.1)^2 + (E.2 - H.2)^2) = 80
  -- Q is on EF
  Q_on_EF : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (E.1 + t * (F.1 - E.1), E.2 + t * (F.2 - E.2))
  -- Circle is tangent to FG and HE
  tangent_to_FG : ∃ P : ℝ × ℝ, ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ 
    P = (F.1 + s * (G.1 - F.1), F.2 + s * (G.2 - F.2)) ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt ((F.1 - Q.1)^2 + (F.2 - Q.2)^2)
  tangent_to_HE : ∃ R : ℝ × ℝ, ∃ u : ℝ, 0 ≤ u ∧ u ≤ 1 ∧ 
    R = (H.1 + u * (E.1 - H.1), H.2 + u * (E.2 - H.2)) ∧
    Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = Real.sqrt ((F.1 - Q.1)^2 + (F.2 - Q.2)^2)

/-- The main theorem to prove -/
theorem EQ_length_is_ten (t : TrapezoidWithCircle) : 
  Real.sqrt ((t.Q.1 - t.E.1)^2 + (t.Q.2 - t.E.2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_EQ_length_is_ten_l1282_128284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_minimum_implies_a_range_l1282_128263

noncomputable def sequence_a (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 5 then n + 15 / (n : ℝ)
  else a * Real.log ↑n - 1 / 4

theorem sequence_minimum_implies_a_range (a : ℝ) :
  (∀ n : ℕ, sequence_a a n ≥ 31 / 4) →
  a ≥ 8 / Real.log 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_minimum_implies_a_range_l1282_128263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1282_128208

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ)
  (h1 : -π < φ ∧ φ < 0)
  (h2 : ∀ x, f x φ = f (π/4 - x) φ)  -- Symmetry axis at x = π/8
  (h3 : f 0 φ < 0) :
  (φ = -3*π/4) ∧
  (∀ k : ℤ, ∀ x y : ℝ, 
    x ∈ Set.Icc (5*π/8 + k*π) (9*π/8 + k*π) →
    y ∈ Set.Icc (5*π/8 + k*π) (9*π/8 + k*π) →
    x ≤ y → f x φ ≥ f y φ) ∧
  (Set.range (fun x => f x φ) = Set.Icc (-Real.sqrt 2) 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1282_128208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sequence_cosine_sum_l1282_128202

theorem triangle_geometric_sequence_cosine_sum (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi → -- Triangle conditions
  0 < a ∧ 0 < b ∧ 0 < c → -- Positive side lengths
  b^2 = a * c → -- Geometric sequence condition
  Real.cos (A - C) + Real.cos B + Real.cos (2 * B) = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sequence_cosine_sum_l1282_128202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clear_time_approx_l1282_128217

/-- The time for two trains to be completely clear of each other -/
noncomputable def train_clear_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let total_distance := length1 + length2
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  total_distance / relative_speed

/-- Theorem: The time for two trains to be completely clear of each other is approximately 7.35 seconds -/
theorem train_clear_time_approx :
  ∃ ε > 0, |train_clear_time 131 165 80 65 - 7.35| < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_clear_time 131 165 80 65

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clear_time_approx_l1282_128217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l1282_128293

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space. -/
def Point := ℝ × ℝ

/-- Distance between two points. -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Checks if a point is on a circle. -/
def isOnCircle (c : Circle) (p : Point) : Prop :=
  distance c.center p = c.radius

/-- Checks if a line is tangent to a circle at a given point. -/
def isTangent (c : Circle) (p m : Point) : Prop :=
  isOnCircle c p ∧ ∀ q : Point, q ≠ p → isOnCircle c q → distance m p < distance m q

/-- Main theorem statement. -/
theorem tangent_condition (c : Circle) (a b m c' : Point) :
  isOnCircle c a →
  isOnCircle c b →
  isOnCircle c c' →
  (distance m c')^2 = (distance m a) * (distance m b) →
  isTangent c c' m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l1282_128293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_convex_pentagon_division_exists_l1282_128286

/-- A pentagon is a polygon with 5 sides -/
structure Pentagon where
  vertices : Fin 5 → ℝ × ℝ

/-- Calculate the interior angle of a pentagon at a given vertex -/
noncomputable def interiorAngle (p : Pentagon) (i : Fin 5) : ℝ := sorry

/-- A non-convex pentagon has at least one interior angle greater than 180° -/
def NonConvexPentagon (p : Pentagon) : Prop :=
  ∃ (i : Fin 5), interiorAngle p i > Real.pi

/-- Calculate the area of a pentagon -/
noncomputable def area (p : Pentagon) : ℝ := sorry

/-- Two pentagons are equal if they have the same area -/
def EqualPentagons (p1 p2 : Pentagon) : Prop :=
  area p1 = area p2

/-- A division of a pentagon is a line that cuts it into two parts -/
structure PentagonDivision (p : Pentagon) where
  line : ℝ × ℝ → ℝ × ℝ
  part1 : Pentagon
  part2 : Pentagon

/-- A valid division results in two equal pentagons -/
def ValidDivision (p : Pentagon) (d : PentagonDivision p) : Prop :=
  EqualPentagons d.part1 d.part2 ∧ NonConvexPentagon d.part1 ∧ NonConvexPentagon d.part2

/-- Theorem: For any non-convex pentagon, there exists a valid division into two equal non-convex pentagons -/
theorem non_convex_pentagon_division_exists :
  ∀ (p : Pentagon), NonConvexPentagon p → ∃ (d : PentagonDivision p), ValidDivision p d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_convex_pentagon_division_exists_l1282_128286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_divisible_by_four_l1282_128256

theorem sum_of_squares_divisible_by_four (a b c : ℕ) :
  (a^2 + b^2 + c^2) % 4 = 0 ↔ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_divisible_by_four_l1282_128256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_factorials_perfect_square_l1282_128268

def factorial (n : ℕ) : ℕ := Nat.factorial n

def sum_factorials (n : ℕ) : ℕ := Finset.sum (Finset.range n) (λ i => factorial (i + 1))

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem sum_factorials_perfect_square :
  ∀ n : ℕ, n > 0 → (is_perfect_square (sum_factorials n) ↔ n = 1 ∨ n = 3) :=
by
  sorry

#check sum_factorials_perfect_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_factorials_perfect_square_l1282_128268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_ratio_row_l1282_128234

/-- Pascal's Triangle binomial coefficient -/
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  match n, k with
  | _, 0 => 1
  | 0, _ => 0
  | n+1, k+1 => pascal n k + pascal n (k+1)
termination_by pascal n k => n

/-- Checks if three consecutive entries in a row of Pascal's Triangle are in ratio 5:6:7 -/
def hasRatio567 (n : ℕ) (r : ℕ) : Prop :=
  7 * pascal n r = 6 * pascal n (r+1) ∧
  7 * pascal n (r+1) = 6 * pascal n (r+2)

theorem pascal_ratio_row :
  ∃ (r : ℕ), hasRatio567 142 r :=
by
  use 64
  sorry

#eval pascal 142 64
#eval pascal 142 65
#eval pascal 142 66

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_ratio_row_l1282_128234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_constant_ratio_l1282_128251

-- Define the sum of first n terms of an arithmetic sequence
noncomputable def T (b : ℝ) (n : ℕ+) : ℝ := (n : ℝ) * (2 * b + (n - 1 : ℝ) * 4) / 2

-- State the theorem
theorem first_term_of_constant_ratio (b : ℝ) :
  (∃ d : ℝ, ∀ n : ℕ+, T b (4 * n) / T b n = d) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_constant_ratio_l1282_128251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_60_degrees_l1282_128240

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)

-- Define the angle B in radians
noncomputable def angle_B (t : Triangle) : ℝ := Real.arccos ((t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c))

-- State the theorem
theorem angle_B_is_60_degrees (t : Triangle) 
  (h : (t.b + t.c) * (t.b - t.c) = t.a * (t.a - t.c)) : 
  angle_B t = π / 3 := by
  sorry

#check angle_B_is_60_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_60_degrees_l1282_128240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_proof_l1282_128246

noncomputable def smallest_angle_satisfying_equation : ℝ := 90 / 7

theorem smallest_angle_proof (x : ℝ) :
  (x > 0 ∧ Real.sin (3 * x) * Real.sin (4 * x) = Real.cos (3 * x) * Real.cos (4 * x)) →
  x ≥ smallest_angle_satisfying_equation :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_proof_l1282_128246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dinosaur_model_cost_l1282_128209

def model_a_price : ℕ := 100
def model_b_price : ℕ := 120
def model_c_price : ℕ := 140

def kindergarten_count : ℕ := 2
def elementary_count : ℕ := 2 * kindergarten_count
def high_school_count : ℕ := 3 * kindergarten_count

def total_cost : ℕ := 
  model_a_price * kindergarten_count + 
  model_b_price * elementary_count + 
  model_c_price * high_school_count

def discount_rate (cost : ℕ) : ℚ :=
  if cost > 2000 then 15/100
  else if cost > 1000 then 10/100
  else if cost > 500 then 5/100
  else 0

theorem dinosaur_model_cost : 
  Int.floor (↑total_cost * (1 - discount_rate total_cost)) = 1368 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dinosaur_model_cost_l1282_128209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_pair_in_thirteen_reals_l1282_128201

theorem existence_of_pair_in_thirteen_reals (S : Finset ℝ) (h : S.card = 13) :
  ∃ c d, c ∈ S ∧ d ∈ S ∧ c ≠ d ∧ 0 < (c - d) / (1 + c * d) ∧ (c - d) / (1 + c * d) < 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_pair_in_thirteen_reals_l1282_128201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_sum_25_l1282_128205

/-- A three-digit number -/
def ThreeDigitNumber : Type := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- The set of three-digit numbers whose digits sum to 25 -/
def digitsSum25 : Set ThreeDigitNumber :=
  { n | sumOfDigits n.val = 25 }

/-- Prove that digitsSum25 is finite -/
instance : Fintype digitsSum25 := by
  sorry

/-- The main theorem stating that there are 6 three-digit numbers whose digits sum to 25 -/
theorem count_three_digit_sum_25 : Fintype.card digitsSum25 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_sum_25_l1282_128205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_l1282_128231

theorem cosine_sum (α β : ℝ) : 
  0 < α ∧ α < π / 2 →
  -π / 2 < β ∧ β < 0 →
  Real.cos (π / 4 + α) = 1 / 3 →
  Real.cos (π / 4 - β) = Real.sqrt 3 / 3 →
  Real.cos (α + β) = 5 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_l1282_128231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_sin_sum_l1282_128281

/-- The minimum positive period of sin(π/3 - 2x) + sin(2x) is π -/
theorem min_period_sin_sum : 
  ∃ T : ℝ, T > 0 ∧ 
  (∀ x : ℝ, Real.sin (π/3 - 2*x) + Real.sin (2*x) = Real.sin (π/3 - 2*(x + T)) + Real.sin (2*(x + T))) ∧ 
  (∀ S : ℝ, S > 0 → 
    (∀ x : ℝ, Real.sin (π/3 - 2*x) + Real.sin (2*x) = Real.sin (π/3 - 2*(x + S)) + Real.sin (2*(x + S))) 
    → T ≤ S) ∧
  T = π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_sin_sum_l1282_128281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_inscribed_l1282_128282

/-- An equilateral triangle with inscribed equilateral triangle -/
structure TriangleWithInscribed where
  -- The side length of the outer equilateral triangle
  side : ℝ
  -- The side length of the inscribed equilateral triangle
  inner_side : ℝ
  -- Condition that the inner triangle has side length 10
  inner_side_eq : inner_side = 10

/-- The area of an equilateral triangle with an inscribed equilateral triangle -/
noncomputable def area (t : TriangleWithInscribed) : ℝ :=
  (Real.sqrt 3 / 4) * t.side^2

/-- Theorem: The area of the described triangle is 400√3 / 9 -/
theorem area_of_triangle_with_inscribed (t : TriangleWithInscribed) : 
  area t = 400 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_with_inscribed_l1282_128282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_sum_of_triangle_l1282_128267

/-- The line equation forming a triangle with coordinate axes -/
def line_equation (x y : ℝ) : Prop := 10 * x + 3 * y = 30

/-- The triangle formed by the line and coordinate axes -/
def triangle := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ line_equation p.1 p.2}

/-- The sum of the lengths of the altitudes of the triangle -/
noncomputable def altitude_sum : ℝ := 13 + 30 / Real.sqrt 109

/-- Theorem stating that the sum of the lengths of the altitudes of the triangle
    formed by the line 10x + 3y = 30 and the coordinate axes is 13 + 30/√109 -/
theorem altitude_sum_of_triangle : 
  ∃ (h₁ h₂ h₃ : ℝ), 
    h₁ + h₂ + h₃ = altitude_sum ∧ 
    (∀ p ∈ triangle, h₁ ≤ dist p ⟨3, 0⟩) ∧
    (∀ p ∈ triangle, h₂ ≤ dist p ⟨0, 10⟩) ∧
    (∀ p ∈ triangle, h₃ ≤ p.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_sum_of_triangle_l1282_128267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1282_128232

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := |x + 8/m| + |x - 2*m|

theorem f_properties (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, f m x ≥ 8) ∧
  (f m 1 > 10 ↔ m ∈ Set.Ioo 0 1 ∨ m ∈ Set.Ioi 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1282_128232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_distribution_l1282_128228

theorem cookie_distribution (n : Nat) : 
  n = 540 → 
  (Finset.filter (fun k => k > 0 ∧ k ≤ 180 ∧ n % k = 0 ∧ n / k ≥ 3) (Finset.range (n + 1))).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_distribution_l1282_128228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l1282_128274

theorem triangle_side_ratio_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Triangle is acute
  A + B + C = π ∧          -- Sum of angles in a triangle
  B = 2 * A ∧              -- Given condition
  Real.sin A / a = Real.sin B / b ∧  -- Law of Sines
  Real.sin A / a = Real.sin C / c →  -- Law of Sines
  Real.sqrt 2 < b / a ∧ b / a < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l1282_128274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_inequality_l1282_128254

theorem lcm_inequality (n : ℕ) 
  (h : ∀ k : ℕ, k ∈ Finset.range 35 → Nat.lcm n (n + k) > Nat.lcm n (n + k + 1)) :
  Nat.lcm n (n + 35) > Nat.lcm n (n + 36) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_inequality_l1282_128254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PZ_equals_eight_l1282_128200

-- Define the triangles and points
variable (X Y Z P Q R : ℝ × ℝ)

-- Define the side lengths of triangle XYZ
def XY : ℝ := 20
def YZ : ℝ := 26
def XZ : ℝ := 22

-- Define the conditions for inscribed triangle PQR
def P_on_YZ (X Y Z P : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • Y + t • Z
def Q_on_XZ (X Y Z Q : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • X + t • Z
def R_on_XY (X Y Z R : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (1 - t) • X + t • Y

-- Define the equality of arcs
def arc_RY_eq_QZ (R Y Q Z : ℝ × ℝ) : Prop := ‖R - Y‖ = ‖Q - Z‖
def arc_RX_eq_PY (R X P Y : ℝ × ℝ) : Prop := ‖R - X‖ = ‖P - Y‖
def arc_QX_eq_PZ (Q X P Z : ℝ × ℝ) : Prop := ‖Q - X‖ = ‖P - Z‖

-- State the theorem
theorem PZ_equals_eight 
  (h1 : P_on_YZ X Y Z P) 
  (h2 : Q_on_XZ X Y Z Q) 
  (h3 : R_on_XY X Y Z R) 
  (h4 : arc_RY_eq_QZ R Y Q Z) 
  (h5 : arc_RX_eq_PY R X P Y) 
  (h6 : arc_QX_eq_PZ Q X P Z) : 
  ‖P - Z‖ = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PZ_equals_eight_l1282_128200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_length_l1282_128278

-- Define the circle and points
variable (O X Y Z A B : EuclideanSpace ℝ 2)
variable (r : ℝ)

-- Define the conditions
axiom on_circle : (dist O X = r) ∧ (dist O Y = r) ∧ (dist O Z = r)
axiom XY_length : dist X Y = 12
axiom A_on_XY : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A = (1 - t) • X + t • Y
axiom B_on_XY : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ B = (1 - s) • X + s • Y
axiom equal_distances : dist O A = dist A Z ∧ dist A Z = dist Z B ∧ dist Z B = dist B O ∧ dist B O = 5

-- State the theorem
theorem AB_length : dist A B = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_length_l1282_128278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_arrangement_l1282_128266

-- Define the Letter type
inductive Letter
| A
| B
deriving BEq, Inhabited

-- Define a function to represent whether a letter tells the truth
def tellsTruth (l : Letter) : Bool :=
  match l with
  | Letter.A => true
  | Letter.B => false

-- Define the statements made by each letter
def statement1 (letters : List Letter) : Bool :=
  letters.count letters.head! = 1

def statement2 (letters : List Letter) : Bool :=
  letters.count Letter.A < 2

def statement3 (letters : List Letter) : Bool :=
  letters.count Letter.B = 1

-- Define the main theorem
theorem letter_arrangement :
  ∀ (letters : List Letter),
    letters.length = 3 →
    (tellsTruth (letters.get! 0) = statement1 letters) →
    (tellsTruth (letters.get! 1) = statement2 letters) →
    (tellsTruth (letters.get! 2) = statement3 letters) →
    letters = [Letter.B, Letter.A, Letter.A] :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_arrangement_l1282_128266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_exponents_l1282_128214

/-- Two terms are like terms if their variables and corresponding exponents match -/
def are_like_terms (m n : ℤ) : Prop :=
  ∃ c₁ c₂ : ℚ, ∀ x y : ℕ, c₁ * (x^2)^m * y^3 = c₂ * x^2 * y^(n+1)

theorem like_terms_exponents (m n : ℤ) :
  are_like_terms m n → m = 1 ∧ n = 2 :=
by
  intro h
  sorry -- Proof skipped

#check like_terms_exponents

end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_exponents_l1282_128214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l1282_128253

-- Define the rectangle
def rectangle_width : ℚ := 6
def rectangle_height : ℚ := 8

-- Define the triangle vertices
def point_D : ℚ × ℚ := (6, 0)
def point_E : ℚ × ℚ := (0, 3)
def point_F : ℚ × ℚ := (4, 8)

-- Function to calculate the area of a triangle given its vertices
def triangle_area (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem stating that the area of triangle DEF is 21 square units
theorem area_of_triangle_DEF : triangle_area point_D point_E point_F = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l1282_128253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l1282_128250

/-- Given a function f: ℝ → ℝ with tangent line y = 4x - 1 at x = 2, prove f(2) + f'(2) = 11 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : ∀ x, f 2 + (deriv f 2) * (x - 2) = 4 * x - 1) : 
  f 2 + deriv f 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l1282_128250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_950_l1282_128207

def closest_perfect_square (n : ℕ) : ℕ := 
  let sqrt_n := n.sqrt
  if (sqrt_n + 1)^2 - n < n - sqrt_n^2 then (sqrt_n + 1)^2 else sqrt_n^2

theorem closest_perfect_square_to_950 :
  closest_perfect_square 950 = 961 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_950_l1282_128207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_and_beth_age_problem_l1282_128283

theorem joey_and_beth_age_problem (
  joey_current_age : ℕ := 9
) (joey_past_age : ℕ := 4) (years_until_equal : ℕ := 5) :
  joey_current_age + years_until_equal = joey_current_age + (joey_current_age - joey_past_age) :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_and_beth_age_problem_l1282_128283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_midpoint_l1282_128271

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the parabola
def parabola_eq (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l
def line_l_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem intersection_and_midpoint :
  ∃ (A B : ℝ × ℝ),
    circle_eq A.1 A.2 ∧
    circle_eq B.1 B.2 ∧
    line_l_eq A.1 A.2 ∧
    line_l_eq B.1 B.2 ∧
    (A.1 + B.1) / 2 = focus.1 ∧
    (A.2 + B.2) / 2 = focus.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_midpoint_l1282_128271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l1282_128215

-- Define the concept of terminal side for angles
noncomputable def same_terminal_side (α β : ℝ) : Prop := 
  ∃ k : ℤ, α - β = 2 * k * Real.pi

-- Define the propositions
def proposition1 (θ : ℝ) : Prop := θ < Real.pi / 2 → θ > 0
def proposition2 (θ : ℝ) : Prop := Real.pi / 2 < θ ∧ θ < Real.pi → θ > Real.pi / 2
def proposition3 (α β : ℝ) : Prop := same_terminal_side α β → α = β
def proposition4 (α β : ℝ) : Prop := same_terminal_side α β → ∃ k : ℤ, α - β = 2 * k * Real.pi

theorem correct_propositions :
  (∃ θ : ℝ, ¬(proposition1 θ)) ∧
  (∃ θ : ℝ, ¬(proposition2 θ)) ∧
  (∃ α β : ℝ, ¬(proposition3 α β)) ∧
  (∀ α β : ℝ, proposition4 α β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l1282_128215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escape_car_is_black_buick_l1282_128285

structure Car where
  make : String
  color : String

inductive Witness
  | Brown
  | Jones
  | Smith

def witness_statement (w : Witness) (c : Car) : Prop :=
  match w with
  | Witness.Brown => c.make = "Buick" ∨ c.color = "blue"
  | Witness.Jones => c.make = "Chrysler" ∨ c.color = "black"
  | Witness.Smith => c.make = "Ford" ∧ c.color ≠ "blue"

def one_correct (w : Witness) (c : Car) : Prop :=
  (witness_statement w c) ∧ 
  ((c.make = "Buick" ∧ w = Witness.Brown) ∨
   (c.color = "blue" ∧ w = Witness.Brown) ∨
   (c.make = "Chrysler" ∧ w = Witness.Jones) ∨
   (c.color = "black" ∧ w = Witness.Jones) ∨
   (c.make = "Ford" ∧ w = Witness.Smith) ∨
   (c.color ≠ "blue" ∧ w = Witness.Smith))

theorem escape_car_is_black_buick :
  ∃ (c : Car), 
    (∀ w : Witness, witness_statement w c) ∧
    (∃! w : Witness, one_correct w c) ∧
    c.make = "Buick" ∧
    c.color = "black" := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escape_car_is_black_buick_l1282_128285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outbound_speed_is_60_l1282_128227

/-- Represents a round trip with different speeds for outbound and return journeys -/
structure RoundTrip where
  outbound_time : ℝ
  return_time : ℝ
  speed_difference : ℝ

/-- Calculates the outbound speed given a RoundTrip -/
noncomputable def calculate_outbound_speed (trip : RoundTrip) : ℝ :=
  (trip.outbound_time * trip.speed_difference) / (trip.return_time - trip.outbound_time)

theorem outbound_speed_is_60 (trip : RoundTrip) 
  (h1 : trip.outbound_time = 6)
  (h2 : trip.return_time = 5)
  (h3 : trip.speed_difference = 12) :
  calculate_outbound_speed trip = 60 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_outbound_speed ⟨6, 5, 12⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outbound_speed_is_60_l1282_128227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_from_complex_distance_l1282_128216

theorem cos_difference_from_complex_distance (α β : ℝ) :
  let z₁ : ℂ := Complex.exp (α * Complex.I)
  let z₂ : ℂ := Complex.exp (β * Complex.I)
  (Complex.abs (z₁ - z₂) = (2 / 5) * Real.sqrt 5) →
  Real.cos (α - β) = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_from_complex_distance_l1282_128216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_return_proof_l1282_128230

/-- Calculates the percentage return on an investment -/
noncomputable def percentage_return (investment : ℝ) (earnings : ℝ) : ℝ :=
  (earnings / investment) * 100

theorem investment_return_proof (investment earnings : ℝ) 
  (h1 : investment = 1620)
  (h2 : earnings = 135) :
  ∃ (ε : ℝ), ε > 0 ∧ |percentage_return investment earnings - 8.33| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_return_proof_l1282_128230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l1282_128287

noncomputable def dataset : List ℝ := [8, 5, 2, 5, 6, 4]

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

theorem variance_of_dataset :
  variance dataset = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_dataset_l1282_128287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_angles_theorem_l1282_128262

theorem adjacent_angles_theorem (θ₁ θ₂ : ℝ) : 
  θ₁ + θ₂ = 180 →  -- Adjacent angles sum to 180°
  θ₂ = 3 * θ₁ →    -- One angle is three times the other
  (θ₁ = 45 ∧ θ₂ = 135) := by
  intro h1 h2
  -- Proof steps would go here
  sorry

#check adjacent_angles_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_angles_theorem_l1282_128262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_perpendicular_l1282_128273

/-- 
Theorem: For any non-zero real number a, if x₀ is a point of intersection 
between y = cos x and y = a tan x, then the product of the slopes of the 
tangent lines to these functions at x₀ is equal to -1.
-/
theorem tangents_perpendicular (a : ℝ) (x₀ : ℝ) (h1 : a ≠ 0) 
  (h2 : Real.cos x₀ = a * Real.tan x₀) : 
  (-Real.sin x₀) * (a / (Real.cos x₀)^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_perpendicular_l1282_128273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_N_with_condition_sum_of_digits_of_greatest_N_l1282_128221

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the condition for N
def condition (N : ℕ) : Prop :=
  floor (Real.sqrt (floor (Real.sqrt (floor (Real.sqrt (N : ℝ)))))) = 4

-- State the theorem
theorem greatest_N_with_condition :
  ∃ N : ℕ, condition N ∧ ∀ M : ℕ, M > N → ¬condition M :=
sorry

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ :=
  let rec aux (m : ℕ) (acc : ℕ) : ℕ :=
    if m = 0 then acc
    else aux (m / 10) (acc + m % 10)
  aux n 0

-- State the final theorem
theorem sum_of_digits_of_greatest_N :
  ∃ N : ℕ, condition N ∧ ∀ M : ℕ, M > N → ¬condition M ∧ sumOfDigits N = 24 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_N_with_condition_sum_of_digits_of_greatest_N_l1282_128221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_coefficients_l1282_128242

/-- The sum of the squares of the coefficients of 5(x^4 + 2x^3 + 3x^2 + 2) is 450 -/
theorem sum_of_squares_of_coefficients : 
  let p : Polynomial ℤ := 5 * (X^4 + 2*X^3 + 3*X^2 + 2)
  (Finset.range 5).sum (λ i => (p.coeff i)^2) = 450 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_coefficients_l1282_128242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_rearrangement_exists_l1282_128220

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A configuration of five points in a plane -/
structure Configuration where
  points : Fin 5 → Point

/-- A line in the plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point is symmetric to another point with respect to a line -/
def isSymmetric (p q : Point) (l : Line) : Prop :=
  sorry

/-- Checks if a configuration is symmetric with respect to a line -/
def isSymmetricConfig (c : Configuration) (l : Line) : Prop :=
  ∃ (perm : Fin 5 → Fin 5), ∀ i, isSymmetric (c.points i) (c.points (perm i)) l

/-- The main theorem to prove -/
theorem symmetric_rearrangement_exists (c : Configuration) :
  ∃ (c' : Configuration) (l : Line),
    (∀ i j, distance (c'.points i) (c'.points j) = distance (c.points i) (c.points j) ∨
            c'.points i = c.points i ∨ c'.points j = c.points j) ∧
    isSymmetricConfig c' l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_rearrangement_exists_l1282_128220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_girl_height_l1282_128277

/-- A group of girls with their heights and weights -/
structure GirlGroup where
  count : ℕ
  avgHeight : ℝ
  heights : Fin count → ℝ
  weights : Fin count → ℝ

/-- The correlation coefficient between two lists of real numbers -/
noncomputable def correlationCoefficient (xs ys : List ℝ) : ℝ := sorry

theorem new_girl_height
  (originalGroup : GirlGroup)
  (replacedGirl : { height : ℝ // height = 160 })
  (replacedGirlWeight : ℝ)
  (heightIncrease : ℝ)
  (newGroup : GirlGroup)
  (h1 : originalGroup.count = 25)
  (h2 : newGroup.count = 25)
  (h3 : replacedGirlWeight = 55)
  (h4 : heightIncrease = 2)
  (h5 : newGroup.avgHeight = originalGroup.avgHeight + heightIncrease)
  (h6 : correlationCoefficient (List.ofFn newGroup.heights) (List.ofFn newGroup.weights) = 0.8)
  : ∃ (newGirlHeight : ℝ), newGirlHeight = 210 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_girl_height_l1282_128277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mitotic_chromosome_count_l1282_128270

/-- Represents the number of chromosomes in a cell -/
structure ChromosomeCount where
  count : ℕ

/-- The number of chromosomes at the late stage of the second meiotic division -/
def meioticChromosomes : ChromosomeCount := ⟨24⟩

/-- The number of chromosomes at the late stage of mitosis -/
def mitoticChromosomes : ChromosomeCount := ⟨meioticChromosomes.count * 2⟩

/-- Theorem stating that the number of chromosomes at the late stage of mitosis is 48 -/
theorem mitotic_chromosome_count : mitoticChromosomes.count = 48 := by
  rfl

#eval mitoticChromosomes.count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mitotic_chromosome_count_l1282_128270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_constant_term_value_l1282_128233

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := 
  Nat.choose n k

-- Define the expression (marked as noncomputable due to Real.sqrt)
noncomputable def expression (x : ℝ) : ℝ := 
  (Real.sqrt x + 7 / x) ^ 11

-- Theorem statement
theorem constant_term_of_expansion :
  ∃ (c : ℝ), c = 792330 ∧ 
  ∀ (x : ℝ), x > 0 → 
    expression x = c + x * (Real.sqrt x * Real.sqrt x * sorry + sorry / x + sorry) :=
by sorry

-- Additional theorem to state the exact value of the constant term
theorem constant_term_value : 
  binomial 11 4 * 7^4 = 792330 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_constant_term_value_l1282_128233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_volume_l1282_128219

/-- Regular triangular pyramid with inscribed cylinder -/
structure PyramidWithCylinder where
  /-- Side edge length of the pyramid -/
  a : ℝ
  /-- Angle between side edge and base plane -/
  α : ℝ
  /-- Assumption: a > 0 -/
  a_pos : a > 0
  /-- Assumption: 0 < α < π/2 -/
  α_range : 0 < α ∧ α < π/2

/-- Volume of the inscribed cylinder -/
noncomputable def cylinderVolume (p : PyramidWithCylinder) : ℝ :=
  (Real.pi * p.a^3 * Real.sqrt 2 * (Real.sin (2 * p.α))^3) / (128 * (Real.sin (Real.pi/4 + p.α))^3)

/-- Theorem: The volume of the inscribed cylinder is as calculated -/
theorem inscribed_cylinder_volume (p : PyramidWithCylinder) :
  ∃ V, V = cylinderVolume p ∧ V > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_volume_l1282_128219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l1282_128206

theorem complex_absolute_value : 
  Complex.abs (1/3 - Complex.I * (5/7 : ℝ)) = Real.sqrt 274 / 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l1282_128206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_l1282_128226

/-- The projection matrix onto a vector in 3D space -/
noncomputable def projection_matrix (v : Fin 3 → ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  let norm_sq := v 0 ^ 2 + v 1 ^ 2 + v 2 ^ 2
  Matrix.of (λ i j => (v i * v j) / norm_sq)

/-- The vector to project onto -/
def u : Fin 3 → ℝ := ![3, 2, -6]

/-- The projection matrix Q -/
noncomputable def Q : Matrix (Fin 3) (Fin 3) ℝ := projection_matrix u

theorem det_projection_matrix_zero :
  Matrix.det Q = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_l1282_128226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_start_days_l1282_128276

-- Define the days of the week
inductive Day : Type
| sunday : Day
| monday : Day
| tuesday : Day
| wednesday : Day
| thursday : Day
| friday : Day
| saturday : Day

-- Define a function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.sunday => Day.monday
  | Day.monday => Day.tuesday
  | Day.tuesday => Day.wednesday
  | Day.wednesday => Day.thursday
  | Day.thursday => Day.friday
  | Day.friday => Day.saturday
  | Day.saturday => Day.sunday

-- Define a function to advance a day by n days
def advanceDay (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Define the closure days
def isClosed (d : Day) : Bool :=
  match d with
  | Day.wednesday => true
  | Day.saturday => true
  | _ => false

-- Define a function to check if a starting day is valid for 8 weeks
def isValidStartDay (startDay : Day) : Bool :=
  let redeemDays := List.range 8 |>.map (fun i => advanceDay startDay (i * 7))
  redeemDays.all (fun d => !isClosed d)

-- Theorem statement
theorem valid_start_days :
  ∀ (d : Day), isValidStartDay d ↔ (d = Day.sunday ∨ d = Day.monday ∨ d = Day.tuesday ∨ d = Day.thursday) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_start_days_l1282_128276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sight_time_is_48_seconds_l1282_128291

-- Define the constants
noncomputable def pathDistance : ℝ := 300
noncomputable def kennySpeed : ℝ := 4
noncomputable def jennySpeed : ℝ := 2
noncomputable def buildingDiameter : ℝ := 150
noncomputable def initialDistance : ℝ := 300

-- Define the positions of Jenny and Kenny as functions of time
noncomputable def jennyPosition (t : ℝ) : ℝ × ℝ := (-75 + jennySpeed * t, 150)
noncomputable def kennyPosition (t : ℝ) : ℝ × ℝ := (-75 + kennySpeed * t, -150)

-- Define the equation of the line connecting Jenny and Kenny
noncomputable def lineSightEquation (t : ℝ) (x : ℝ) : ℝ := 
  -150 / t * x + 300 - 11250 / t

-- Define the equation of the circular building
def buildingEquation (x y : ℝ) : Prop := 
  x^2 + y^2 = (buildingDiameter / 2)^2

-- Theorem statement
theorem sight_time_is_48_seconds : 
  ∃ t : ℝ, t = 48 ∧ 
  (∀ x y : ℝ, buildingEquation x y → 
    (y = lineSightEquation t x → 
      (x * t = 150 * y ∧ 
       x = 7500 / Real.sqrt (150^2 + t^2)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sight_time_is_48_seconds_l1282_128291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l1282_128248

/-- Definition of an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- Predicate to check if a given angle belongs to the triangle -/
def is_angle (t : EquilateralTriangle) (θ : ℝ) : Prop := sorry

/-- Predicate to check if a given length is a side of the triangle -/
def is_side (t : EquilateralTriangle) (s : ℝ) : Prop := sorry

/-- Function to calculate the area of the triangle -/
noncomputable def area (t : EquilateralTriangle) : ℝ := sorry

/-- Function to calculate the radius of the circumscribed circle -/
noncomputable def circumradius (t : EquilateralTriangle) : ℝ := sorry

/-- Properties of equilateral triangles -/
theorem equilateral_triangle_properties (t : EquilateralTriangle) :
  -- All angles are equal
  (∀ θ₁ θ₂ : ℝ, is_angle t θ₁ → is_angle t θ₂ → θ₁ = θ₂) ∧
  -- All sides have the same length
  (∀ s : ℝ, is_side t s → s = t.side) ∧
  -- The area is proportional to the square of the side length
  (∃ k : ℝ, area t = k * t.side^2) ∧
  -- The radius of the circumscribed circle is not constant for all equilateral triangles
  (∃ t₁ t₂ : EquilateralTriangle, t₁.side ≠ t₂.side → 
    circumradius t₁ ≠ circumradius t₂) ∧
  -- All angles are 60 degrees
  (∀ θ : ℝ, is_angle t θ → θ = 60) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l1282_128248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bees_in_hive_l1282_128247

theorem bees_in_hive (initial_bees final_bees incoming_bees : ℕ) :
  initial_bees + incoming_bees = final_bees →
  incoming_bees = 10 →
  final_bees = 26 →
  initial_bees = 16 := by
  intros h1 h2 h3
  rw [h2, h3] at h1
  linarith

#eval 16 + 10 -- Should output 26

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bees_in_hive_l1282_128247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earth_sun_distance_calculation_l1282_128229

/-- The speed of light in meters per second -/
def speed_of_light : ℝ := 3e8

/-- The time it takes for sunlight to reach Earth in seconds -/
def time_to_earth : ℝ := 5e2

/-- The distance between the Earth and the Sun in meters -/
def earth_sun_distance : ℝ := speed_of_light * time_to_earth

theorem earth_sun_distance_calculation :
  ∃ ε > 0, |earth_sun_distance - 1.5e11| < ε := by
  -- The proof goes here
  sorry

#eval earth_sun_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earth_sun_distance_calculation_l1282_128229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_food_duration_l1282_128298

/-- Represents the amount of food each dog eats per meal in grams -/
def dog_food_per_meal : Fin 4 → ℕ
| 0 => 250
| 1 => 350
| 2 => 450
| 3 => 550

/-- The number of meals per day -/
def meals_per_day : ℕ := 2

/-- The weight of each sack of dog food in grams -/
def sack_weight : ℕ := 50 * 1000

/-- The number of sacks of dog food -/
def num_sacks : ℕ := 2

/-- Calculates the total amount of food consumed by all dogs in one day -/
def total_food_per_day : ℕ :=
  meals_per_day * (Finset.sum Finset.univ (λ i => dog_food_per_meal i))

/-- Calculates the total amount of dog food available -/
def total_food_available : ℕ :=
  num_sacks * sack_weight

/-- The theorem stating how many full days the dog food will last -/
theorem dog_food_duration : 
  (total_food_available / total_food_per_day : ℕ) = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_food_duration_l1282_128298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrationality_of_sqrt_2_and_rationality_of_others_l1282_128212

theorem irrationality_of_sqrt_2_and_rationality_of_others : 
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 2 = (p : ℚ) / q) ∧ 
  (∃ (p q : ℤ), q ≠ 0 ∧ -5 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (2 : ℚ) / 3 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (314 : ℚ) / 100 = (p : ℚ) / q) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrationality_of_sqrt_2_and_rationality_of_others_l1282_128212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1282_128279

/-- The eccentricity of a hyperbola is √2, given a parabola and the intersection of the hyperbola's asymptotes -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), y^2 = 12*x ∧ x^2/a^2 - y^2/b^2 = 1) →
  (∃ (x y : ℝ), y = x ∧ x = 12) →
  Real.sqrt 2 = (Real.sqrt (a^2 + b^2)) / a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1282_128279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_becomes_white_l1282_128213

-- Define the grid as a function from coordinates to colors
def Grid := ℕ → ℕ → Bool

-- Define the update rule for a single cell
def updateCell (g : Grid) (x y : ℕ) : Bool :=
  let count := (if g x y then 1 else 0) +
                (if g x (y + 1) then 1 else 0) +
                (if g (x + 1) y then 1 else 0)
  count ≥ 2

-- Define the update rule for the entire grid
def updateGrid (g : Grid) : Grid :=
  λ x y ↦ updateCell g x y

-- Define a predicate to check if a grid is all white
def allWhite (g : Grid) : Prop :=
  ∀ x y, g x y = false

-- Define the number of grey cells in a grid
def greyCount (g : Grid) : ℕ :=
  sorry

-- The main theorem
theorem grid_becomes_white (g : Grid) (n : ℕ) (h : greyCount g = n) :
  ∃ k : ℕ, k ≤ n ∧ allWhite (Nat.iterate updateGrid k g) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_becomes_white_l1282_128213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dvaneft_percentage_bounds_l1282_128280

/-- Represents the share package of an oil extraction company -/
structure SharePackage where
  price : ℚ
  quantity : ℚ

/-- Represents the bidding lot consisting of three share packages -/
structure BiddingLot where
  razneft : SharePackage
  dvaneft : SharePackage
  trineft : SharePackage

/-- The conditions of the bidding lot as described in the problem -/
def valid_bidding_lot (lot : BiddingLot) : Prop :=
  -- The number of shares in Razneft and Dvaneft packages combined equals the number in Trineft package
  lot.razneft.quantity + lot.dvaneft.quantity = lot.trineft.quantity
  -- The Dvaneft package is four times cheaper than the Razneft package
  ∧ lot.dvaneft.price * lot.dvaneft.quantity = lot.razneft.price * lot.razneft.quantity / 4
  -- The total cost of Razneft and Dvaneft packages equals the cost of Trineft package
  ∧ lot.razneft.price * lot.razneft.quantity + lot.dvaneft.price * lot.dvaneft.quantity
    = lot.trineft.price * lot.trineft.quantity
  -- The price difference between one Razneft share and one Dvaneft share is between 16,000 and 20,000
  ∧ 16000 ≤ lot.razneft.price - lot.dvaneft.price
  ∧ lot.razneft.price - lot.dvaneft.price ≤ 20000
  -- The price of one Trineft share is between 42,000 and 60,000
  ∧ 42000 ≤ lot.trineft.price
  ∧ lot.trineft.price ≤ 60000

/-- The percentage of Dvaneft shares in the total lot -/
def dvaneft_percentage (lot : BiddingLot) : ℚ :=
  lot.dvaneft.quantity / (lot.razneft.quantity + lot.dvaneft.quantity + lot.trineft.quantity) * 100

/-- The theorem stating that the percentage of Dvaneft shares is bounded between 12.5% and 15% -/
theorem dvaneft_percentage_bounds (lot : BiddingLot) :
  valid_bidding_lot lot → (125 : ℚ)/10 ≤ dvaneft_percentage lot ∧ dvaneft_percentage lot ≤ 15 := by
  sorry -- The proof is omitted as per the instructions


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dvaneft_percentage_bounds_l1282_128280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l1282_128224

noncomputable def f (α : ℝ) : ℝ := (Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.tan (Real.pi + α)) / 
                     (Real.tan (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : ℝ) : f α = -Real.cos α := by sorry

theorem f_specific_value : f (-31 * Real.pi / 3) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l1282_128224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1282_128297

theorem equation_solution :
  ∃ x : ℝ, (27 : ℝ) ^ (3 * x - 7) = (1 / 3 : ℝ) ^ (2 * x + 4) ↔ x = 17 / 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1282_128297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l1282_128236

/-- The distance of the race in yards -/
def d : ℝ := sorry

/-- The speed of runner A -/
def a : ℝ := sorry

/-- The speed of runner B -/
def b : ℝ := sorry

/-- The speed of runner C -/
def c : ℝ := sorry

/-- A can beat B by 25 yards -/
axiom A_beats_B : d / a = (d - 25) / b

/-- B can beat C by 15 yards -/
axiom B_beats_C : d / b = (d - 15) / c

/-- A can beat C by 37 yards -/
axiom A_beats_C : d / a = (d - 37) / c

/-- The race distance is 125 yards -/
theorem race_distance : d = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l1282_128236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_six_l1282_128290

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  abs (l.a * p.1 + l.b * p.2 + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Calculate the chord length given a circle and a line -/
noncomputable def chordLength (c : Circle) (l : Line) : ℝ :=
  2 * Real.sqrt (c.radius^2 - (distancePointToLine c.center l)^2)

theorem chord_length_is_six :
  let c : Circle := { center := (2, 1), radius := 5 }
  let l : Line := { a := 3, b := 4, c := 10 }
  chordLength c l = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_six_l1282_128290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_extrema_l1282_128222

noncomputable def a (n : ℝ) : ℝ := (n - Real.sqrt 80) / (n - Real.sqrt 79)

theorem sequence_extrema :
  (∀ k ∈ Finset.range 50, a 9 ≤ a (k + 1)) ∧
  (∀ k ∈ Finset.range 50, a (k + 1) ≤ a 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_extrema_l1282_128222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_missed_one_day_l1282_128288

/-- Calculates the number of days Julie was not able to work based on her work schedule and actual monthly salary. -/
def days_not_worked (hourly_rate : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) (actual_monthly_salary : ℚ) : ℕ :=
  let daily_earnings := hourly_rate * hours_per_day
  let working_days_per_month := days_per_week * 4
  let full_monthly_earnings := daily_earnings * working_days_per_month
  let earnings_difference := full_monthly_earnings - actual_monthly_salary
  Int.natAbs ((earnings_difference / daily_earnings).num)

/-- Theorem stating that Julie missed 1 day of work given her work schedule and actual monthly salary. -/
theorem julie_missed_one_day :
  days_not_worked 5 8 6 920 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_missed_one_day_l1282_128288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1282_128265

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def proposition_q (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

def exclusive_or (p q : Prop) : Prop :=
  (p ∨ q) ∧ ¬(p ∧ q)

theorem range_of_a :
  ∃ S : Set ℝ, S = Set.Iic (-2) ∪ Set.Ico 1 2 ∧
  ∀ a : ℝ, a ∈ S ↔ exclusive_or (proposition_p a) (proposition_q a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1282_128265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_rational_to_22_7_l1282_128249

noncomputable def closest_rational (target : ℚ) (max_denominator : ℕ) : ℚ :=
  sorry

theorem closest_rational_to_22_7 :
  let target : ℚ := 22 / 7
  let max_denominator : ℕ := 99
  let closest : ℚ := closest_rational target max_denominator
  closest = 311 / 99 ∧ 
  closest ≠ target ∧
  (∀ p q : ℕ, q < max_denominator → (p : ℚ) / q ≠ target → |(p : ℚ) / q - target| ≥ |closest - target|) ∧
  (closest.num - 3 * closest.den = 14) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_rational_to_22_7_l1282_128249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l1282_128204

/-- Given two triangles, if the cosines of the interior angles of one triangle
    are equal to the sines of the interior angles of the other triangle,
    then the first triangle is acute and the second is obtuse. -/
theorem triangle_angle_relation (A₁ B₁ C₁ A₂ B₂ C₂ : Real) :
  (0 < A₁) ∧ (A₁ < π) ∧ (0 < B₁) ∧ (B₁ < π) ∧ (0 < C₁) ∧ (C₁ < π) ∧
  (0 < A₂) ∧ (A₂ < π) ∧ (0 < B₂) ∧ (B₂ < π) ∧ (0 < C₂) ∧ (C₂ < π) ∧
  (A₁ + B₁ + C₁ = π) ∧ (A₂ + B₂ + C₂ = π) ∧
  (Real.cos A₁ = Real.sin A₂) ∧ (Real.cos B₁ = Real.sin B₂) ∧ (Real.cos C₁ = Real.sin C₂) →
  (A₁ < π/2 ∧ B₁ < π/2 ∧ C₁ < π/2) ∧
  (A₂ > π/2 ∨ B₂ > π/2 ∨ C₂ > π/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l1282_128204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_k_value_l1282_128223

-- Define the line equation
def line_equation (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ - Real.pi/4) = 4

-- Define the circle equation
def circle_equation (ρ θ k : ℝ) : Prop :=
  ρ = 2 * k * Real.cos (θ + Real.pi/4)

-- Define the minimum distance condition
def min_distance_condition (k : ℝ) : Prop :=
  |k + 4| - |k| = 2

-- Main theorem
theorem circle_center_and_k_value
  (k : ℝ)
  (h_k_nonzero : k ≠ 0)
  (h_min_distance : min_distance_condition k) :
  (∃ (x y : ℝ), x = -Real.sqrt 2 / 2 ∧ y = Real.sqrt 2 / 2) ∧ k = -1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_k_value_l1282_128223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_divisors_eq_11_l1282_128289

/-- The number of integer divisors of 60 that are less than 60 -/
def num_divisors : ℕ :=
  (Finset.filter (· < 60) (Nat.divisors 60)).card

/-- The theorem stating that the number of such divisors is 11 -/
theorem num_divisors_eq_11 : num_divisors = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_divisors_eq_11_l1282_128289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_functions_l1282_128238

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → f x * f y = y^h * f (x/2) + x^k * f (y/2)

/-- The main theorem characterizing functions satisfying the functional equation -/
theorem characterize_functions
  (f : ℝ → ℝ) (h k : ℝ) (hf : SatisfiesFunctionalEquation f h k) :
  (h ≠ k ∧ ∀ x : ℝ, x > 0 → f x = 0) ∨
  (h = k ∧ ((∀ x : ℝ, x > 0 → f x = 0) ∨ (∀ x : ℝ, x > 0 → f x = 2 * (x/2)^h))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_functions_l1282_128238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invisible_dots_count_l1282_128210

/- Define the number of dice -/
def num_dice : ℕ := 4

/- Define the number of faces on each die -/
def faces_per_die : ℕ := 6

/- Define the number of visible faces -/
def visible_faces : ℕ := 8

/- Define the list of visible numbers -/
def visible_numbers : List ℕ := [1, 1, 2, 3, 3, 4, 5, 6]

/- Function to calculate the sum of numbers on one die -/
def sum_of_one_die : ℕ := (List.range faces_per_die).map (· + 1) |>.sum

/- Theorem: The total number of dots not visible is 59 -/
theorem invisible_dots_count :
  num_dice * sum_of_one_die - visible_numbers.sum = 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invisible_dots_count_l1282_128210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_quadrilateral_eq_four_times_triangle_l1282_128218

-- Define the points
variable (A B C D M N O : ℝ × ℝ)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A

-- Define M as midpoint of AC
def is_midpoint_AC (M A C : ℝ × ℝ) : Prop :=
  M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define N as midpoint of BD
def is_midpoint_BD (N B D : ℝ × ℝ) : Prop :=
  N = ((B.1 + D.1) / 2, (B.2 + D.2) / 2)

-- Define O as intersection of extensions of BA and CD
def is_intersection_BACD (O A B C D : ℝ × ℝ) : Prop :=
  ∃ t s : ℝ, O = (A.1 + t * (A.1 - B.1), A.2 + t * (A.2 - B.2)) ∧
             O = (C.1 + s * (C.1 - D.1), C.2 + s * (C.2 - D.2))

-- Define area of a triangle
noncomputable def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) / 2

-- Define area of a quadrilateral
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  area_triangle A B C + area_triangle A C D

-- State the theorem
theorem area_quadrilateral_eq_four_times_triangle
  (h1 : is_quadrilateral A B C D)
  (h2 : is_midpoint_AC M A C)
  (h3 : is_midpoint_BD N B D)
  (h4 : is_intersection_BACD O A B C D) :
  area_quadrilateral A B C D = 4 * area_triangle O M N :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_quadrilateral_eq_four_times_triangle_l1282_128218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_midpoint_to_origin_l1282_128299

/-- Given a line L and two points P₁ and P₂ on L, prove that the distance from the midpoint P of P₁P₂ to (1, -2) is |t₁ + t₂|/2 -/
theorem distance_midpoint_to_origin (t₁ t₂ : ℝ) : 
  let L := fun t : ℝ => (1 + t/2, -2 + Real.sqrt 3 * t/2)
  let P₁ := L t₁
  let P₂ := L t₂
  let P := ((P₁.1 + P₂.1)/2, (P₁.2 + P₂.2)/2)
  ‖P - (1, -2)‖ = |t₁ + t₂|/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_midpoint_to_origin_l1282_128299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lance_hourly_wage_l1282_128292

/-- Lance's weekly work schedule and earnings --/
structure WorkSchedule where
  hours_per_week : ℚ
  workdays_per_week : ℚ
  earnings_per_day : ℚ

/-- Calculate hourly wage given a work schedule --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  (schedule.earnings_per_day * schedule.workdays_per_week) / schedule.hours_per_week

/-- Theorem: Lance's hourly wage is $9 --/
theorem lance_hourly_wage :
  let schedule : WorkSchedule := {
    hours_per_week := 35,
    workdays_per_week := 5,
    earnings_per_day := 63
  }
  hourly_wage schedule = 9 := by
  -- The proof goes here
  sorry

#eval hourly_wage { hours_per_week := 35, workdays_per_week := 5, earnings_per_day := 63 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lance_hourly_wage_l1282_128292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l1282_128269

theorem negation_of_cosine_inequality :
  (¬ (∀ x : ℝ, Real.cos x ≤ 1)) ↔ (∃ x : ℝ, Real.cos x > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l1282_128269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_asymptote_sum_l1282_128241

/-- Represents a rational function -/
structure RationalFunction where
  numerator : Polynomial ℝ
  denominator : Polynomial ℝ

/-- Counts the number of holes in the graph of a rational function -/
noncomputable def count_holes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of vertical asymptotes in the graph of a rational function -/
noncomputable def count_vertical_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of horizontal asymptotes in the graph of a rational function -/
noncomputable def count_horizontal_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of oblique asymptotes in the graph of a rational function -/
noncomputable def count_oblique_asymptotes (f : RationalFunction) : ℕ := sorry

theorem rational_function_asymptote_sum :
  let f : RationalFunction := {
    numerator := Polynomial.X^2 - Polynomial.X - 6,
    denominator := Polynomial.X^3 - 2*Polynomial.X^2 - Polynomial.X + 2
  }
  let a := count_holes f
  let b := count_vertical_asymptotes f
  let c := count_horizontal_asymptotes f
  let d := count_oblique_asymptotes f
  a + 2*b + 3*c + 4*d = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_asymptote_sum_l1282_128241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divisible_by_9999_length_of_MN_l1282_128260

theorem smallest_k_divisible_by_9999 : 
  (∃ k : ℕ, k > 0 ∧ (∀ m : ℕ, m > 0 → m < k → ¬((10^m - 1) % (9999 * 9) = 0)) ∧ 
   ((10^k - 1) % (9999 * 9) = 0)) → 
  (∃ k : ℕ, k = 180 ∧ k > 0 ∧ (∀ m : ℕ, m > 0 → m < k → ¬((10^m - 1) % (9999 * 9) = 0)) ∧ 
   ((10^k - 1) % (9999 * 9) = 0)) :=
by sorry

theorem length_of_MN (T : ℝ) (h1 : AQ = Real.sqrt T) (h2 : BQ = 7) (h3 : AB = 8) :
  MN = 128 / 21 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divisible_by_9999_length_of_MN_l1282_128260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1282_128225

open Real

theorem trigonometric_equation_solution :
  ∀ x : ℝ, (cos x)^3 + (1/2) * sin (2*x) - cos x * (sin x)^3 + 4 * sin x + 4 = 0 ↔
  ∃ k : ℤ, x = (π/2) * (4*k - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1282_128225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_m_range_l1282_128257

/-- A function f(x) that depends on a parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (4*m - 1) * x^2 + (15*m^2 - 2*m - 7) * x + 2

/-- The derivative of f(x) with respect to x -/
noncomputable def f' (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*(4*m - 1)*x + (15*m^2 - 2*m - 7)

theorem increasing_f_implies_m_range :
  ∀ m : ℝ, (∀ x : ℝ, (f' m x ≥ 0)) → (2 ≤ m ∧ m ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_m_range_l1282_128257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_j_is_self_inverse_l1282_128239

/-- A function that is symmetric about the line y = x-1 --/
def h : ℝ → ℝ := sorry

/-- The property of h being symmetric about y = x-1 --/
axiom h_symmetry : ∀ x y : ℝ, h x = y ↔ h (y + 1) = x + 1

/-- The function j defined as a shift of h --/
def j (x : ℝ) : ℝ := h (x - 1)

/-- Theorem stating that j is symmetric about y = x --/
theorem j_is_self_inverse : ∀ x y : ℝ, j x = y ↔ j y = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_j_is_self_inverse_l1282_128239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_hundredth_l1282_128252

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The sum of 34.2791 and 15.73684 rounded to the nearest hundredth equals 50.02 -/
theorem sum_and_round_to_hundredth : roundToHundredth (34.2791 + 15.73684) = 50.02 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_hundredth_l1282_128252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_max_area_OAB_l1282_128296

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 16

-- Define point F
def point_F : ℝ × ℝ := (-1, 0)

-- Define the ellipse E (trajectory of P)
def ellipse_E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define a line passing through F
def line_through_F (m : ℝ) (x y : ℝ) : Prop := x = m * y - 1

-- Theorem for the trajectory of point P
theorem trajectory_of_P :
  ∀ x y : ℝ, (∃ M : ℝ × ℝ, circle_C M.1 M.2 ∧ 
    (∃ P : ℝ × ℝ, P.1 = x ∧ P.2 = y ∧ 
      (P.1 - M.1)^2 + (P.2 - M.2)^2 = (P.1 - point_F.1)^2 + (P.2 - point_F.2)^2)) 
  → ellipse_E x y :=
sorry

-- Theorem for the maximum area of triangle OAB
theorem max_area_OAB :
  (∃ m : ℝ, ∃ A B : ℝ × ℝ, 
    line_through_F m A.1 A.2 ∧ 
    line_through_F m B.1 B.2 ∧ 
    ellipse_E A.1 A.2 ∧ 
    ellipse_E B.1 B.2 ∧ 
    A ≠ B) →
  (∀ m : ℝ, ∀ A B : ℝ × ℝ, 
    line_through_F m A.1 A.2 → 
    line_through_F m B.1 B.2 → 
    ellipse_E A.1 A.2 → 
    ellipse_E B.1 B.2 → 
    A ≠ B →
    abs ((A.1 * B.2 - A.2 * B.1) / 2) ≤ 3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_max_area_OAB_l1282_128296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l1282_128261

-- Define the function representing the equation
def f (a b : ℝ) : Prop := 2 * a^2 - 5 * Real.log a - b = 0

-- Define the distance function
noncomputable def distance (a b c : ℝ) : ℝ := Real.sqrt ((a - c)^2 + (b + c)^2)

-- State the theorem
theorem min_distance_theorem (a b c : ℝ) (h : f a b) :
  ∃ (min_val : ℝ), min_val = (3 * Real.sqrt 2) / 2 ∧
  ∀ (x y : ℝ), f x y → distance x y c ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l1282_128261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_700_to_900_sum_18_l1282_128245

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Count of integers in a range with a specific sum of digits -/
def count_integers_with_sum_of_digits (lower upper sum : ℕ) : ℕ :=
  (List.range (upper - lower + 1)).map (λ i => lower + i)
    |>.filter (λ n => sum_of_digits n = sum)
    |>.length

theorem count_integers_700_to_900_sum_18 :
  count_integers_with_sum_of_digits 700 900 18 = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_700_to_900_sum_18_l1282_128245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeff_storage_capacity_is_320_l1282_128259

/-- Calculates the number of songs Jeff can store on his phone given the specified conditions. -/
def jeff_storage_capacity : ℕ :=
  let total_storage : ℕ := 32 * 1000  -- 32 GB in MB
  let used_storage : ℕ := 7 * 1000    -- 7 GB in MB
  let app_storage : ℕ := 5 * 450 + 5 * 300 + 5 * 150
  let photo_storage : ℕ := 300 * 4 + 50 * 8
  let video_storage : ℕ := 15 * 400 + 30 * 200
  let pdf_storage : ℕ := 25 * 20
  let new_data_storage : ℕ := app_storage + photo_storage + video_storage + pdf_storage
  let total_used_storage : ℕ := used_storage + new_data_storage
  let remaining_storage : ℕ := total_storage - total_used_storage
  let song_size : ℕ := 20
  remaining_storage / song_size

theorem jeff_storage_capacity_is_320 : jeff_storage_capacity = 320 := by
  rfl

#eval jeff_storage_capacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeff_storage_capacity_is_320_l1282_128259
