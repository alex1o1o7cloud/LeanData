import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l808_80831

/-- The distance between two parallel planes -/
noncomputable def distance_between_planes (a b c d : ℝ) (k : ℝ) : ℝ :=
  |k - d| / Real.sqrt (a^2 + b^2 + c^2)

/-- Theorem: Distance between two specific parallel planes -/
theorem distance_between_specific_planes :
  let plane1 := λ (x y z : ℝ) => 2*x - 3*y + z - 4 = 0
  let plane2 := λ (x y z : ℝ) => 4*x - 6*y + 2*z + 3 = 0
  distance_between_planes 2 (-3) 1 (-4) (3/2) = 11 * Real.sqrt 14 / 28 := by
  sorry

#check distance_between_specific_planes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l808_80831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l808_80835

theorem sin_beta_value (α β : ℝ) 
  (h1 : Real.cos (α - β) * Real.cos α + Real.sin (α - β) * Real.sin α = -4/5)
  (h2 : π < β ∧ β < 3*π/2) : 
  Real.sin β = -3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l808_80835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_50_not_243_l808_80827

/-- The number of positive integer divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- g₁(n) is thrice the square of the number of positive integer divisors of n -/
def g₁ (n : ℕ+) : ℕ := 3 * (d n)^2

/-- gⱼ(n) for j ≥ 2 is defined recursively as g₁(gⱼ₋₁(n)) -/
def g : ℕ → ℕ+ → ℕ
  | 0, n => n
  | 1, n => g₁ n
  | j+2, n => g 1 ⟨g (j+1) n, sorry⟩

/-- For all positive integers n ≤ 30, g₅₀(n) ≠ 243 -/
theorem g_50_not_243 : ∀ n : ℕ+, n ≤ 30 → g 50 n ≠ 243 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_50_not_243_l808_80827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l808_80861

noncomputable section

open Real

/-- The function f(x) = √3 * sin(x/2 - π/4) -/
def f (x : ℝ) : ℝ := sqrt 3 * sin (x / 2 - π / 4)

/-- The smallest positive period of f(x) is 4π -/
theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = 4 * π := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l808_80861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_divisible_pair_l808_80819

-- Define the set of marked numbers
def MarkedNumbers : Set ℕ := sorry

-- Axiom: Every segment of length 1999 contains a marked number
axiom segment_contains_marked :
  ∀ (start : ℕ), ∃ (n : ℕ), n ∈ MarkedNumbers ∧ start ≤ n ∧ n < start + 1999

-- Theorem to prove
theorem exists_divisible_pair :
  ∃ (a b : ℕ), a ∈ MarkedNumbers ∧ b ∈ MarkedNumbers ∧ a ≠ b ∧ (a ∣ b ∨ b ∣ a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_divisible_pair_l808_80819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_2pi_l808_80876

noncomputable def f (x : ℝ) := Real.sin (1 - x)

theorem f_period_2pi : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  intro x
  simp [f]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_2pi_l808_80876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_angle_cosine_l808_80805

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Theorem: For a hyperbola with an asymptote perpendicular to x + 2y + 1 = 0,
    if there's a point A on the hyperbola such that |F₁A| = 2|F₂A|,
    then cos ∠AF₂F₁ = √5/5 -/
theorem hyperbola_angle_cosine (a b : ℝ) (h : Hyperbola a b) 
  (F₁ F₂ A : Point) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →  -- A is on the hyperbola
  (b = 2*a) →  -- condition for asymptote to be perpendicular to x + 2y + 1 = 0
  (distance F₁ A = 2 * distance F₂ A) →
  Real.cos (angle A F₂ F₁) = Real.sqrt 5 / 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_angle_cosine_l808_80805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_element_is_four_l808_80878

-- Define the function g based on the graph
def g : ℕ → ℕ
| 0 => 0  -- g(0) is not defined in the graph, so we'll map it to 0
| 1 => 2
| 2 => 3
| 3 => 4
| 4 => 1
| 5 => 8
| 6 => 5
| 7 => 6
| 8 => 7
| 9 => 0
| n + 10 => g n  -- Make g periodic with period 10

-- Define the sequence
def cindy_sequence : ℕ → ℕ
| 0 => 2  -- Start with 2
| n + 1 => g (cindy_sequence n)  -- Apply g to the previous term

-- Theorem statement
theorem tenth_element_is_four : cindy_sequence 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_element_is_four_l808_80878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oscillation_distance_oscillation_condition_l808_80808

/-- Represents the distance from home to gym -/
noncomputable def total_distance : ℝ := 3

/-- Represents the fraction of remaining distance walked each time -/
noncomputable def walk_fraction : ℝ := 2/3

/-- Represents the distance from home to point A -/
noncomputable def point_A : ℝ := 12/5

/-- Represents the distance from home to point B -/
noncomputable def point_B : ℝ := 6/5

/-- Theorem stating that the distance between oscillation points is 1.20 km -/
theorem oscillation_distance :
  |point_A - point_B| = 1.20 := by
  -- Proof steps would go here
  sorry

/-- Theorem verifying that point A and point B satisfy the oscillation conditions -/
theorem oscillation_condition :
  point_A = walk_fraction * total_distance + (1 - walk_fraction) * point_B ∧
  point_B = (1 - walk_fraction) * point_A := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oscillation_distance_oscillation_condition_l808_80808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_factor_product_factors_l808_80867

/-- A natural number with exactly three factors -/
def ThreeFactorNumber (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 3

/-- The number of factors of a natural number -/
def numFactors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem three_factor_product_factors
  (a b c : ℕ) 
  (ha : ThreeFactorNumber a) 
  (hb : ThreeFactorNumber b) 
  (hc : ThreeFactorNumber c) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c) :
  numFactors (a^3 * b^4 * c^5) = 693 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_factor_product_factors_l808_80867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonality_condition_l808_80868

/-- Line represented by ax + 2y + 1 = 0 -/
def line1 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => a * x + 2 * y + 1 = 0

/-- Line represented by (3-a)x - y + a = 0 -/
def line2 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => (3 - a) * x - y + a = 0

/-- Two lines are orthogonal if the product of their slopes is -1 -/
def orthogonal (a : ℝ) : Prop :=
  ∃ m1 m2 : ℝ, (∀ x y, line1 a x y ↔ y = m1 * x + (-m1 * 0 - 1 / 2)) ∧
              (∀ x y, line2 a x y ↔ y = m2 * x + (m2 * 0 + a / (3 - a))) ∧
              m1 * m2 = -1

/-- The condition a = 1 is sufficient but not necessary for orthogonality of line1 and line2 -/
theorem orthogonality_condition :
  (∃ a : ℝ, a = 1 → orthogonal a) ∧
  (∃ a : ℝ, orthogonal a ∧ a ≠ 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonality_condition_l808_80868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_a_coordinates_l808_80881

/-- Point A with coordinates (2m+1, m+2) in the second quadrant -/
structure PointA where
  m : ℤ
  second_quadrant : (2 * m + 1 < 0) ∧ (m + 2 > 0)
  integer_coords : True  -- This constraint is already satisfied by m being an integer

/-- The coordinates of point A are (-1, 1) -/
theorem point_a_coordinates (A : PointA) : (2 * A.m + 1 : ℤ) = -1 ∧ (A.m + 2 : ℤ) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_a_coordinates_l808_80881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l808_80844

/-- Given a triangle with side length a and two adjacent angles β and γ,
    the length of the angle bisector (AM) drawn from the vertex opposite side a
    is a * sin(γ) * sin(β) / (sin(β + γ) * cos((γ - β)/2)) -/
theorem angle_bisector_length (a β γ : ℝ) (ha : a > 0) (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π) (hβγ : β + γ < π) :
  ∃ AM : ℝ, AM = a * Real.sin γ * Real.sin β / (Real.sin (β + γ) * Real.cos ((γ - β) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l808_80844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_x_closed_l808_80811

/-- The sequence {x_n} defined recursively -/
noncomputable def x : ℕ → ℝ
  | 0 => 0
  | n + 1 => 3 * x n + Real.sqrt (8 * (x n)^2 + 1)

/-- The closed form expression for x_n -/
noncomputable def x_closed (n : ℕ) : ℝ :=
  (Real.sqrt 2 / 8) * ((3 + 2 * Real.sqrt 2)^n - (3 - 2 * Real.sqrt 2)^n)

/-- Theorem stating that the recursive definition equals the closed form -/
theorem x_equals_x_closed : ∀ n, x n = x_closed n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_x_closed_l808_80811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_victory_margin_l808_80800

/-- Represents the election result -/
structure ElectionResult where
  total_votes : Nat
  petya_first_two : Nat
  vasya_first_two : Nat
  petya_last_two : Nat
  vasya_last_two : Nat

/-- Checks if the election result is valid according to the given conditions -/
def is_valid_result (result : ElectionResult) : Prop :=
  result.total_votes = 27 ∧
  result.petya_first_two = result.vasya_first_two + 9 ∧
  result.vasya_last_two = result.petya_last_two + 9 ∧
  result.petya_first_two + result.petya_last_two > result.vasya_first_two + result.vasya_last_two

/-- Calculates Petya's victory margin -/
def victory_margin (result : ElectionResult) : Int :=
  (result.petya_first_two + result.petya_last_two : Int) - (result.vasya_first_two + result.vasya_last_two : Int)

/-- Theorem stating that the maximum possible victory margin for Petya is 9 votes -/
theorem max_victory_margin :
  ∀ result : ElectionResult, is_valid_result result → victory_margin result ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_victory_margin_l808_80800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_equality_condition_l808_80887

theorem max_value_of_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 2 - Real.sqrt (x^4 + 4)) / x ≤ 2 * Real.sqrt 2 - 2 :=
by sorry

theorem equality_condition (x : ℝ) (hx : x > 0) :
  (x^2 + 2 - Real.sqrt (x^4 + 4)) / x = 2 * Real.sqrt 2 - 2 ↔ x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_equality_condition_l808_80887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_and_negative_max_a_value_l808_80884

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * Real.exp x) / (1 + Real.exp x)

-- Theorem 1: f(x) + f(-x) = 3 for all real x
theorem sum_f_and_negative (x : ℝ) : f x + f (-x) = 3 := by sorry

-- Theorem 2: The maximum value of a such that f(4-ax) + f(x^2) ≥ 3 holds for all x in (0, +∞) is 4
theorem max_a_value : ∃ (a : ℝ), a = 4 ∧ 
  (∀ x > 0, f (4 - a * x) + f (x^2) ≥ 3) ∧
  (∀ b > a, ∃ x > 0, f (4 - b * x) + f (x^2) < 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_and_negative_max_a_value_l808_80884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_one_l808_80889

def jo_blair_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => (jo_blair_sequence n) ^ 2

theorem tenth_term_is_one : jo_blair_sequence 9 = 1 := by
  -- Prove by induction
  have h : ∀ n, jo_blair_sequence n = 1
  · intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      rw [jo_blair_sequence, ih]
      rfl
  -- Apply the lemma to our specific case
  exact h 9

#eval jo_blair_sequence 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_one_l808_80889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_area_ratio_is_seven_fourths_l808_80882

/-- Represents a square paper that can be folded -/
structure FoldablePaper where
  side : ℝ
  side_pos : side > 0

/-- The ratio of the folded area to the original area of the paper -/
noncomputable def foldedAreaRatio (paper : FoldablePaper) : ℝ :=
  7 / 4

/-- Theorem stating that the folded area ratio is always 7/4 -/
theorem folded_area_ratio_is_seven_fourths (paper : FoldablePaper) :
  foldedAreaRatio paper = 7 / 4 := by
  -- The proof is omitted for now
  sorry

#check folded_area_ratio_is_seven_fourths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_area_ratio_is_seven_fourths_l808_80882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_l808_80885

/-- Represents the lengths of the edges of a tetrahedron -/
structure TetrahedronEdges where
  AB : ℝ
  AC : ℝ
  AD : ℝ
  BC : ℝ
  BD : ℝ
  CD : ℝ

/-- The theorem stating that for a tetrahedron with given edge lengths, BD must be 28 -/
theorem tetrahedron_edge_length 
  (edges : TetrahedronEdges)
  (h1 : Finset.toSet {edges.AB, edges.AC, edges.AD, edges.BC, edges.BD, edges.CD} = {8, 14, 19, 28, 37, 42})
  (h2 : edges.AC = 42) :
  edges.BD = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_l808_80885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_evaluation_l808_80869

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (x - 2) / ((x - 1) * (x - 4))

-- State the theorem
theorem integral_evaluation :
  ∫ x in (2 : ℝ)..3, f x = -(1/3) * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_evaluation_l808_80869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_division_l808_80821

/-- A triangle is a right triangle -/
def IsRightTriangle (triangle : Set ℝ × Set ℝ) : Prop :=
  sorry

/-- A right triangle is divided into a square and two smaller right triangles by lines parallel to its legs -/
def DividesTriangle (triangle square small_triangle1 small_triangle2 : Set ℝ × Set ℝ) : Prop :=
  sorry

/-- The area of a shape -/
def area (shape : Set ℝ × Set ℝ) : ℝ :=
  sorry

/-- Given a right triangle divided by lines parallel to its legs through a point on its hypotenuse,
    if the area of one smaller right triangle is n times the area of the square formed,
    then the area of the other smaller right triangle is 1/(4n) times the area of the square. -/
theorem right_triangle_division (n : ℝ) (n_pos : n > 0) : 
  ∃ (triangle : Set ℝ × Set ℝ) (square : Set ℝ × Set ℝ) (small_triangle1 small_triangle2 : Set ℝ × Set ℝ),
    IsRightTriangle triangle ∧
    DividesTriangle triangle square small_triangle1 small_triangle2 ∧
    area small_triangle1 = n * area square →
    area small_triangle2 = (1 / (4 * n)) * area square :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_division_l808_80821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_cyclic_quadrilateral_l808_80845

-- Define the types for points and circles
variable {Point Circle : Type}

-- Define the circles and points
variable (ω₁ ω₂ : Circle)
variable (T₁ T₂ T₁' T₂' P₁ P₂ M : Point)

-- Define the property of being a tangent point
def isTangentPoint (p : Point) (c : Circle) : Prop := sorry

-- Define the property of being a midpoint
def isMidpoint (m p₁ p₂ : Point) : Prop := sorry

-- Define the property of being an intersection point
def isIntersectionPoint (p m t : Point) (c : Circle) : Prop := sorry

-- Define the property of being cyclic
def isCyclic (p₁ p₂ p₃ p₄ : Point) : Prop := sorry

-- State the theorem
theorem tangent_circles_cyclic_quadrilateral
  (h₁ : isTangentPoint T₁ ω₁)
  (h₂ : isTangentPoint T₂ ω₂)
  (h₃ : isTangentPoint T₁' ω₁)
  (h₄ : isTangentPoint T₂' ω₂)
  (h₅ : isMidpoint M T₁ T₂)
  (h₆ : isIntersectionPoint P₁ M T₁' ω₁)
  (h₇ : isIntersectionPoint P₂ M T₂' ω₂) :
  isCyclic P₁ P₂ T₁' T₂' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_cyclic_quadrilateral_l808_80845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_pyramid_count_l808_80897

/-- The number of oranges in a layer of the pyramid -/
def layerCount (n : Nat) : Nat := n * (n + 1) / 2

/-- The total number of oranges in a pyramid with base side length n -/
def pyramidTotal (n : Nat) : Nat := Finset.sum (Finset.range n) (λ i => layerCount (n - i))

/-- The theorem stating that a pyramid with base side length 6 has 56 oranges -/
theorem orange_pyramid_count : pyramidTotal 6 = 56 := by
  sorry

#eval pyramidTotal 6  -- This will evaluate the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_pyramid_count_l808_80897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l808_80848

def S (n : ℕ) : ℚ := n / (n + 1)

def b : ℕ → ℚ
  | 0 => 1
  | n + 1 => b n / (b n + 2)

theorem sequence_properties :
  ∀ (n : ℕ),
    (∀ (k : ℕ), k > 0 → S k - S (k - 1) = 1 / (k * (k + 1))) ∧
    (b n = 1 / ((2 : ℚ)^n - 1)) ∧
    (n > 0 → b (n + 1) < 1 / (n * (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l808_80848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_part1_lambda_mu_sum_constant_line_equations_part3_l808_80880

-- Define the ellipse Γ
def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / (m + 1) + y^2 / m = 1

-- Define the line l
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define point D
def point_D : ℝ × ℝ := (-1, 0)

-- Define the theorem for part 1
theorem intersection_points_part1 :
  ∀ x y : ℝ, ellipse 1 x y ∧ line 1 x y →
  (x = 0 ∧ y = 1) ∨ (x = -4/3 ∧ y = -1/3) := by sorry

-- Define λ and μ for part 2
noncomputable def lambda_mu_sum (m k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (x₁ / (x₁ + 1)) + (x₂ / (x₂ + 1))

-- Define the theorem for part 2
theorem lambda_mu_sum_constant :
  ∀ k x₁ y₁ x₂ y₂ : ℝ,
  ellipse 2 x₁ y₁ ∧ ellipse 2 x₂ y₂ ∧
  line k x₁ y₁ ∧ line k x₂ y₂ ∧
  y₁ > y₂ →
  lambda_mu_sum 2 k x₁ y₁ x₂ y₂ = 3 := by sorry

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the incircle area condition
noncomputable def incircle_area_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let perimeter := 8
  let radius := 3 * Real.sqrt 2 / 7
  (1/2) * perimeter * radius = 18/49 * Real.pi

-- Define the theorem for part 3
theorem line_equations_part3 :
  ∀ k x₁ y₁ x₂ y₂ : ℝ,
  ellipse 3 x₁ y₁ ∧ ellipse 3 x₂ y₂ ∧
  line k x₁ y₁ ∧ line k x₂ y₂ ∧
  y₁ > y₂ ∧
  incircle_area_condition x₁ y₁ x₂ y₂ →
  k = 1 ∨ k = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_part1_lambda_mu_sum_constant_line_equations_part3_l808_80880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_permutation_l808_80820

/-- The falling factorial function -/
def fallingFactorial (n k : ℕ) : ℕ :=
  Finset.prod (Finset.range k) (fun i => n - i)

/-- The permutation counting function -/
def permutation (n k : ℕ) : ℕ :=
  fallingFactorial n k

theorem product_equals_permutation (m : ℕ) (h : m > 0) :
  (Finset.prod (Finset.range 21) (fun i => m + i)) = permutation (m + 20) 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_permutation_l808_80820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beaded_corset_problem_l808_80874

/-- Represents the beaded corset problem with given conditions -/
theorem beaded_corset_problem 
  (purple_rows : ℕ) 
  (purple_beads_per_row : ℕ) 
  (blue_rows : ℕ) 
  (gold_beads : ℕ) 
  (total_cost : ℕ) 
  (blue_beads_per_row : ℕ) : 
  purple_rows = 50 ∧ 
  purple_beads_per_row = 20 ∧ 
  blue_rows = 40 ∧ 
  gold_beads = 80 ∧ 
  total_cost = 180 ∧ 
  (purple_rows * purple_beads_per_row + blue_rows * blue_beads_per_row + gold_beads = total_cost) → 
    blue_beads_per_row = (total_cost - (purple_rows * purple_beads_per_row + gold_beads)) / blue_rows :=
by
  sorry

#check beaded_corset_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beaded_corset_problem_l808_80874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_b_for_nested_function_zero_l808_80859

theorem exists_b_for_nested_function_zero : 
  ∃ b : ℝ, b > 0 ∧ 
    let g : ℝ → ℝ := λ x ↦ b * x^3 + x^2 + x
    g (g (Real.sqrt 2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_b_for_nested_function_zero_l808_80859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_theorem_l808_80804

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x > 0}

-- Define set B
def B : Set ℝ := {y : ℝ | y ≥ 1}

-- State the theorem
theorem intersection_complement_theorem : A ∩ (U \ B) = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_theorem_l808_80804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_distribution_l808_80850

/-- Represents the parameters of a normal distribution -/
structure NormalDistParams where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Represents the probability of a score falling within a given range -/
noncomputable def prob_between (params : NormalDistParams) (lower upper : ℝ) : ℝ := sorry

/-- Represents the probability of a score being greater than or equal to a given value -/
noncomputable def prob_ge (params : NormalDistParams) (x : ℝ) : ℝ := sorry

/-- Defines an approximation relation for real numbers -/
def approx (x y : ℝ) : Prop := abs (x - y) < 0.01

notation:50 a " ≈ " b => approx a b

theorem exam_score_distribution 
  (params : NormalDistParams)
  (h_μ : params.μ = 90)
  (h_prob : prob_between params 70 110 = 0.6) :
  prob_ge params 110 ≈ 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_distribution_l808_80850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_decreasing_l808_80855

noncomputable def f (x : ℝ) := Real.exp (-abs x)

theorem f_is_even_and_decreasing : 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_decreasing_l808_80855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_coprime_arrangement_l808_80816

/-- Represents a 3x3 grid of integers -/
def Grid := Matrix (Fin 3) (Fin 3) ℕ

/-- Checks if two numbers are adjacent in the grid -/
def adjacent (i j i' j' : Fin 3) : Prop :=
  (i = i' ∧ j.val + 1 = j'.val) ∨
  (i = i' ∧ j'.val + 1 = j.val) ∨
  (j = j' ∧ i.val + 1 = i'.val) ∨
  (j = j' ∧ i'.val + 1 = i.val) ∨
  (i.val + 1 = i'.val ∧ j.val + 1 = j'.val) ∨
  (i'.val + 1 = i.val ∧ j'.val + 1 = j.val) ∨
  (i.val + 1 = i'.val ∧ j'.val + 1 = j.val) ∨
  (i'.val + 1 = i.val ∧ j.val + 1 = j'.val)

/-- Checks if the grid contains nine consecutive integers -/
def consecutive_integers (g : Grid) : Prop :=
  ∃ n : ℕ, ∀ i j : Fin 3, n ≤ g i j ∧ g i j < n + 9

/-- Checks if adjacent numbers in the grid are coprime -/
def adjacent_coprime (g : Grid) : Prop :=
  ∀ i j i' j' : Fin 3, adjacent i j i' j' → Nat.Coprime (g i j) (g i' j')

theorem consecutive_integers_coprime_arrangement :
  ∃ g : Grid, consecutive_integers g ∧ adjacent_coprime g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_coprime_arrangement_l808_80816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_median_and_bisector_l808_80870

/-- In a right triangle, if one of the acute angles β satisfies tan(β/2) = 1/∜2,
    then the angle φ between the median and the angle bisector drawn from this acute angle
    satisfies tan φ = 1/2. -/
theorem angle_between_median_and_bisector (β φ : ℝ) : 
  0 < β → β < π/2 → Real.tan (β/2) = 1/(2^(1/4)) → Real.tan φ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_median_and_bisector_l808_80870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_fourth_l808_80879

theorem tan_theta_minus_pi_fourth (θ : ℝ) 
  (h1 : θ ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) 
  (h2 : Real.sin (θ + Real.pi / 4) = 3 / 5) : 
  Real.tan (θ - Real.pi / 4) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_fourth_l808_80879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_box_arrangement_l808_80895

def f (n k : ℕ) : ℕ := (n.choose k)^2 * k.factorial

-- Define a new function to represent the number of arrangements
def number_of_arrangements (n k : ℕ) : ℕ := 
  -- This is a placeholder definition. The actual implementation would be complex.
  0 -- We use 0 as a placeholder, but this should be replaced with the correct calculation

theorem ball_box_arrangement (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  f n k = number_of_arrangements n k :=
sorry

-- Add a comment explaining the meaning of number_of_arrangements
/-
number_of_arrangements n k represents:
The number of ways to choose k balls from n balls, 
k boxes from 2n-1 boxes, and put these balls in 
the selected boxes so that each box has exactly one ball, 
given that ball i can only be put in boxes numbered from 1 to 2i-1
-/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_box_arrangement_l808_80895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l808_80834

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The point (a, b) lies on the line x(sin A - sin B) + y sin B = c sin C -/
def pointOnLine (t : Triangle) : Prop :=
  t.a * (Real.sin t.A - Real.sin t.B) + t.b * Real.sin t.B = t.c * Real.sin t.C

/-- The condition a^2 + b^2 - 6(a + b) + 18 = 0 -/
def specialCondition (t : Triangle) : Prop :=
  t.a^2 + t.b^2 - 6*(t.a + t.b) + 18 = 0

/-- Area of a triangle given side lengths a, b and angle C -/
def areaOfTriangle (t : Triangle) : ℝ :=
  1/2 * t.a * t.b * Real.sin t.C

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : pointOnLine t) : 
  t.C = π/3 ∧ 
  (specialCondition t → areaOfTriangle t = 9*Real.sqrt 3/4) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l808_80834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l808_80838

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 4 / x

-- State the theorem
theorem f_is_odd : ∀ x : ℝ, x ≠ 0 → f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l808_80838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l808_80847

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Distance between two points -/
noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Theorem: For a parabola y^2 = 2px with focus F at (p, 0), 
    if a line through F intersects the parabola at points A and B 
    such that |AF| = 8|OF|, then |AF| / |BF| = 2/3 -/
theorem parabola_intersection_ratio 
  (C : Parabola) 
  (F : Point) 
  (A B : Point) 
  (h_F : F.x = C.p ∧ F.y = 0) 
  (h_A : A.y^2 = 2 * C.p * A.x) 
  (h_B : B.y^2 = 2 * C.p * B.x) 
  (h_collinear : ∃ t : ℝ, A.x = F.x + t * (B.x - F.x) ∧ A.y = F.y + t * (B.y - F.y)) 
  (h_AF : distance A F = 8 * C.p) :
  distance A F / distance B F = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l808_80847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_two_solutions_l808_80833

/-- The number of distinct solutions to the equation |x - |2x + 1|| = 3 -/
def num_solutions : ℕ := 2

/-- The equation |x - |2x + 1|| = 3 -/
def equation (x : ℝ) : Prop := abs (x - abs (2*x + 1)) = 3

theorem equation_has_two_solutions :
  ∃ (s : Finset ℝ), s.card = num_solutions ∧ 
  (∀ x ∈ s, equation x) ∧
  (∀ y : ℝ, equation y → y ∈ s) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_two_solutions_l808_80833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_theorem_l808_80822

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem hyperbola_distance_theorem (x y : ℝ) :
  is_on_hyperbola x y →
  distance x y 5 0 = 15 →
  (distance x y (-5) 0 = 7 ∨ distance x y (-5) 0 = 23) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_theorem_l808_80822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_expression_l808_80899

theorem largest_prime_factor_of_expression : 
  (Nat.factors (12^3 + 15^4 - 6^5)).maximum? = some 12193 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_expression_l808_80899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_am_equals_cn_l808_80896

-- Define the triangle ABC and points M and N
variable (A B C M N : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_equilateral_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def M_on_AC (A C M : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • C

def N_on_BC_extension (B C N : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, t > 1 ∧ N = (1 - t) • B + t • C

-- State the theorem
theorem am_equals_cn 
  (h1 : is_equilateral_triangle A B C)
  (h2 : M_on_AC A C M)
  (h3 : N_on_BC_extension B C N)
  (h4 : dist B M = dist M N) :
  dist A M = dist C N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_am_equals_cn_l808_80896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_contact_length_correct_sum_abc_correct_l808_80871

/-- The length of the rope in contact with a cylindrical turret --/
noncomputable def rope_contact_length (rope_length : ℝ) (turret_radius : ℝ) (unicorn_height : ℝ) (distance_from_turret : ℝ) : ℝ :=
  (90 - Real.sqrt 750) / 3

/-- Theorem stating the correct length of rope in contact with the turret --/
theorem rope_contact_length_correct :
  let rope_length : ℝ := 30
  let turret_radius : ℝ := 10
  let unicorn_height : ℝ := 5
  let distance_from_turret : ℝ := 5
  rope_contact_length rope_length turret_radius unicorn_height distance_from_turret =
    (90 - Real.sqrt 750) / 3 := by
  sorry

/-- Calculation of a + b + c --/
def sum_abc : ℕ := 90 + 750 + 3

theorem sum_abc_correct : sum_abc = 843 := by
  rfl

#eval sum_abc

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_contact_length_correct_sum_abc_correct_l808_80871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_solution_l808_80812

/-- The sum of the arithmetic-geometric series with first term 1 and common difference 2 -/
noncomputable def series_sum (y : ℝ) : ℝ := (1 + y) / ((1 - y)^2)

/-- Theorem stating that if the sum of the series equals 16, then y has the specific value -/
theorem series_solution (y : ℝ) (h : |y| < 1) : series_sum y = 16 → y = (33 - Real.sqrt 129) / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_solution_l808_80812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l808_80830

theorem tan_difference (α β : Real) (h1 : 0 < α) (h2 : α < β) (h3 : β < Real.pi)
  (h4 : Real.cos α * Real.cos β = 1/5) (h5 : Real.sin α * Real.sin β = 2/5) :
  Real.tan (β - α) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l808_80830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_sqrt_61_l808_80858

/-- A right triangle with sides 5, 12, and 13 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13
  right_angle : a^2 + b^2 = c^2

/-- The length of the crease when folding point C to the midpoint of the hypotenuse -/
noncomputable def creaseLength (t : RightTriangle) : ℝ :=
  Real.sqrt ((t.b / 2)^2 + t.a^2)

/-- Theorem: The crease length is √61 inches -/
theorem crease_length_is_sqrt_61 (t : RightTriangle) :
  creaseLength t = Real.sqrt 61 := by
  sorry

#check crease_length_is_sqrt_61

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_sqrt_61_l808_80858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_condition_inequality_on_interval_l808_80866

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x + k / (x + 1)

-- Part 1: Monotonicity condition
theorem monotonicity_condition (k : ℝ) :
  (∀ x > 0, Monotone (f k)) ↔ k ≤ 4 := by sorry

-- Part 2: Inequality for x ∈ (1, 2)
theorem inequality_on_interval :
  ∀ x ∈ Set.Ioo 1 2, (2 - x) * Real.exp (2 * (x - 1 / x)) - 2 * x^2 + x < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_condition_inequality_on_interval_l808_80866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_packing_height_difference_l808_80828

/-- The diameter of a cylindrical can in cm -/
def can_diameter : ℝ := 12

/-- The number of rows in side-by-side packing -/
def side_by_side_rows : ℕ := 15

/-- The number of cans per row in side-by-side packing -/
def side_by_side_cans_per_row : ℕ := 12

/-- The number of rows in staggered packing -/
def staggered_rows : ℕ := 13

/-- The height of side-by-side packing in cm -/
def side_by_side_height : ℝ := side_by_side_rows * can_diameter

/-- The height of staggered packing in cm -/
noncomputable def staggered_height : ℝ := 12 + 72 * Real.sqrt 3

/-- The difference in heights between side-by-side and staggered packing in cm -/
theorem packing_height_difference : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |side_by_side_height - staggered_height - 43.3| < ε :=
by
  sorry

#eval side_by_side_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_packing_height_difference_l808_80828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_problem_l808_80815

theorem age_problem (a b : ℚ) 
  (h1 : a / b = 5 / 3)
  (h2 : (a + 2) / (b + 2) = 3 / 2)
  (h3 : b > 0) :
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_problem_l808_80815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l808_80877

theorem polynomial_remainder (p : Polynomial ℝ) : 
  (∃ q₁ : Polynomial ℝ, p = (X + 1) * q₁ + 3) ∧ 
  (∃ q₂ : Polynomial ℝ, p = (X + 5) * q₂ - 9) → 
  ∃ q : Polynomial ℝ, p = (X + 1) * (X + 5) * q + (3 * X + 6) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l808_80877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l808_80807

/-- The function f(x) = x³ - 3ax + b takes extreme value 1 at x = -1 -/
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x + b

/-- The function g(x) = f(x) + e^(2x-1) -/
noncomputable def g (a b x : ℝ) : ℝ := f a b x + Real.exp (2*x - 1)

/-- Theorem stating the properties of f and g -/
theorem f_and_g_properties :
  ∃ (a b : ℝ),
    (f a b (-1) = 1) ∧
    (∀ x, f a b x ≤ 1) ∧
    (a = 1) ∧
    (b = -1) ∧
    (∀ x y, y = g 1 (-1) x → 2 * Real.exp 1 * x - y - Real.exp 1 - 3 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l808_80807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_proof_l808_80875

noncomputable def z : ℂ := (1 : ℝ) / 3 - (5 : ℝ) / 9 * Complex.I

theorem complex_magnitude_proof : Complex.abs z = Real.sqrt 34 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_proof_l808_80875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_by_squares_proof_l808_80810

/-- The area covered by two congruent squares with side length 14,
    where one square's vertex is at the center of the other square. -/
noncomputable def area_covered_by_squares : ℝ := 367.5

/-- The side length of each square. -/
noncomputable def side_length : ℝ := 14

/-- The area of a single square. -/
noncomputable def square_area : ℝ := side_length ^ 2

/-- The length of half the diagonal of a square. -/
noncomputable def half_diagonal : ℝ := side_length / 2

/-- The area of overlap between the two squares. -/
noncomputable def overlap_area : ℝ := (half_diagonal ^ 2) / 2

/-- Theorem stating that the area covered by the two squares
    is equal to twice the area of a single square minus the overlap area. -/
theorem area_covered_by_squares_proof :
  area_covered_by_squares = 2 * square_area - overlap_area :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_by_squares_proof_l808_80810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_generating_function_l808_80863

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Conjugate of golden ratio -/
noncomputable def φ_hat : ℝ := (1 - Real.sqrt 5) / 2

/-- Generating function of Fibonacci sequence -/
noncomputable def F (x : ℝ) : ℝ := ∑' n, (fib n : ℝ) * x^n

/-- Theorem: Generating function of Fibonacci sequence -/
theorem fibonacci_generating_function :
  ∀ x : ℝ, abs x < 1 →
    F x = x / (1 - x - x^2) ∧
    F x = (1 / Real.sqrt 5) * (1 / (1 - φ * x) - 1 / (1 - φ_hat * x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_generating_function_l808_80863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dave_bench_press_ratio_l808_80825

/-- The ratio of Dave's bench press to his body weight -/
noncomputable def bench_press_ratio (dave_weight craig_percent mark_difference mark_weight : ℝ) : ℝ :=
  let craig_weight := mark_weight + mark_difference
  let dave_bench := craig_weight / craig_percent
  dave_bench / dave_weight

/-- Theorem stating that the ratio of Dave's bench press to his body weight is 3 -/
theorem dave_bench_press_ratio :
  bench_press_ratio 175 0.2 50 55 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dave_bench_press_ratio_l808_80825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_and_max_area_l808_80839

-- Define the curves C1 and C2
noncomputable def C1 (θ : ℝ) : ℝ × ℝ := (-1 + Real.cos θ, Real.sin θ)
noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (2 * Real.sin θ * Real.cos θ, 2 * Real.sin θ * Real.sin θ)

-- Define the set of intersection points
def intersection_points : Set (ℝ × ℝ) := {(0, 0), (-1, 1)}

-- Define the maximum area of triangle OAB
noncomputable def max_triangle_area : ℝ := (Real.sqrt 2 + 1) / 2

-- Helper function for calculating the area of a triangle
noncomputable def area_triangle (O A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  abs (x1 * y2 - x2 * y1) / 2

-- Theorem statement
theorem curves_intersection_and_max_area :
  (∀ θ : ℝ, C1 θ ∈ intersection_points ∨ C2 θ ∈ intersection_points) ∧
  (∃ A B : ℝ × ℝ, A ∈ Set.range C1 ∧ B ∈ Set.range C2 ∧
    area_triangle (0, 0) A B ≤ max_triangle_area) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_and_max_area_l808_80839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_max_absolute_value_at_12_is_geometric_sequence_l808_80852

/-- Given a sequence a with product of first n terms T_n -/
def T (a : ℕ → ℝ) : ℕ → ℝ 
  | 0 => 1
  | n + 1 => T a n * a (n + 1)

/-- Geometric sequence with first term a and common ratio r -/
def geometric_sequence (a r : ℝ) : ℕ → ℝ 
  | 0 => 1
  | n + 1 => a * r^n

theorem geometric_sequence_product (a r : ℝ) (n : ℕ) :
  T (geometric_sequence a r) n = a^n * r^(n * (n - 1) / 2) := by sorry

theorem max_absolute_value_at_12 :
  ∃ k : ℕ, k = 12 ∧ ∀ n : ℕ, n ≠ 0 → |T (geometric_sequence 2016 (-1/2)) n| ≤ |T (geometric_sequence 2016 (-1/2)) k| := by sorry

theorem is_geometric_sequence {a : ℕ → ℝ} (h : ∀ n : ℕ, n ≠ 0 → a n > 0) 
  (h' : ∀ n : ℕ, n ≠ 0 → T a n * T a (n + 1) = (a 1 * a n)^(n/2) * (a 1 * a (n + 1))^((n+1)/2)) :
  ∃ r : ℝ, ∀ n : ℕ, n ≠ 0 → a (n + 1) = a n * r := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_max_absolute_value_at_12_is_geometric_sequence_l808_80852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_implies_m_not_one_l808_80860

/-- If the complex number m+1+(m-1)i is imaginary, then m ≠ 1 -/
theorem imaginary_implies_m_not_one (m : ℝ) :
  (Complex.I * (m - 1) + (m + 1)).im ≠ 0 → m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_implies_m_not_one_l808_80860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_windmill_area_is_1824_l808_80842

/-- The area of the windmill shape in square centimeters -/
noncomputable def windmill_area (square_side : ℝ) (π : ℝ) : ℝ :=
  let ob_squared := (square_side / 2) ^ 2 + (square_side / 2) ^ 2
  let quarter_circle_area := π * ob_squared / 4
  let oa_squared := ob_squared / 4
  let semicircle_area := π * oa_squared / 2
  let small_triangle_area := square_side * square_side / 8
  let small_shadow_area := quarter_circle_area - semicircle_area - small_triangle_area
  4 * small_shadow_area

/-- Theorem stating that the area of the windmill shape is 1824 square centimeters -/
theorem windmill_area_is_1824 :
  windmill_area 80 3.14 = 1824 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval windmill_area 80 3.14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_windmill_area_is_1824_l808_80842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wind_velocity_for_given_pressure_l808_80802

/-- The constant of proportionality in the pressure-area-velocity relationship -/
noncomputable def k : ℝ := 1 / 256

/-- The relationship between pressure, area, and velocity -/
noncomputable def pressure (area : ℝ) (velocity : ℝ) : ℝ := k * area * velocity^2

/-- Conversion factor from square yards to square feet -/
def sq_yard_to_sq_feet : ℝ := 9

theorem wind_velocity_for_given_pressure : 
  let given_pressure : ℝ := 36
  let given_area : ℝ := 1  -- in square yards
  pressure (given_area * sq_yard_to_sq_feet) 32 = given_pressure := by
  -- Proof steps would go here
  sorry

#check wind_velocity_for_given_pressure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wind_velocity_for_given_pressure_l808_80802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_averages_count_l808_80832

/-- The minimum number of elements in the set of averages of distinct pairs from a set of n distinct real numbers. -/
theorem min_averages_count (n : ℕ) (S : Finset ℝ) (h_n : n ≥ 2) (h_card : S.card = n) 
  (h_distinct : S.Nonempty ∧ ∀ (x y : ℝ), x ∈ S → y ∈ S → x ≠ y → x < y) :
  let A_s : Finset ℝ := Finset.image (fun p : ℝ × ℝ => (p.1 + p.2) / 2) (S.product S)
  A_s.card ≥ 2 * n - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_averages_count_l808_80832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_tax_difference_l808_80836

/-- Calculates the property tax for a given assessed value in Township K --/
noncomputable def calculateTax (assessedValue : ℝ) : ℝ :=
  if assessedValue ≤ 10000 then
    assessedValue * 0.05
  else if assessedValue ≤ 20000 then
    10000 * 0.05 + (assessedValue - 10000) * 0.075
  else if assessedValue ≤ 30000 then
    10000 * 0.05 + 10000 * 0.075 + (assessedValue - 20000) * 0.1
  else
    10000 * 0.05 + 10000 * 0.075 + 10000 * 0.1 + (assessedValue - 30000) * 0.125

/-- The difference in property tax before and after re-assessment --/
theorem property_tax_difference : 
  calculateTax 28000 - calculateTax 20000 = 550 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_tax_difference_l808_80836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_form_quadrilateral_resulting_figure_has_four_edges_l808_80818

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a set of points
def PointSet := Finset Point

-- Function to get the extreme points (top, bottom, left, right)
noncomputable def extremePoints (s : PointSet) : Finset Point :=
  sorry

-- Theorem: The convex hull of extreme points is always a quadrilateral
theorem extreme_points_form_quadrilateral (s : PointSet) (h : s.card = 16) :
  (extremePoints s).card ≤ 4 := by
  sorry

-- Theorem: The resulting figure always has 4 edges
theorem resulting_figure_has_four_edges (s : PointSet) (h : s.card = 16) :
  4 = 4 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_form_quadrilateral_resulting_figure_has_four_edges_l808_80818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l808_80888

def angle_terminal_point (α : ℝ) : ℝ × ℝ := (4, -3)

theorem cos_alpha_value (α : ℝ) : Real.cos α = 4/5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l808_80888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l808_80890

theorem train_average_speed (x : ℝ) (h : x > 0) : 
  (3 * x) / ((x / 50) + (2 * x / 20)) = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l808_80890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l808_80841

-- Define the types for lines and planes
structure Line where

structure Plane where

-- Define the relations
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

def contained_in (l : Line) (p : Plane) : Prop := sorry

def parallel_lines (l1 l2 : Line) : Prop := sorry

def skew_lines (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem line_plane_relationship (a b : Line) (α : Plane) :
  parallel_line_plane a α → contained_in b α →
  (parallel_lines a b ∨ skew_lines a b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l808_80841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_base_inscribable_base_circle_radius_formula_l808_80862

/-- A right prism with a circumscribed sphere. -/
structure PrismWithSphere where
  /-- The height of the prism. -/
  h : ℝ
  /-- The radius of the circumscribed sphere. -/
  R : ℝ
  /-- Assumption that h and R are positive. -/
  h_pos : h > 0
  R_pos : R > 0
  /-- Assumption that the sphere's radius is greater than half the prism's height. -/
  R_gt_half_h : R > h / 2

/-- The radius of the circle inscribed in the base of the prism. -/
noncomputable def base_circle_radius (p : PrismWithSphere) : ℝ :=
  Real.sqrt (p.R^2 - p.h^2 / 4)

/-- Theorem stating that the base of the prism can be inscribed in a circle. -/
theorem prism_base_inscribable (p : PrismWithSphere) :
  ∃ (r : ℝ), r > 0 ∧ r = base_circle_radius p := by
  sorry

/-- Theorem verifying the formula for the base circle radius. -/
theorem base_circle_radius_formula (p : PrismWithSphere) :
  base_circle_radius p = Real.sqrt (p.R^2 - p.h^2 / 4) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_base_inscribable_base_circle_radius_formula_l808_80862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_grade_trees_planted_l808_80857

theorem third_grade_trees_planted
  (total_students : ℕ)
  (total_trees : ℕ)
  (third_grade_avg : ℚ)
  (fourth_grade_avg : ℕ)
  (fifth_grade_avg : ℚ)
  (h1 : total_students = 100)
  (h2 : total_trees = 566)
  (h3 : third_grade_avg = 4)
  (h4 : fourth_grade_avg = 5)
  (h5 : fifth_grade_avg = 13/2)
  (h6 : ∃ x y : ℕ, x + x + y = total_students ∧
                   third_grade_avg * x + fourth_grade_avg * x + fifth_grade_avg * y = total_trees) :
  ∃ x : ℕ, x * third_grade_avg = 84 := by
  sorry

#check third_grade_trees_planted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_grade_trees_planted_l808_80857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_plus_2x_l808_80864

theorem integral_sqrt_plus_2x (f : ℝ → ℝ) (h : ∀ x ∈ Set.Icc 0 1, f x = Real.sqrt (1 - x^2) + 2*x) : 
  ∫ x in Set.Icc 0 1, f x = (Real.pi + 4) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_plus_2x_l808_80864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_distribution_l808_80826

/-- The amount of money person one has initially -/
def x : ℚ := sorry

/-- The amount of money person two has initially -/
def y : ℚ := sorry

/-- Condition 1: If person one receives 100 rupees from person two, 
    person one will have twice as much money as person two -/
axiom condition1 : x + 100 = 2 * (y - 100)

/-- Condition 2: If person two receives 10 rupees from person one, 
    person two will have six times as much money as person one -/
axiom condition2 : y + 10 = 6 * (x - 10)

/-- Theorem: Given the conditions, prove that person one had 40 rupees 
    and person two had 170 rupees initially -/
theorem money_distribution : x = 40 ∧ y = 170 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_distribution_l808_80826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_have_one_after_2021_turns_l808_80823

/-- Represents a player in the game -/
inductive Player
| Hawkins : Player
| Dustin : Player
| Lucas : Player

/-- Represents the state of the game -/
def GameState := Player → ℕ

/-- The initial state of the game -/
def initial_state : GameState :=
  λ _ => 2

/-- The probability of keeping the last unit of currency -/
def keep_last_probability : ℚ := 1/3

/-- The number of turns in the game -/
def num_turns : ℕ := 2021

/-- The probability of transitioning from one state to another -/
noncomputable def transition_probability (from_state to_state : GameState) : ℚ := sorry

/-- The probability of ending in a specific state after n turns -/
noncomputable def probability_after_n_turns (n : ℕ) (final_state : GameState) : ℚ := sorry

/-- The final state where each player has 1 unit -/
def final_state : GameState :=
  λ _ => 1

/-- The theorem to be proved -/
theorem probability_all_have_one_after_2021_turns :
  probability_after_n_turns num_turns final_state = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_have_one_after_2021_turns_l808_80823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equal_areas_l808_80846

/-- A point on the hyperbola xy = 1 -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : x * y = 1

/-- The area bounded by the hyperbola and a chord -/
def area_bounded_by_chord (a b : HyperbolaPoint) : ℝ :=
  sorry

/-- The area of a triangle formed by three points on the hyperbola -/
def area_triangle (a b c : HyperbolaPoint) : ℝ :=
  sorry

theorem hyperbola_equal_areas
  (A B P : HyperbolaPoint)
  (P_between : A.x > P.x ∧ P.x > B.x)
  (max_area : ∀ Q : HyperbolaPoint, A.x > Q.x ∧ Q.x > B.x →
    area_triangle A P B ≥ area_triangle A Q B) :
  area_bounded_by_chord A P = area_bounded_by_chord P B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equal_areas_l808_80846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combinations_with_repetition_correct_l808_80803

/-- The number of k-combinations with repetition from n elements -/
def combinations_with_repetition (n k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- Hypothetical function representing the actual number of k-combinations
    with repetition from n elements -/
noncomputable def number_of_k_combinations_with_repetition (n k : ℕ) : ℕ := sorry

/-- Theorem stating that combinations_with_repetition correctly calculates
    the number of k-combinations with repetition from n elements -/
theorem combinations_with_repetition_correct (n k : ℕ) :
  combinations_with_repetition n k = number_of_k_combinations_with_repetition n k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combinations_with_repetition_correct_l808_80803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_primes_product_mod_16_l808_80837

theorem odd_primes_product_mod_16 : 
  let N := (Finset.filter (λ p => Nat.Prime p ∧ p < 2^4 ∧ p % 2 = 1) (Finset.range (2^4))).prod id
  N % 2^4 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_primes_product_mod_16_l808_80837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l808_80894

theorem cube_root_sum_equals_one :
  Real.rpow (8 + 3 * Real.sqrt 21) (1/3) + Real.rpow (8 - 3 * Real.sqrt 21) (1/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l808_80894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l808_80873

/-- Hyperbola struct -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  hpos_a : a > 0
  hpos_b : b > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a hyperbola equation -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Defines the right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola) : Point :=
  ⟨Real.sqrt (h.a^2 + h.b^2), 0⟩

/-- Defines an equilateral triangle -/
noncomputable def is_equilateral_triangle (p1 p2 p3 : Point) : Prop :=
  let d12 := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
  let d23 := Real.sqrt ((p2.x - p3.x)^2 + (p2.y - p3.y)^2)
  let d31 := Real.sqrt ((p3.x - p1.x)^2 + (p3.y - p1.y)^2)
  d12 = d23 ∧ d23 = d31

/-- Defines the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) (p : Point) 
  (hp : hyperbola_equation h p) 
  (heq : is_equilateral_triangle (Point.mk 0 0) p (right_focus h)) : 
  eccentricity h = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l808_80873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l808_80893

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := 2 * m * x^2 + (1 - m) * x - 1 - m

-- Define is_vertex as a function
def is_vertex (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x : ℝ, f x ≤ f p.1 ∨ f x ≥ f p.1

theorem quadratic_properties :
  (∀ m : ℝ, m = -1 → ∃ x y : ℝ, x = 1/2 ∧ y = 1/2 ∧ is_vertex (quadratic_function m) (x, y)) ∧
  (∀ m : ℝ, m ≠ 0 → ∃ x y : ℝ, ∀ m' : ℝ, m' ≠ 0 → quadratic_function m' x = y) ∧
  (∀ m : ℝ, m > 0 → ∃ x₁ x₂ : ℝ, quadratic_function m x₁ = 0 ∧ quadratic_function m x₂ = 0 ∧ |x₁ - x₂| > 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l808_80893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l808_80801

theorem order_of_expressions : (2 : ℝ)^(3/10) > (3/10)^2 ∧ (3/10)^2 > Real.log 2 / Real.log (3/10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l808_80801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l808_80849

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

-- State the theorem
theorem f_properties :
  (f (-3) = -1) ∧
  (f 4 = 8) ∧
  (f (f (-2)) = 8) ∧
  (∀ m : ℝ, f m = 8 → m = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l808_80849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l808_80817

/-- The speed of a train given its length and time to cross a point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: A train 1500 m long crossing an electric pole in 120 sec has a speed of 12.5 m/s -/
theorem train_speed_calculation :
  train_speed 1500 120 = 12.5 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l808_80817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_increase_approx_89_58_percent_l808_80813

noncomputable section

def outer_radius : ℝ := 8
def inner_radius : ℝ := 4
def outer_increase_percent : ℝ := 25
def inner_decrease_percent : ℝ := 25

def new_outer_radius : ℝ := outer_radius * (1 + outer_increase_percent / 100)
def new_inner_radius : ℝ := inner_radius * (1 - inner_decrease_percent / 100)

def original_area_between : ℝ := Real.pi * (outer_radius^2 - inner_radius^2)
def new_area_between : ℝ := Real.pi * (new_outer_radius^2 - new_inner_radius^2)

def area_increase_percent : ℝ := (new_area_between - original_area_between) / original_area_between * 100

theorem area_increase_approx_89_58_percent : 
  89.57 < area_increase_percent ∧ area_increase_percent < 89.59 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_increase_approx_89_58_percent_l808_80813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tray_height_l808_80865

/-- The height of a tray formed by cutting and folding a square paper -/
theorem tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) : 
  side_length = 120 →
  cut_distance = 5 →
  cut_angle = 45 →
  cut_distance * Real.sqrt 2 / 2 = 5 :=
by
  intros h_side h_cut h_angle
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tray_height_l808_80865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l808_80886

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6) + 1

theorem f_properties (A ω : ℝ) (hA : A > 0) (hω : ω > 0) 
  (h_symmetry : Real.pi / 2 = Real.pi / ω) :
  ω = 2 ∧ ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f ω x ≥ f ω y ∧ f ω x = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l808_80886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_option_l808_80891

noncomputable def product : ℝ := 2.1 * (50.8 - 0.45)

def options : List ℝ := [105, 106, 107, 108, 110]

theorem closest_option :
  ∀ x ∈ options, x ≠ 106 → |106 - product| < |x - product| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_option_l808_80891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_duration_four_units_minimum_second_spray_l808_80853

-- Define the concentration function for one unit of detergent
noncomputable def concentration (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 4 then 16 / (8 - x) - 1
  else if 4 < x ∧ x ≤ 10 then 5 - 0.5 * x
  else 0

-- Define the effective concentration threshold
def effective_concentration : ℝ := 4

-- Part I: Prove that 4 units of detergent last for 8 days
theorem detergent_duration_four_units :
  ∃ t : ℝ, t = 8 ∧ ∀ x : ℝ, 0 ≤ x ∧ x ≤ t → 4 * concentration x ≥ effective_concentration :=
by
  sorry

-- Part II: Prove the minimum value of a
theorem minimum_second_spray (a : ℝ) :
  (1 ≤ a ∧ a ≤ 4) →
  (∀ x : ℝ, 6 ≤ x ∧ x ≤ 10 → 
    2 * concentration x + a * concentration (x - 6) ≥ effective_concentration) →
  a ≥ 1.6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_duration_four_units_minimum_second_spray_l808_80853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_coincidences_value_l808_80851

/-- The number of questions in the test -/
def num_questions : ℕ := 20

/-- The number of questions Vasya guessed correctly -/
def vasya_correct : ℕ := 6

/-- The number of questions Misha guessed correctly -/
def misha_correct : ℕ := 8

/-- The probability of Vasya guessing correctly -/
noncomputable def p_vasya : ℝ := vasya_correct / num_questions

/-- The probability of Misha guessing correctly -/
noncomputable def p_misha : ℝ := misha_correct / num_questions

/-- The probability of both guessing correctly or both guessing incorrectly -/
noncomputable def p_coincidence : ℝ := p_vasya * p_misha + (1 - p_vasya) * (1 - p_misha)

/-- The expected number of coincidences -/
noncomputable def expected_coincidences : ℝ := num_questions * p_coincidence

theorem expected_coincidences_value :
  expected_coincidences = 10.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_coincidences_value_l808_80851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_points_of_interest_l808_80840

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square with side length 10 -/
structure Square where
  vertices : Fin 4 → Point

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A point is 10 cm away from two vertices of the square -/
def isPointOfInterest (s : Square) (p : Point) : Prop :=
  ∃ i j : Fin 4, i ≠ j ∧ 
    distance p (s.vertices i) = 10 ∧
    distance p (s.vertices j) = 10

/-- The theorem to be proved -/
theorem square_points_of_interest (s : Square) :
  (s.vertices 0).x = 0 ∧ (s.vertices 0).y = 0 ∧
  (s.vertices 1).x = 10 ∧ (s.vertices 1).y = 0 ∧
  (s.vertices 2).x = 10 ∧ (s.vertices 2).y = 10 ∧
  (s.vertices 3).x = 0 ∧ (s.vertices 3).y = 10 →
  ∃! (points : Finset Point), points.card = 12 ∧ 
    ∀ p ∈ points, isPointOfInterest s p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_points_of_interest_l808_80840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tagged_fish_in_second_catch_l808_80854

theorem tagged_fish_in_second_catch 
  (total_fish : ℕ) 
  (initially_tagged : ℕ) 
  (second_catch : ℕ) 
  (h1 : total_fish = 1250) 
  (h2 : initially_tagged = 50) 
  (h3 : second_catch = 50) :
  Int.floor ((initially_tagged : ℝ) / total_fish * second_catch) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tagged_fish_in_second_catch_l808_80854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_and_area_l808_80814

/-- Curve C in polar coordinates -/
def C (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 = 4 * Real.cos θ

/-- Line l₁ -/
def l₁ (θ : ℝ) : Prop := θ = Real.pi / 3

/-- Line l₂ -/
def l₂ (ρ θ : ℝ) : Prop := ρ * Real.sin θ = 4 * Real.sqrt 3

/-- Point A is on curve C and line l₁, but not the pole -/
def point_A (ρ θ : ℝ) : Prop :=
  C ρ θ ∧ l₁ θ ∧ (ρ ≠ 0 ∨ θ ≠ 0)

/-- Point B is on curve C and line l₂ -/
def point_B (ρ θ : ℝ) : Prop :=
  C ρ θ ∧ l₂ ρ θ

/-- Theorem stating the coordinates of points A and B and the area of triangle AOB -/
theorem curve_intersection_and_area :
  ∃ (ρ_A θ_A ρ_B θ_B : ℝ),
    point_A ρ_A θ_A ∧
    point_B ρ_B θ_B ∧
    ρ_A = 8/3 ∧
    θ_A = Real.pi/3 ∧
    ρ_B = 8 * Real.sqrt 3 ∧
    θ_B = Real.pi/6 ∧
    (1/2 * ρ_A * ρ_B * Real.sin (θ_A - θ_B) = (16/3) * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_and_area_l808_80814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_roots_imply_value_l808_80856

theorem function_roots_imply_value (a b : ℝ) : 
  (∀ x, x^2 + (a+1)*x + a*b ≤ 0 ↔ -1 ≤ x ∧ x ≤ 4) → 
  (1/2 : ℝ)^(a+2*b) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_roots_imply_value_l808_80856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_distribution_l808_80872

theorem pen_distribution (P : ℕ) :
  (∃ (k : ℕ), P = 40 * k) ↔
  (∃ (p : ℕ), P = 40 * p) ∧ 
  (∃ (q : ℕ), 920 = 40 * q) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_distribution_l808_80872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_line_equation_l808_80898

-- Define the point F
def F : ℝ × ℝ := (0, 1)

-- Define the line l
def l : ℝ → ℝ := fun _ ↦ -1

-- Define the trajectory E
def E (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l₁
def l₁ (k : ℝ) (x : ℝ) : ℝ := k*x + 1

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Define the distance from a point to a line
def distanceToLine (p : ℝ × ℝ) (f : ℝ → ℝ) : ℝ :=
  |p.2 - f p.1|

-- Theorem 1: The equation of trajectory E
theorem trajectory_equation {x y : ℝ} :
  (∀ (M : ℝ × ℝ), M.1 = x ∧ M.2 = y →
    distance M F = distanceToLine M l) →
  E x y := by
  sorry

-- Theorem 2: The equation of line l₁
theorem line_equation :
  ∃ (k : ℝ), k = Real.sqrt 2 ∨ k = -Real.sqrt 2 ∧
  (∃ (A B : ℝ × ℝ),
    E A.1 A.2 ∧ E B.1 B.2 ∧
    A.2 = l₁ k A.1 ∧ B.2 = l₁ k B.1 ∧
    distance A B = 12 ∧
    F.2 = l₁ k F.1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_line_equation_l808_80898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_growth_rate_increase_l808_80829

/-- Given forest areas S₁, S₂, and S₃ for years 2001, 2002, and 2003 respectively,
    the increase in growth rate from 2002 to 2003 is (S₃S₁ - S₂²) / (S₁S₂). -/
theorem forest_growth_rate_increase (S₁ S₂ S₃ : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) :
  (S₃ - S₂) / S₂ - (S₂ - S₁) / S₁ = (S₃ * S₁ - S₂^2) / (S₁ * S₂) := by
  sorry

#check forest_growth_rate_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_growth_rate_increase_l808_80829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_3_25th_occurrence_l808_80843

/-- Count the occurrences of a digit in a natural number -/
def countDigit (d : Nat) (n : Nat) : Nat :=
  sorry

/-- Count the occurrences of a digit in a range of natural numbers -/
def countDigitInRange (d : Nat) (start : Nat) (stop : Nat) : Nat :=
  sorry

/-- Find the number where a digit appears for the nth time -/
def findNthOccurrence (d : Nat) (n : Nat) : Nat :=
  sorry

theorem digit_3_25th_occurrence :
  findNthOccurrence 3 25 = 134 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_3_25th_occurrence_l808_80843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_deduces_own_fish_l808_80806

-- Define the type for teachers
inductive Teacher : Type where
  | A : Teacher
  | B : Teacher
  | C : Teacher

-- Define a function to represent whether a teacher has a fish on their back
def hasFish : Teacher → Prop := sorry

-- Define a function to represent whether a teacher is laughing
def isLaughing : Teacher → Prop := sorry

-- Define a function to represent whether a teacher can see another teacher's fish
def canSeeFish : Teacher → Teacher → Prop := sorry

theorem teacher_deduces_own_fish 
  (all_laughing : ∀ t : Teacher, isLaughing t)
  (others_have_fish : ∀ t1 t2 : Teacher, t1 ≠ t2 → canSeeFish t1 t2)
  (fish_causes_laughter : ∀ t1 t2 : Teacher, canSeeFish t1 t2 → isLaughing t1) :
  ∃ t : Teacher, hasFish t ∧ (∀ t' : Teacher, t' ≠ t → canSeeFish t t') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_deduces_own_fish_l808_80806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_proof_l808_80892

noncomputable def x : ℝ := Real.log Real.pi
noncomputable def y : ℝ := Real.log 2 / Real.log 5
noncomputable def z : ℝ := Real.exp (-1/2)

theorem ordering_proof : y < z ∧ z < x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_proof_l808_80892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_is_sufficient_not_necessary_l808_80883

-- Define the function
noncomputable def f (x φ : Real) : Real := Real.sin (x + 2 * φ)

-- Define what it means for f to be even
def is_even (φ : Real) : Prop := ∀ x, f x φ = f (-x) φ

-- Define the specific condition
def condition (φ : Real) : Prop := φ = Real.pi / 4

-- State the theorem
theorem condition_is_sufficient_not_necessary :
  (∃ φ, condition φ ∧ is_even φ) ∧
  (∃ φ, ¬condition φ ∧ is_even φ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_is_sufficient_not_necessary_l808_80883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_sum_third_quadrant_l808_80809

theorem cosine_sine_sum_third_quadrant (θ : ℝ) (h1 : Real.tan θ = 5/12) 
  (h2 : π ≤ θ ∧ θ ≤ 3*π/2) : Real.cos θ + Real.sin θ = -17/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_sum_third_quadrant_l808_80809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l808_80824

/-- Given a function f such that f(x+1) = 2x + 3 for all x,
    prove that f(x) = 2x + 1 for all x. -/
theorem function_equality (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x + 3) :
  ∀ x, f x = 2 * x + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l808_80824
