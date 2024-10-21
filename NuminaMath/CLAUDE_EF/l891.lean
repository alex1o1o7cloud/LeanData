import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_average_distance_l891_89103

noncomputable def square_side_length : ℝ := 15
noncomputable def diagonal_distance : ℝ := 9.3
noncomputable def perpendicular_distance : ℝ := 3

noncomputable def lemming_position (s d p : ℝ) : ℝ × ℝ :=
  let diag_fraction := d / (s * Real.sqrt 2)
  let x := diag_fraction * s + p
  let y := diag_fraction * s
  (x, y)

noncomputable def distance_to_side (pos : ℝ × ℝ) (side : ℝ) : ℝ :=
  min pos.1 (side - pos.1)

theorem lemming_average_distance :
  let pos := lemming_position square_side_length diagonal_distance perpendicular_distance
  let dist_left := pos.1
  let dist_bottom := pos.2
  let dist_right := square_side_length - pos.1
  let dist_top := square_side_length - pos.2
  (dist_left + dist_bottom + dist_right + dist_top) / 4 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_average_distance_l891_89103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_randy_biscuits_left_l891_89187

/-- The number of biscuits Randy is left with after receiving gifts and his brother eating some -/
noncomputable def biscuits_left (initial : ℝ) (father_gift : ℝ) (mother_gift : ℝ) (brother_ate_percent : ℝ) : ℝ :=
  let total := initial + father_gift + mother_gift
  let brother_ate := (brother_ate_percent / 100) * total
  total - brother_ate

/-- Theorem stating that Randy is left with approximately 33.37 biscuits -/
theorem randy_biscuits_left :
  ∃ ε > 0, |biscuits_left 32 (2/3) 15 30 - 33.37| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_randy_biscuits_left_l891_89187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_perpendicular_lines_l891_89173

/-- Given a point P and a line l, prove the equations of parallel and perpendicular lines through P -/
theorem parallel_perpendicular_lines 
  (P : ℝ × ℝ) 
  (l : Set (ℝ × ℝ)) 
  (hl : l = {p : ℝ × ℝ | p.1 - p.2 + 4 = 0}) 
  (hP : P = (2, 1)) :
  (∃ m : Set (ℝ × ℝ), 
    m = {p : ℝ × ℝ | p.1 - p.2 - 1 = 0} ∧ 
    (∀ p : ℝ × ℝ, p ∈ m ↔ p.1 - p.2 = P.1 - P.2) ∧ 
    P ∈ m) ∧
  (∃ n : Set (ℝ × ℝ), 
    n = {p : ℝ × ℝ | p.1 + p.2 - 3 = 0} ∧ 
    (∀ p q : ℝ × ℝ, p ∈ l ∧ q ∈ l → (p.1 - q.1) * (P.1 - p.1) + (p.2 - q.2) * (P.2 - p.2) = 0) ∧ 
    P ∈ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_perpendicular_lines_l891_89173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_given_log_l891_89128

theorem cos_double_angle_given_log (x : ℝ) (h : Real.log (Real.cos x) = -1/2) :
  Real.cos (2 * x) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_given_log_l891_89128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_tiling_l891_89119

/-- Represents a square floor of size n × n -/
structure Floor (n : ℕ) where
  size : n > 0

/-- Represents types of tiles -/
inductive Tile
| square : Tile    -- 2 × 2 tile
| rectangle : Tile -- 3 × 1 tile

/-- Defines a valid tiling of the floor -/
def ValidTiling (n : ℕ) (num_square : ℕ) (num_rectangle : ℕ) : Prop :=
  num_square = num_rectangle ∧
  4 * num_square + 3 * num_rectangle = n * n

theorem floor_tiling (n : ℕ) (h : Floor n) :
  (∃ (num_square num_rectangle : ℕ), ValidTiling n num_square num_rectangle) →
  ∃ (k : ℕ), n = 7 * k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_tiling_l891_89119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l891_89117

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 6) + 1 / 2

-- State the theorem
theorem omega_value (α β : ℝ) (ω : ℝ) (h1 : f ω α = -1/2) (h2 : f ω β = 1/2) 
  (h3 : ∀ γ δ : ℝ, f ω γ = -1/2 → f ω δ = 1/2 → |γ - δ| ≥ 3*Real.pi/4) 
  (h4 : |α - β| = 3*Real.pi/4) : ω = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l891_89117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_eq_solution_pairs_l891_89195

/-- A cubic polynomial with real coefficients -/
def CubicPolynomial := ℝ → ℝ

/-- The set of points at which we evaluate the polynomials -/
def EvaluationPoints : Set ℝ := {1, 2, 3, 4}

/-- Predicate to check if a polynomial takes only values 0 or 1 at evaluation points -/
def TakesOnlyZeroOrOne (p : CubicPolynomial) : Prop :=
  ∀ x ∈ EvaluationPoints, p x = 0 ∨ p x = 1

/-- The conditions that P and Q must satisfy -/
def SatisfiesConditions (P Q : CubicPolynomial) : Prop :=
  TakesOnlyZeroOrOne P ∧ TakesOnlyZeroOrOne Q ∧
  ((P 1 = 0 ∨ P 2 = 1) → Q 1 = 1 ∧ Q 3 = 1) ∧
  ((P 2 = 0 ∨ P 4 = 0) → Q 2 = 0 ∧ Q 4 = 0) ∧
  ((P 3 = 1 ∨ P 4 = 1) → Q 1 = 0)

/-- The set of all valid polynomial pairs -/
def ValidPairs : Set (CubicPolynomial × CubicPolynomial) := {(P, Q) | SatisfiesConditions P Q}

/-- The specific polynomials mentioned in the solution -/
noncomputable def R₁ : CubicPolynomial := λ x => -1/2*x^3 + 7/2*x^2 - 7*x + 4
noncomputable def R₂ : CubicPolynomial := λ x => 1/2*x^3 - 4*x^2 + 19/2*x - 6
noncomputable def R₃ : CubicPolynomial := λ x => -1/6*x^3 + 3/2*x^2 - 13/3*x + 4
noncomputable def R₄ : CubicPolynomial := λ x => -2/3*x^3 + 5*x^2 - 34/3*x + 8
noncomputable def R₅ : CubicPolynomial := λ x => -1/2*x^3 + 4*x^2 - 19/2*x + 7
noncomputable def R₆ : CubicPolynomial := λ x => 1/3*x^3 - 5/2*x^2 + 31/6*x - 2

/-- The set of polynomial pairs given in the solution -/
def SolutionPairs : Set (CubicPolynomial × CubicPolynomial) := 
  {(R₂, R₄), (R₃, R₁), (R₃, R₃), (R₃, R₄), (R₄, R₁), (R₅, R₁), (R₆, R₄)}

/-- The main theorem stating that the set of valid pairs is equal to the solution pairs -/
theorem valid_pairs_eq_solution_pairs : ValidPairs = SolutionPairs := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_eq_solution_pairs_l891_89195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l891_89174

def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ+, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio
  (a : ℕ+ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_condition : ∀ n : ℕ+, a n * a (n + 4) = (9 : ℝ) ^ (n.val + 1)) :
  ∃ q : ℝ, (q = 3 ∨ q = -3) ∧ ∀ n : ℕ+, a (n + 1) = q * a n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l891_89174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_feathers_left_l891_89161

/-- The number of feathers Jerry has left after collecting, giving away, and selling some. -/
def feathers_left (hawk_feathers : ℕ) (eagle_ratio : ℕ) (given_away : ℕ) (sell_percentage : ℚ) : ℕ :=
  let total_feathers := hawk_feathers + hawk_feathers * eagle_ratio
  let remaining_after_gift := total_feathers - given_away
  let sold_feathers := (remaining_after_gift : ℚ) * sell_percentage
  remaining_after_gift - (sold_feathers.floor.toNat)

/-- Theorem stating that Jerry has 138 feathers left under the given conditions. -/
theorem jerry_feathers_left : 
  feathers_left 23 24 25 (3/4) = 138 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_feathers_left_l891_89161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_B_side_b_l891_89146

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  Real.sin (t.A + t.C) = 8 * (Real.sin (t.B / 2))^2 ∧
  t.a + t.c = 6 ∧
  (1/2) * t.a * t.c * Real.sin t.B = 2

-- Theorem 1
theorem cosine_B (t : Triangle) : 
  triangle_conditions t → Real.cos t.B = 15/17 := by sorry

-- Theorem 2
theorem side_b (t : Triangle) :
  triangle_conditions t → t.b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_B_side_b_l891_89146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_descent_route_length_l891_89108

/-- Proves that the length of the descent route is 21 miles given the specified hiking conditions. -/
theorem descent_route_length 
  (ascent_rate : ℝ) 
  (ascent_time : ℝ) 
  (descent_rate_multiplier : ℝ) 
  (h1 : ascent_rate = 7) 
  (h2 : ascent_time = 2) 
  (h3 : descent_rate_multiplier = 1.5) 
  (h4 : ascent_time = descent_time) : 
  descent_rate * descent_time = 21 := by
  sorry

where
  descent_time : ℝ := ascent_time
  descent_rate : ℝ := ascent_rate * descent_rate_multiplier

end NUMINAMATH_CALUDE_ERRORFEEDBACK_descent_route_length_l891_89108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l891_89106

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  16 * x^2 - 64 * x - 4 * y^2 + 8 * y + 100 = 0

/-- The distance between the vertices of the hyperbola -/
noncomputable def vertex_distance : ℝ := Real.sqrt 10

theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_equation x y → 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  ((x - 2)^2 / a^2 - (y - 1)^2 / b^2 = 1) ∧
  vertex_distance = 2 * a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l891_89106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l891_89168

/-- A rectangle type -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Property that a rectangle is divided into four identical squares -/
def Rectangle.isDividedIntoFourIdenticalSquares (r : Rectangle) : Prop :=
  r.length = 4 * r.width

/-- Given a rectangle ABCD divided into four identical squares with a perimeter of 160 cm,
    its area is 1024 square centimeters. -/
theorem rectangle_area (ABCD : Rectangle) : 
  ABCD.isDividedIntoFourIdenticalSquares →
  ABCD.perimeter = 160 →
  ABCD.area = 1024 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l891_89168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_equals_one_l891_89171

/-- Given a function f(x) = 2^|x-a| that is symmetric about x = 1, prove that a = 1 -/
theorem symmetry_implies_a_equals_one (a : ℝ) : 
  (∀ x : ℝ, (fun x => (2 : ℝ)^(|x - a|)) = (fun x => (fun y => (2 : ℝ)^(|y - a|)) (2 - x))) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_equals_one_l891_89171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequences_l891_89141

/-- Arithmetic sequence a_n with given conditions -/
def a : ℕ → ℝ := sorry

/-- Sum of first n terms of sequence b_n -/
def S : ℕ → ℝ := sorry

/-- Sequence b_n defined in terms of S_n -/
def b (n : ℕ) : ℝ := 2 - 2 * S n

theorem arithmetic_and_geometric_sequences :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) ∧  -- a_n is arithmetic
  a 5 = 14 ∧ 
  a 7 = 20 →
  (∀ n : ℕ, a n = 3 * n - 1) ∧  -- General formula for a_n
  (∀ n : ℕ, n ≥ 2 → b n / b (n - 1) = (1 : ℝ) / 3) :=  -- b_n is geometric
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequences_l891_89141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l891_89107

theorem expression_equality : 
  |(-1 : Real)| - (-3.14 + Real.pi)^(0 : Real) + 2^(-1 : Real) + (Real.cos (30 * π / 180))^2 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l891_89107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_negative_one_eighth_l891_89169

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^6 - 2) / 4

-- State the theorem
theorem inverse_f_at_negative_one_eighth :
  f⁻¹ (-1/8) = (3/2)^(1/6) :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_negative_one_eighth_l891_89169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_c_greater_b_l891_89129

-- Define the constants
noncomputable def a : ℝ := (0.3 : ℝ) ^ 3
noncomputable def b : ℝ := Real.log 0.3 / Real.log 3
noncomputable def c : ℝ := Real.log 3 / Real.log 0.3

-- State the theorem
theorem a_greater_c_greater_b : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_c_greater_b_l891_89129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_game_result_l891_89162

def NumberRemovalGame (n : ℕ) : Type :=
  {s : Finset ℕ // s ⊆ Finset.range (n + 1)}

def optimal_strategy (g : NumberRemovalGame 1024) : ℕ :=
  32

theorem optimal_game_result :
  ∀ (g : NumberRemovalGame 1024),
  ∃ (a b : ℕ),
    a ∈ g.val ∧ b ∈ g.val ∧
    ∀ (x y : ℕ), x ∈ g.val → y ∈ g.val → x ≠ y →
    (Int.natAbs (x - y) ≥ Int.natAbs (a - b)) ∧
    Int.natAbs (a - b) = optimal_strategy g :=
sorry

#check optimal_game_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_game_result_l891_89162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parametric_equation_specific_line_equation_l891_89149

/-- The parametric equation of a line passing through a point with a given inclination angle. -/
theorem line_parametric_equation (x₀ y₀ : ℝ) (θ : ℝ) :
  (λ t : ℝ ↦ (x₀ + t * Real.cos θ, y₀ + t * Real.sin θ)) =
  (λ t : ℝ ↦ (x₀ + t * Real.cos θ, y₀ + t * Real.sin θ)) :=
by sorry

/-- The parametric equation of a line passing through (1,5) with inclination angle π/3. -/
theorem specific_line_equation :
  (λ t : ℝ ↦ (1 + t * Real.cos (π/3), 5 + t * Real.sin (π/3))) =
  (λ t : ℝ ↦ (1 + (1/2) * t, 5 + (Real.sqrt 3 / 2) * t)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parametric_equation_specific_line_equation_l891_89149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_l891_89177

/-- The equation from the problem -/
noncomputable def f (x : ℝ) : ℝ := 
  4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20) - (x^2 - 12*x - 5)

/-- The largest real solution to the equation -/
def m : ℝ := 20

theorem largest_solution : 
  f m = 0 ∧ ∀ x : ℝ, f x = 0 → x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_l891_89177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l891_89127

/-- Quadrilateral type -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- Predicate for parallelogram -/
def IsParallelogram (q : Quadrilateral) : Prop := sorry

/-- Measure of an angle in a quadrilateral -/
def MeasureAngle (q : Quadrilateral) (vertex : Fin 4) : ℝ := sorry

/-- Length of a diagonal in a parallelogram -/
def DiagonalLength (q : Quadrilateral) (d : Fin 2) : ℝ := sorry

/-- Function to determine the shorter diagonal -/
def shorter_diagonal (q : Quadrilateral) : Fin 2 := sorry

/-- Function to determine the longer diagonal -/
def longer_diagonal (q : Quadrilateral) : Fin 2 := sorry

/-- Length of the perpendicular from diagonal intersection to longer side -/
def PerpendicularLength (q : Quadrilateral) : ℝ := sorry

/-- Length of a side in a parallelogram -/
def SideLength (q : Quadrilateral) (s : Fin 4) : ℝ := sorry

/-- Function to determine the shorter side -/
def shorter_side (q : Quadrilateral) : Fin 4 := sorry

/-- Function to determine the longer side -/
def longer_side (q : Quadrilateral) : Fin 4 := sorry

theorem parallelogram_properties (ABCD : Quadrilateral) (h1 : IsParallelogram ABCD)
  (h2 : MeasureAngle ABCD 0 = Real.pi / 3)
  (h3 : DiagonalLength ABCD (shorter_diagonal ABCD) = 2 * Real.sqrt 31)
  (h4 : PerpendicularLength ABCD = Real.sqrt 75 / 2) :
  (SideLength ABCD (shorter_side ABCD) = 10 ∧
   SideLength ABCD (longer_side ABCD) = 14 ∧
   DiagonalLength ABCD (longer_diagonal ABCD) = 2 * Real.sqrt 91) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l891_89127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l891_89150

/-- The function f(x) = x - 3/x -/
noncomputable def f (x : ℝ) : ℝ := x - 3 / x

/-- The derivative of f(x) -/
noncomputable def f_derivative (x : ℝ) : ℝ := 1 + 3 / (x^2)

/-- The y-intercept of the tangent line at point (x₀, f(x₀)) -/
noncomputable def y_intercept (x₀ : ℝ) : ℝ := -6 / x₀

/-- The x-coordinate of the intersection point of the tangent line with y = x -/
noncomputable def x_intersection (x₀ : ℝ) : ℝ := 2 * x₀

/-- The area of the triangle formed by the tangent line, x = 0, and y = x -/
noncomputable def triangle_area (x₀ : ℝ) : ℝ := (1/2) * x_intersection x₀ * |y_intercept x₀|

theorem constant_triangle_area (x₀ : ℝ) (h : x₀ ≠ 0) : 
  triangle_area x₀ = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l891_89150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_COE_is_pi_R_squared_over_six_l891_89114

noncomputable section

open Real

/-- The area of the figure COE in a circle divided into six equal parts -/
def area_COE (R : ℝ) : ℝ := (π * R^2) / 6

/-- Theorem stating that the area of COE is π*R^2/6 -/
theorem area_COE_is_pi_R_squared_over_six (R : ℝ) (h : R > 0) :
  area_COE R = (π * R^2) / 6 := by
  -- Unfold the definition of area_COE
  unfold area_COE
  
  -- The proof would involve several steps:
  -- 1. Show that the area of triangle AOE is (R^2 * √3) / 4
  -- 2. Show that the area of sector FOME is (π * R^2) / 6
  -- 3. Show that the area of segment OME is (π * R^2) / 6 - (R^2 * √3) / 4
  -- 4. Combine these results to show that the area of COE is (π * R^2) / 6
  
  -- For now, we'll use sorry to skip the detailed proof
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_COE_is_pi_R_squared_over_six_l891_89114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l891_89176

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def frac (x : ℝ) : ℝ := x - floor x

noncomputable def a : ℕ → ℝ
| 0 => Real.sqrt 3
| n + 1 => floor (a n) + 1 / frac (a n)

theorem a_2017_value : a 2016 = 3024 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l891_89176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l891_89105

/-- The value of n for which the expansion of (∛(x²) + 3x²)ⁿ satisfies the given condition -/
def n : ℕ := 5

/-- The sum of coefficients of all terms in the expansion -/
def sum_of_coefficients : ℕ := 4^n

/-- The sum of binomial coefficients in the expansion -/
def sum_of_binomial_coefficients : ℕ := 2^n

/-- The condition given in the problem -/
axiom condition : sum_of_coefficients = sum_of_binomial_coefficients + 992

/-- The terms with the largest binomial coefficient -/
def largest_binomial_coeff_terms : ℕ × ℕ := (90, 270)

/-- The term with the largest coefficient -/
def largest_coeff_term : ℕ := 405

theorem expansion_properties :
  (sum_of_coefficients = sum_of_binomial_coefficients + 992) →
  (largest_binomial_coeff_terms = (90, 270)) ∧
  (largest_coeff_term = 405) :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l891_89105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l891_89197

theorem max_value_expression (a b c d : ℝ) 
  (ha : a ∈ Set.Icc (-12.5 : ℝ) 12.5)
  (hb : b ∈ Set.Icc (-12.5 : ℝ) 12.5)
  (hc : c ∈ Set.Icc (-12.5 : ℝ) 12.5)
  (hd : d ∈ Set.Icc (-12.5 : ℝ) 12.5) :
  (∀ x y z w : ℝ, x ∈ Set.Icc (-12.5 : ℝ) 12.5 → 
              y ∈ Set.Icc (-12.5 : ℝ) 12.5 → 
              z ∈ Set.Icc (-12.5 : ℝ) 12.5 → 
              w ∈ Set.Icc (-12.5 : ℝ) 12.5 → 
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 650) ∧ 
  (∃ x y z w : ℝ, x ∈ Set.Icc (-12.5 : ℝ) 12.5 ∧ 
              y ∈ Set.Icc (-12.5 : ℝ) 12.5 ∧ 
              z ∈ Set.Icc (-12.5 : ℝ) 12.5 ∧ 
              w ∈ Set.Icc (-12.5 : ℝ) 12.5 ∧ 
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 650) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l891_89197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l891_89198

theorem tan_alpha_value (α : Real) (h1 : Real.sin α = 4/5) (h2 : π/2 < α ∧ α < π) : Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l891_89198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l891_89104

theorem sin_double_angle_special (α : Real) :
  π/2 < α ∧ α < π →
  Real.sin (α + π/6) = 1/3 →
  Real.sin (2*α + π/3) = -4*Real.sqrt 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l891_89104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_max_value_expression_angle_at_max_value_l891_89136

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c ∧
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi

-- Define the arithmetic sequence property
def isArithmeticSequence (t : Triangle) : Prop :=
  t.B = (t.A + t.C) / 2

-- Define the given conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.b = 7 ∧ t.a + t.c = 13

-- State the theorems to be proved
theorem area_of_triangle (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : isArithmeticSequence t) 
  (h3 : satisfiesConditions t) : 
  (1/2) * t.a * t.c * Real.sin t.B = 10 * Real.sqrt 3 := by sorry

theorem max_value_expression (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : isArithmeticSequence t) : 
  (∀ A, 0 < A → A < 2*Real.pi/3 → Real.sqrt 3 * Real.sin A + Real.sin (t.C - Real.pi/6) ≤ 2) := by sorry

theorem angle_at_max_value (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : isArithmeticSequence t) : 
  (∃ A, A = Real.pi/3 ∧ Real.sqrt 3 * Real.sin A + Real.sin (t.C - Real.pi/6) = 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_max_value_expression_angle_at_max_value_l891_89136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_positive_power_of_two_l891_89147

theorem negation_of_universal_positive_power_of_two (p : Prop) :
  (p ↔ ∀ x : ℝ, (2 : ℝ)^x > 0) →
  (¬p ↔ ∃ x : ℝ, (2 : ℝ)^x ≤ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_positive_power_of_two_l891_89147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_f_2004_l891_89126

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_triple (x : ℝ) (h : 0 < x) : f (3 * x) = 3 * f x

axiom f_def (x : ℝ) (h : 1 ≤ x ∧ x ≤ 4) : f x = 1 - |x - 3|

-- Define the theorem
theorem smallest_x_equals_f_2004 :
  ∃ (x : ℝ), x > 0 ∧ f x = f 2004 ∧ ∀ (y : ℝ), y > 0 ∧ f y = f 2004 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_f_2004_l891_89126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_plus_pi_4_l891_89110

noncomputable def angle_alpha (x y : ℝ) : ℝ := Real.arctan (y / x)

theorem tan_2alpha_plus_pi_4 (x y : ℝ) (h : x = 2 ∧ y = 3) :
  Real.tan (2 * angle_alpha x y + π/4) = -7/17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_plus_pi_4_l891_89110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bryce_received_16_raisins_l891_89142

-- Define the number of raisins Bryce and Carter received
def bryce_raisins : ℕ := sorry
def carter_raisins : ℕ := sorry

-- Condition 1: Bryce received 8 more raisins than Carter
axiom bryce_more_than_carter : bryce_raisins = carter_raisins + 8

-- Condition 2: Carter received half the number of raisins Bryce received
axiom carter_half_of_bryce : carter_raisins = bryce_raisins / 2

-- Theorem: Bryce received 16 raisins
theorem bryce_received_16_raisins : bryce_raisins = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bryce_received_16_raisins_l891_89142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l891_89189

noncomputable def parabola (x : ℝ) : ℝ := x^2 - 6*x + 12

noncomputable def line (x : ℝ) : ℝ := 2*x - 5

noncomputable def distance_to_line (a : ℝ) : ℝ :=
  |2*a - (parabola a) - 5| / Real.sqrt 5

theorem shortest_distance :
  ∃ (min_dist : ℝ), min_dist = 1 / Real.sqrt 5 ∧
  ∀ (a : ℝ), distance_to_line a ≥ min_dist := by
  sorry

#check shortest_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l891_89189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_comparison_l891_89118

theorem revenue_comparison (base_revenue : ℝ) (projected_increase_percent : ℝ) 
  (actual_decrease_percent : ℝ) :
  projected_increase_percent = 25 →
  actual_decrease_percent = 25 →
  (base_revenue * (1 - actual_decrease_percent / 100)) / 
  (base_revenue * (1 + projected_increase_percent / 100)) * 100 = 60 := by
  intro h1 h2
  simp [h1, h2]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_comparison_l891_89118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_calories_eaten_l891_89153

-- Define the constants
noncomputable def carrot_weight : ℝ := 1
noncomputable def carrot_calories_per_pound : ℝ := 51
noncomputable def broccoli_weight_ratio : ℝ := 2
noncomputable def broccoli_calories_ratio : ℝ := 1/3

-- Define the theorem
theorem total_calories_eaten : 
  let broccoli_weight := carrot_weight * broccoli_weight_ratio
  let broccoli_calories_per_pound := carrot_calories_per_pound * broccoli_calories_ratio
  let carrot_calories := carrot_weight * carrot_calories_per_pound
  let broccoli_calories := broccoli_weight * broccoli_calories_per_pound
  carrot_calories + broccoli_calories = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_calories_eaten_l891_89153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_225_square_equals_two_triangles_l891_89102

/-- A square with sides of 15 units -/
structure Square where
  side_length : ℝ
  is_fifteen : side_length = 15

/-- The area of a square -/
noncomputable def square_area (s : Square) : ℝ := s.side_length * s.side_length

/-- A right-angled triangle formed by dividing the square along its diagonal -/
structure RightTriangle (s : Square) where
  base : ℝ
  height : ℝ
  is_half_square : base = s.side_length ∧ height = s.side_length

/-- The area of a right-angled triangle -/
noncomputable def triangle_area {s : Square} (t : RightTriangle s) : ℝ := (t.base * t.height) / 2

/-- Theorem: The area of the square is 225 square units -/
theorem square_area_is_225 (s : Square) :
  square_area s = 225 := by
  sorry

/-- Theorem: The area of the square equals the area of two right-angled triangles -/
theorem square_equals_two_triangles (s : Square) (t : RightTriangle s) :
  square_area s = 2 * triangle_area t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_225_square_equals_two_triangles_l891_89102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l891_89191

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (3*θ) = -23/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l891_89191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_problem_l891_89133

-- Define the ellipse C₁
def C₁ (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def l (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the circle C₂
def C₂ (x y r : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = r^2

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem ellipse_and_circle_problem 
  (a b : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (h₃ : a^2 - b^2 = 4) -- Derived from eccentricity e = 1/2
  (h₄ : ∃ x₀ y₀, l x₀ y₀ ∧ x₀ = -2 ∧ y₀ = 0) -- Left focus on line l
  (h₅ : ∃ r, r > 0 ∧ ∀ x y, l x y → C₂ x y r → distance x y 3 3 = 2) :
  (∀ x y, C₁ x y a b ↔ C₁ x y 4 (2 * Real.sqrt 3)) ∧ 
  (¬∃ x y, C₂ x y 2 ∧ distance x y (-2) 0 = (a/b) * distance x y 2 0) := 
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_problem_l891_89133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_square_on_cubic_curve_l891_89155

/-- A cubic curve defined by y = x^3 + ax^2 + bx + c -/
def cubic_curve (a b c : ℝ) : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + b*x + c

/-- The condition that a point (x, y) lies on the cubic curve -/
def on_curve (a b c : ℝ) (x y : ℝ) : Prop :=
  y = cubic_curve a b c x

/-- The condition that four points form a square -/
def is_square (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  let d12 := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2
  let d23 := (p2.1 - p3.1)^2 + (p2.2 - p3.2)^2
  let d34 := (p3.1 - p4.1)^2 + (p3.2 - p4.2)^2
  let d41 := (p4.1 - p1.1)^2 + (p4.2 - p1.2)^2
  let d13 := (p1.1 - p3.1)^2 + (p1.2 - p3.2)^2
  let d24 := (p2.1 - p4.1)^2 + (p2.2 - p4.2)^2
  d12 = d23 ∧ d23 = d34 ∧ d34 = d41 ∧ d13 = d24

theorem unique_square_on_cubic_curve :
  ∀ a b c : ℝ,
  (∃! p1 p2 p3 p4 : ℝ × ℝ,
    on_curve a b c p1.1 p1.2 ∧
    on_curve a b c p2.1 p2.2 ∧
    on_curve a b c p3.1 p3.2 ∧
    on_curve a b c p4.1 p4.2 ∧
    is_square p1 p2 p3 p4) →
  ∃ p1 p2 p3 p4 : ℝ × ℝ,
    on_curve a b c p1.1 p1.2 ∧
    on_curve a b c p2.1 p2.2 ∧
    on_curve a b c p3.1 p3.2 ∧
    on_curve a b c p4.1 p4.2 ∧
    is_square p1 p2 p3 p4 ∧
    (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 = (72 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_square_on_cubic_curve_l891_89155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l891_89151

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

noncomputable def common_ratio (b : ℕ → ℝ) : ℝ :=
  b 2 / b 1

theorem geometric_sequence_common_ratio
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a_increasing : a 1 < a 2)
  (h_b_increasing : b 1 < b 2)
  (h_relation : ∀ i : ℕ, i ∈ ({1, 2, 3} : Finset ℕ) → b i = (a i) ^ 2) :
  common_ratio b = 3 + 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l891_89151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_count_correct_l891_89115

def pet_count (dog_cat_ratio : ℚ) (dogs_given : ℕ) (dogs_remaining : ℕ) : ℕ :=
  let initial_dogs := dogs_remaining + dogs_given
  let cat_count := (initial_dogs * dog_cat_ratio.den) / dog_cat_ratio.num
  dogs_remaining + cat_count.toNat

#eval pet_count (10/17) 10 60  -- Should evaluate to 179

theorem pet_count_correct :
  pet_count (10/17) 10 60 = 179 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_count_correct_l891_89115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_equidistant_circle_radius_l891_89181

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a kite with four vertices -/
structure Kite where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The given kite in the problem -/
def problemKite : Kite :=
  { A := { x := 0, y := 0 },
    B := { x := 1, y := 2 },
    C := { x := 4, y := 0 },
    D := { x := 1, y := -2 } }

/-- Theorem: The radius of the largest circle equidistant from all vertices of the given kite -/
theorem largest_equidistant_circle_radius (k : Kite) (h : k = problemKite) :
  ∃ (r : ℝ), r = (Real.sqrt 65 + Real.sqrt 97) / 8 ∧
    ∀ (center : Point),
      (distance center k.A = r ∧
       distance center k.B = r ∧
       distance center k.C = r ∧
       distance center k.D = r) →
      ∀ (r' : ℝ),
        (∀ (center' : Point),
          distance center' k.A = r' ∧
          distance center' k.B = r' ∧
          distance center' k.C = r' ∧
          distance center' k.D = r') →
        r' ≤ r :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_equidistant_circle_radius_l891_89181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_range_l891_89179

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The intersection of two lines -/
noncomputable def intersection (l1 l2 : Line) : Point := sorry

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (p : Point) (l : Line) : ℝ := sorry

/-- The set of all intersection points as m varies -/
def intersection_set (m : ℝ) : Set Point := sorry

theorem intersection_distance_range :
  ∀ m : ℝ,
  let l1 : Line := { a := m, b := -1, c := -3*m+1 }
  let l2 : Line := { a := 1, b := m, c := -3*m-1 }
  let l3 : Line := { a := 1, b := 1, c := 0 }
  let P := intersection l1 l2
  let d := distance_point_to_line P l3
  d ≥ Real.sqrt 2 ∧ d ≤ 3 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_range_l891_89179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_point_l891_89100

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a

-- Define the derivative of f(x)
noncomputable def f_deriv (x : ℝ) : ℝ := Real.log x + 1

-- Theorem statement
theorem tangent_line_passes_through_point (a : ℝ) :
  (f a 1 + f_deriv 1 * (2 - 1) = 2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_point_l891_89100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homomorphism_injective_iff_trivial_kernel_l891_89199

variable {G H : Type*} [Group G] [Group H]

def is_group_homomorphism (φ : G → H) : Prop :=
  ∀ x y : G, φ (x * y) = φ x * φ y

def kernel (φ : G → H) : Set G :=
  {x : G | φ x = 1}

theorem homomorphism_injective_iff_trivial_kernel
  (φ : G → H) (hφ : is_group_homomorphism φ) :
  Function.Injective φ ↔ kernel φ = {1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homomorphism_injective_iff_trivial_kernel_l891_89199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_l891_89101

theorem unique_triple : ∃! (a b c : ℕ),
  a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ 
  (Real.logb a b = c^3) ∧
  (a + b + c = 300) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_l891_89101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_poles_count_l891_89156

/-- Calculates the number of poles needed for a side of a fence -/
def poles_for_side (length : ℕ) (spacing : ℕ) : ℕ :=
  (length / spacing).succ + 1

/-- Represents a quadrilateral plot of land -/
structure QuadrilateralPlot where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  side4 : ℕ

/-- Calculates the total number of poles needed for a quadrilateral plot -/
def total_poles (plot : QuadrilateralPlot) (spacing1 spacing2 : ℕ) : ℕ :=
  poles_for_side plot.side1 spacing1 +
  poles_for_side plot.side2 spacing1 +
  poles_for_side plot.side3 spacing2 +
  poles_for_side plot.side4 spacing2 - 3

theorem fence_poles_count :
  let plot := QuadrilateralPlot.mk 100 80 90 70
  total_poles plot 15 10 = 33 := by
  sorry

#eval total_poles (QuadrilateralPlot.mk 100 80 90 70) 15 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_poles_count_l891_89156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_location_l891_89182

/-- A circular segment with area S, smaller than a semicircle, is spanned by a chord of length h. -/
structure CircularSegment where
  S : ℝ
  h : ℝ
  is_smaller_than_semicircle : S > 0
  chord_positive : h > 0

/-- The distance from the center of the circle to the centroid of the segment -/
noncomputable def centroid_distance (segment : CircularSegment) : ℝ :=
  (2 * segment.h^3) / (3 * segment.S)

/-- Theorem stating the location of the centroid of a circular segment -/
theorem centroid_location (segment : CircularSegment) :
  ∃ (OZ : ℝ), OZ = centroid_distance segment := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_location_l891_89182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l891_89144

-- Define the distance function between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the function g
noncomputable def g (x y z : ℝ) : ℝ :=
  max (distance x y z z) (max (distance x y (6-z) (z-6)) (distance x y 0 0))

-- State the theorem
theorem min_value_of_g :
  ∀ (z : ℝ), z ≠ 0 → z ≠ 6 →
  (∃ (m : ℝ), m = 3 ∧ ∀ (x y : ℝ), g x y z ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l891_89144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l891_89157

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x + 6 else x + 6

-- Define the solution set
def solution_set : Set ℝ := Set.Ioo (-3) 1 ∪ Set.Ioi 3

-- Theorem statement
theorem f_inequality_solution_set :
  {x : ℝ | f x > f 1} = solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l891_89157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_circle_from_three_points_l891_89160

/-- A structure representing a point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A predicate to check if three points are collinear --/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- A structure representing a circle --/
structure Circle where
  center : Point
  radius : ℝ

/-- A function to check if a point is on a circle --/
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- A theorem stating that exactly one circle can be determined by three non-collinear points --/
theorem unique_circle_from_three_points (p1 p2 p3 : Point) 
  (h : ¬ collinear p1 p2 p3) : 
  ∃! c : Circle, on_circle p1 c ∧ on_circle p2 c ∧ on_circle p3 c :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_circle_from_three_points_l891_89160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_antifreeze_percentage_l891_89186

/-- Proves that the initial percentage of antifreeze in a radiator mixture is approximately 10%,
    given specific conditions about fluid replacement and desired final concentration. -/
theorem initial_antifreeze_percentage
  (total_volume : ℝ)
  (drain_volume : ℝ)
  (replacement_percentage : ℝ)
  (final_percentage : ℝ)
  (h1 : total_volume = 4)
  (h2 : drain_volume = 2.2857)
  (h3 : replacement_percentage = 0.8)
  (h4 : final_percentage = 0.5)
  : ∃ (initial_percentage : ℝ),
    (initial_percentage * (total_volume - drain_volume) +
     replacement_percentage * drain_volume) / total_volume = final_percentage ∧
    abs (initial_percentage - 0.1) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_antifreeze_percentage_l891_89186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sine_sum_l891_89113

theorem max_value_of_sine_sum (x y z : Real) 
  (hx : 0 ≤ x ∧ x ≤ Real.pi / 2)
  (hy : 0 ≤ y ∧ y ≤ Real.pi / 2)
  (hz : 0 ≤ z ∧ z ≤ Real.pi / 2) :
  (Real.sin (x - y) + Real.sin (y - z) + Real.sin (z - x)) ≤ Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sine_sum_l891_89113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_max_value_condition_range_condition_l891_89122

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) ^ (a * x^2 - 4*x + 3)

-- Part 1
theorem monotonicity_intervals :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₁ < -2 ∧ x₂ < -2 → f (-1) x₁ > f (-1) x₂) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₁ > -2 ∧ x₂ > -2 → f (-1) x₁ < f (-1) x₂) :=
by sorry

-- Part 2
theorem max_value_condition (a : ℝ) :
  (∃ x, f a x = 3) ∧ (∀ x, f a x ≤ 3) → a = 1 :=
by sorry

-- Part 3
theorem range_condition (a : ℝ) :
  (∀ y, y > 0 → ∃ x, f a x = y) ∧ (∀ x, f a x > 0) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_max_value_condition_range_condition_l891_89122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_study_time_difference_l891_89190

/-- Represents the difference in study time between Tom and Tim for each day of the week -/
def studyTimeDifferences : List Int := [15, -5, 25, 35, -15, -20, 5]

/-- The number of days in a week -/
def daysInWeek : Nat := 7

/-- Calculates the average difference in study time -/
def averageDifference : ℚ :=
  (studyTimeDifferences.sum : ℚ) / daysInWeek

theorem average_study_time_difference :
  Int.floor averageDifference = 5 ∧ Int.ceil averageDifference = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_study_time_difference_l891_89190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l891_89111

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance from the center to a focus of the hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

/-- The theorem stating the eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) 
    (h_distance : (h.b / Real.sqrt 7) * Real.sqrt (h.b^2 + (focal_distance h)^2) = 
                  h.b * (focal_distance h - h.a)) : 
    eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l891_89111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l891_89194

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (-2 + 2 * Real.cos θ, 2 * Real.sin θ)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.sqrt 2 / Real.sin (θ + Real.pi / 4)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), min_dist = 3 * Real.sqrt 2 - 2 ∧
  ∀ (θ₁ θ₂ : ℝ), distance (C₁ θ₁) (C₂ θ₂) ≥ min_dist := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l891_89194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_function_l891_89185

theorem same_function (n : ℕ) (x : ℝ) : 
  (x^(2*n+1))^(1/(2*n+1 : ℝ)) = (x^(1/(2*n-1 : ℝ)))^(2*n-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_function_l891_89185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_l891_89188

theorem divisible_by_four (n : ℕ) : ∃ k : ℕ, (2*n - 1)^(2*n + 1) + (2*n + 1)^(2*n - 1) = 4*k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_l891_89188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_extreme_points_iff_a_range_l891_89183

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * a * x^2 + x - 1

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x + 1

/-- Theorem stating that f(x) has two extreme points iff a ∈ (-∞, -2) ∪ (2, +∞) -/
theorem two_extreme_points_iff_a_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f_deriv a x = 0 ∧ f_deriv a y = 0) ↔ 
  (a < -2 ∨ a > 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_extreme_points_iff_a_range_l891_89183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_reduction_l891_89154

/-- Represents a parallelogram with side lengths a and b -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  a_positive : 0 < a
  b_positive : 0 < b

/-- Represents the process of cutting off a rhombus from a parallelogram -/
noncomputable def cut_rhombus (p : Parallelogram) : Parallelogram where
  a := p.a / 2
  b := p.b / 2
  a_positive := by
    apply half_pos
    exact p.a_positive
  b_positive := by
    apply half_pos
    exact p.b_positive

/-- The set of possible original parallelogram side lengths -/
def possible_sides : Set (ℝ × ℝ) :=
  {(1,5), (4,5), (3,7), (4,7), (3,8), (5,8), (5,7), (2,7)}

/-- The theorem statement -/
theorem parallelogram_reduction (p : Parallelogram) :
  (cut_rhombus (cut_rhombus (cut_rhombus p))).a = 1 ∧
  (cut_rhombus (cut_rhombus (cut_rhombus p))).b = 2 →
  (p.a, p.b) ∈ possible_sides := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_reduction_l891_89154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_continuous_l891_89109

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x * Real.sin (1 / x) else 0

-- State the theorem
theorem f_continuous : Continuous f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_continuous_l891_89109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_properties_l891_89124

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x < f y

theorem log_function_properties (a b : ℝ) :
  (is_even_function (fun x ↦ Real.log (|x - b|) / Real.log a)) ∧
  (is_increasing_on (fun x ↦ Real.log (|x - b|) / Real.log a) (Set.Iio 0)) →
  0 < a ∧ a < 1 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_properties_l891_89124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_slope_values_l891_89125

-- Define points A and B
def A : ℝ × ℝ := (-3, 2)
def B : ℝ × ℝ := (1, 3)

-- Define the line l
def line_l (a x y : ℝ) : Prop := (a - 1) * x + (a + 1) * y + 2 * a - 2 = 0

-- Define the slope of a line passing through two points
noncomputable def line_slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the condition that line l intersects line segment AB
def intersects_AB (a : ℝ) : Prop :=
  ∃ (x y : ℝ), line_l a x y ∧ 
    ((x - A.1) / (B.1 - A.1) = (y - A.2) / (B.2 - A.2)) ∧
    0 ≤ (x - A.1) / (B.1 - A.1) ∧ (x - A.1) / (B.1 - A.1) ≤ 1

-- Theorem statement
theorem line_l_slope_values :
  ∀ (a : ℝ), intersects_AB a →
    (∃ (k : ℝ), (k = 1 ∨ k = 2) ∧
      ∀ (x y : ℝ), line_l a x y → 
        (y - A.2) / (x - A.1) = k ∨ (y - B.2) / (x - B.1) = k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_slope_values_l891_89125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_current_speed_ratio_l891_89134

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ → ℝ := sorry

/-- Represents the speed of the water current -/
def current_speed : ℝ → ℝ := sorry

/-- Represents the distance traveled -/
def distance : ℝ → ℝ := sorry

theorem boat_current_speed_ratio 
  (x : ℝ)
  (t_upstream : ℝ) 
  (t_downstream : ℝ) 
  (h_upstream : t_upstream = 6) 
  (h_downstream : t_downstream = 10) 
  (h_distance_eq : distance (boat_speed x - current_speed x) * t_upstream = 
                   distance (boat_speed x + current_speed x) * t_downstream) :
  boat_speed x = 4 * current_speed x :=
by
  sorry

#check boat_current_speed_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_current_speed_ratio_l891_89134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_y_4_is_integer_four_is_smallest_l891_89163

noncomputable def y : ℕ → ℝ
  | 0 => 0  -- Adding a case for 0 to address the missing case error
  | 1 => 4^(1/4)
  | 2 => (4^(1/4))^(4^(1/4))
  | n+3 => (y (n+2))^(4^(1/4))

theorem smallest_integer_y (n : ℕ) : n < 4 → ¬(∃ m : ℤ, y n = m) := by
  sorry

theorem y_4_is_integer : ∃ m : ℤ, y 4 = m := by
  sorry

theorem four_is_smallest (n : ℕ) : n ≥ 4 → (∃ m : ℤ, y n = m) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_y_4_is_integer_four_is_smallest_l891_89163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_exterior_angle_l891_89193

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- The exterior angle of two adjacent sides of a polygon -/
def exteriorAngle (p q r : ℝ × ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem square_triangle_exterior_angle
  (square : RegularPolygon 4)
  (triangle : RegularPolygon 3)
  (h_coplanar : sorry)  -- Assumption that square and triangle are coplanar
  (h_opposite : sorry)  -- Assumption that square and triangle are on opposite sides of a common edge
  : exteriorAngle (square.vertices 2) (square.vertices 3) (triangle.vertices 2) = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_exterior_angle_l891_89193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_diameter_is_four_l891_89123

/-- Represents a circular well with given volume and depth -/
structure CircularWell where
  volume : ℝ
  depth : ℝ

/-- Calculates the diameter of a circular well -/
noncomputable def wellDiameter (well : CircularWell) : ℝ :=
  2 * Real.sqrt (well.volume / (Real.pi * well.depth))

/-- Theorem stating that a well with the given specifications has a diameter of 4 meters -/
theorem well_diameter_is_four :
  let well : CircularWell := { volume := 301.59289474462014, depth := 24 }
  wellDiameter well = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_diameter_is_four_l891_89123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l891_89180

/-- An isosceles trapezoid with the given measurements -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- The area of the isosceles trapezoid -/
noncomputable def trapezoid_area (t : IsoscelesTrapezoid) : ℝ := 
  (15000 - 2000 * Real.sqrt 11) / 9

/-- Theorem stating that the area of the specific isosceles trapezoid is as calculated -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := { 
    leg_length := 40,
    diagonal_length := 50,
    longer_base := 60
  }
  trapezoid_area t = (15000 - 2000 * Real.sqrt 11) / 9 := by
  sorry

#check specific_trapezoid_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l891_89180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l891_89158

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The length of segment AB -/
def AB : ℝ := 10

/-- The length of segment AC, where C is the golden section point on AB -/
noncomputable def AC : ℝ := AB * (Real.sqrt 5 - 1) / 2

/-- Theorem stating that AC is approximately 6.18 units -/
theorem golden_section_length : 
  ∀ ε > 0, |AC - 6.18| < ε := by
  sorry

#eval AB -- This will work
-- #eval AC -- This won't work due to being noncomputable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l891_89158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_15_seconds_l891_89143

/-- The time taken for a train to cross a pole -/
noncomputable def train_crossing_time (train_speed_kmh : ℝ) (train_length_m : ℝ) : ℝ :=
  train_length_m / (train_speed_kmh * 1000 / 3600)

/-- Theorem stating that the time taken for the train to cross the pole is approximately 15 seconds -/
theorem train_crossing_time_approx_15_seconds 
  (train_speed_kmh : ℝ) 
  (train_length_m : ℝ) 
  (h1 : train_speed_kmh = 60) 
  (h2 : train_length_m = 250.00000000000003) : 
  ∃ ε > 0, |train_crossing_time train_speed_kmh train_length_m - 15| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_15_seconds_l891_89143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_sum_l891_89167

theorem common_tangent_sum (a b c : ℕ+) : 
  (∃ (m : ℚ), 
    (∀ x y : ℝ, y = x^2 + (102:ℝ)/100 → (a : ℝ)*x + (b : ℝ)*y = (c : ℝ) → 
      (∃ x₀ : ℝ, ∀ x' : ℝ, x' ≠ x₀ → (a : ℝ)*x' + (b : ℝ)*(x'^2 + (102:ℝ)/100) ≠ (c : ℝ))) ∧ 
    (∀ x y : ℝ, x = y^2 + (49:ℝ)/4 → (a : ℝ)*x + (b : ℝ)*y = (c : ℝ) → 
      (∃ y₀ : ℝ, ∀ y' : ℝ, y' ≠ y₀ → (a : ℝ)*((y'^2 + (49:ℝ)/4) : ℝ) + (b : ℝ)*y' ≠ (c : ℝ))) ∧
    (a : ℚ) / (b : ℚ) = m) ∧ 
  Nat.gcd a.val (Nat.gcd b.val c.val) = 1 →
  a.val + b.val + c.val = 11 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_sum_l891_89167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_r_l891_89170

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the circle
def circle_eq (x y r : ℝ) : Prop := x^2 + (y-5)^2 = r^2

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the condition for a line to intersect the parabola at two points
def intersects_parabola (l : Line) : Prop := 
  ∃ (x1 y1 x2 y2 : ℝ), x1 ≠ x2 ∧ 
    parabola x1 y1 ∧ parabola x2 y2 ∧
    y1 = l.slope * x1 + l.intercept ∧
    y2 = l.slope * x2 + l.intercept

-- Define the condition for a line to be tangent to the circle
def tangent_to_circle (l : Line) (r : ℝ) : Prop :=
  ∃ (x0 y0 : ℝ), circle_eq x0 y0 r ∧
    y0 = l.slope * x0 + l.intercept ∧
    (l.slope * x0 + l.intercept - 5) * x0 + l.slope * ((x0)^2 + (y0-5)^2) = 0

-- Define the condition for M to be the midpoint of AB
def midpoint_condition (l : Line) : Prop :=
  ∃ (x1 y1 x2 y2 x0 y0 : ℝ), 
    parabola x1 y1 ∧ parabola x2 y2 ∧
    y1 = l.slope * x1 + l.intercept ∧
    y2 = l.slope * x2 + l.intercept ∧
    x0 = (x1 + x2) / 2 ∧
    y0 = (y1 + y2) / 2

-- Main theorem
theorem range_of_r (r : ℝ) : 
  (∃! (l1 l2 l3 l4 : Line), 
    (∀ i, i = l1 ∨ i = l2 ∨ i = l3 ∨ i = l4 → 
      intersects_parabola i ∧ 
      tangent_to_circle i r ∧ 
      midpoint_condition i) ∧
    (∀ j k, (j = l1 ∨ j = l2 ∨ j = l3 ∨ j = l4) → 
            (k = l1 ∨ k = l2 ∨ k = l3 ∨ k = l4) → 
            j ≠ k → j.slope ≠ k.slope ∨ j.intercept ≠ k.intercept)) →
  2 < r ∧ r < 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_r_l891_89170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_orange_relation_orange_pear_relation_pears_for_24_apples_l891_89165

/-- The cost of an apple -/
noncomputable def apple_cost : ℚ := 1

/-- The cost of an orange -/
noncomputable def orange_cost : ℚ := 2

/-- The cost of a pear -/
noncomputable def pear_cost : ℚ := 6/5

/-- Twelve apples cost the same as six oranges -/
theorem apple_orange_relation : 12 * apple_cost = 6 * orange_cost := by sorry

/-- Three oranges cost the same as five pears -/
theorem orange_pear_relation : 3 * orange_cost = 5 * pear_cost := by sorry

/-- The number of pears equivalent in cost to 24 apples is 20 -/
theorem pears_for_24_apples : (24 * apple_cost) / pear_cost = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_orange_relation_orange_pear_relation_pears_for_24_apples_l891_89165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_arc_length_l891_89139

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := (1/2) * Real.cos t - (1/4) * Real.cos (2*t)
noncomputable def y (t : ℝ) : ℝ := (1/2) * Real.sin t - (1/4) * Real.sin (2*t)

-- Define the arc length function
noncomputable def arcLength (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)

-- State the theorem
theorem curve_arc_length :
  arcLength (π/2) (2*π/3) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_arc_length_l891_89139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clea_escalator_theorem_l891_89164

/-- Represents the escalator problem with Clea's walking times. -/
structure EscalatorProblem where
  /-- Clea's walking speed (units per second) -/
  walking_speed : ℝ
  /-- Total distance of the escalator (units) -/
  escalator_distance : ℝ
  /-- Time to walk down stationary escalator (seconds) -/
  stationary_time : ℝ
  /-- Time to walk down moving escalator (seconds) -/
  moving_time : ℝ

/-- The solution to the escalator problem -/
noncomputable def solve_escalator_problem (p : EscalatorProblem) : ℝ :=
  p.stationary_time * p.moving_time / (p.stationary_time - p.moving_time)

/-- Theorem stating the correct solution for Clea's escalator problem -/
theorem clea_escalator_theorem (p : EscalatorProblem) 
  (h1 : p.stationary_time = 75)
  (h2 : p.moving_time = 30) : 
  solve_escalator_problem p = 50 := by
  sorry

#check clea_escalator_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clea_escalator_theorem_l891_89164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l891_89135

/-- The slope angle of a line with equation y = mx + b is the angle between the line and the positive x-axis. -/
noncomputable def slope_angle (m : ℝ) : ℝ := Real.arctan m

/-- The line equation y = -x - 1 -/
def line_equation (x : ℝ) : ℝ := -x - 1

theorem slope_angle_of_line : slope_angle (-1) = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l891_89135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_speed_is_pi_over_15_l891_89196

/-- Represents the track with its properties -/
structure Track :=
  (inner_radius : ℝ)
  (width_difference_small : ℝ)
  (width_difference_large : ℝ)
  (time_difference : ℝ)

/-- Calculates the walking speed given a track -/
noncomputable def walking_speed (t : Track) : ℝ :=
  let outer_radius := t.inner_radius + (t.width_difference_small + t.width_difference_large) / 4
  let circumference_difference := 2 * Real.pi * (outer_radius - t.inner_radius)
  circumference_difference / t.time_difference

/-- Theorem stating the walking speed for the given track conditions -/
theorem walking_speed_is_pi_over_15 (t : Track) 
  (h1 : t.width_difference_small = 4)
  (h2 : t.width_difference_large = 8)
  (h3 : t.time_difference = 180) :
  walking_speed t = Real.pi / 15 := by
  sorry

#check walking_speed_is_pi_over_15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_speed_is_pi_over_15_l891_89196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l891_89132

-- Define the foci
noncomputable def F₁ : ℝ × ℝ := (-4, 2 - Real.sqrt 2)
noncomputable def F₂ : ℝ × ℝ := (-4, 2 + Real.sqrt 2)

-- Define the hyperbola properties
def hyperbola (h k a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1 ↔
    |Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) - Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2)| = 2

-- Theorem statement
theorem hyperbola_sum (h k a b : ℝ) :
  hyperbola h k a b → h + k + a + b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l891_89132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_range_l891_89175

/-- A circle with center (3, -5) and radius r -/
noncomputable def Circle (r : ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 + 5)^2 = r^2}

/-- The line 4x - 3y - 2 = 0 -/
def Line := {p : ℝ × ℝ | 4 * p.1 - 3 * p.2 - 2 = 0}

/-- The distance between a point and the line -/
noncomputable def distanceToLine (p : ℝ × ℝ) : ℝ :=
  |4 * p.1 - 3 * p.2 - 2| / Real.sqrt 25

/-- The set of points on the circle that are at distance 1 from the line -/
noncomputable def PointsAtDistance1 (r : ℝ) :=
  {p ∈ Circle r | distanceToLine p = 1}

/-- The main theorem -/
theorem circle_intersection_range (r : ℝ) :
  (∃ s : Finset (ℝ × ℝ), s.toSet = PointsAtDistance1 r ∧ s.card = 2) → (4 < r ∧ r < 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_range_l891_89175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_division_l891_89121

/-- A convex polygon -/
structure ConvexPolygon where
  sides : ℕ

/-- A pentagon -/
structure Pentagon where
  sides : ℕ
  property : sides = 5

/-- Predicate to check if a polygon can be divided into a list of pentagons by diagonals -/
def DividePolygonByDiagonals (P : ConvexPolygon) (divisions : List Pentagon) : Prop :=
  sorry

theorem convex_polygon_division (k : ℕ+) :
  ¬ ∃ (P : ConvexPolygon) (divisions : List Pentagon),
    P.sides = 3 * k + 1 ∧
    divisions.length = k ∧
    DividePolygonByDiagonals P divisions :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_division_l891_89121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_cyclic_quadrilateral_l891_89137

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a cyclic quadrilateral -/
structure CyclicQuadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point
  O : Point -- Center of the circumscribed circle

/-- Represents the reflections of the center O with respect to sides AB, BC, and CD -/
structure Reflections where
  T₁ : Point -- Reflection of O with respect to AB
  T₂ : Point -- Reflection of O with respect to BC
  T₃ : Point -- Reflection of O with respect to CD

/-- Main theorem: A cyclic quadrilateral can be constructed given the reflections and angle at B -/
theorem construct_cyclic_quadrilateral 
  (reflections : Reflections) 
  (angle_B : ℝ) :
  ∃ (quad : CyclicQuadrilateral), 
    (quad.O.x - quad.A.x) ^ 2 + (quad.O.y - quad.A.y) ^ 2 =
    (quad.O.x - quad.B.x) ^ 2 + (quad.O.y - quad.B.y) ^ 2 ∧
    (quad.O.x - quad.C.x) ^ 2 + (quad.O.y - quad.C.y) ^ 2 =
    (quad.O.x - quad.D.x) ^ 2 + (quad.O.y - quad.D.y) ^ 2 ∧
    ∃ (k : ℝ), k * (quad.B.x - quad.A.x) = reflections.T₁.x - quad.O.x ∧
               k * (quad.B.y - quad.A.y) = reflections.T₁.y - quad.O.y ∧
    ∃ (l : ℝ), l * (quad.C.x - quad.B.x) = reflections.T₂.x - quad.O.x ∧
               l * (quad.C.y - quad.B.y) = reflections.T₂.y - quad.O.y ∧
    ∃ (m : ℝ), m * (quad.D.x - quad.C.x) = reflections.T₃.x - quad.O.x ∧
               m * (quad.D.y - quad.C.y) = reflections.T₃.y - quad.O.y ∧
    (quad.B.x - quad.A.x) * (quad.C.x - quad.B.x) + 
    (quad.B.y - quad.A.y) * (quad.C.y - quad.B.y) = 
    ((quad.B.x - quad.A.x)^2 + (quad.B.y - quad.A.y)^2) * Real.cos angle_B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_cyclic_quadrilateral_l891_89137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l891_89178

-- Define the function
noncomputable def f (x : ℝ) := Real.sqrt (Real.log (3 - x) / Real.log (1/2) + 1)

-- State the theorem about the domain of the function
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc 1 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l891_89178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_direction_vector_l891_89131

/-- An arithmetic sequence with specified conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_sum : a 1 + a 2 = 10
  second_sum : a 3 + a 4 = 26

/-- The direction vector of a line passing through two points of the sequence -/
def direction_vector (seq : ArithmeticSequence) : ℚ × ℚ :=
  (-1/2, -4)

/-- Theorem stating the direction vector for the given arithmetic sequence -/
theorem arithmetic_sequence_direction_vector (seq : ArithmeticSequence) :
  ∀ n : ℕ, n > 0 →
  let P := (n, seq.a n)
  let Q := (n + 1, seq.a (n + 2))
  direction_vector seq = (-1/2, -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_direction_vector_l891_89131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l891_89145

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (Real.pi / 5 + 3 * x)

theorem smallest_positive_period : 
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
  (∀ T' : ℝ, T' > 0 → (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l891_89145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l891_89172

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2 + x

/-- g(x) is the function derived from f(x) to determine the value of a -/
noncomputable def g (x : ℝ) : ℝ := (Real.log x + x) / x^2

/-- Theorem stating the range of a for which f(x) has exactly two zeros -/
theorem f_has_two_zeros (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ 
    ∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂) ↔ 
  0 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l891_89172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l891_89130

theorem max_negative_integers (a b c d e f : Int) 
  (h_range : ∀ x ∈ ({a, b, c, d, e, f} : Set Int), -10 ≤ x ∧ x ≤ 10)
  (h_product : a * b * c * d * e * f < 0) :
  ∃ (neg_count : Nat), neg_count ≤ 5 ∧ 
    (∃ (subset : Finset Int), subset ⊆ ({a, b, c, d, e, f} : Finset Int) ∧
      subset.card = neg_count ∧ ∀ x ∈ subset, x < 0) ∧
    (∀ (larger_subset : Finset Int), larger_subset ⊆ ({a, b, c, d, e, f} : Finset Int) → 
      larger_subset.card > neg_count → ∃ x ∈ larger_subset, x ≥ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l891_89130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_2_4_l891_89116

/-- A power function that passes through the point (2,4) -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

/-- The theorem stating that f(3) = 9 for a power function passing through (2,4) -/
theorem power_function_through_2_4 : ∃ a : ℝ, f a 2 = 4 ∧ f a 3 = 9 := by
  -- We know that a = 2 satisfies the conditions
  use 2
  constructor
  · -- Prove f 2 2 = 4
    simp [f]
    norm_num
  · -- Prove f 2 3 = 9
    simp [f]
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_2_4_l891_89116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l891_89140

-- Define the arithmetic sequence
noncomputable def a (n : ℕ) : ℝ := 3 * (n : ℝ) - 1

-- Define the sum of the first n terms
noncomputable def S (n : ℕ) : ℝ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

-- Theorem statement
theorem arithmetic_sequence_problem :
  (a 2 + a 5 = 19) ∧ 
  (a 6 - a 3 = 9) ∧ 
  (∃ k : ℕ, S 11 + S k = S (k + 2) ∧ k = 30) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l891_89140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l891_89148

theorem trigonometric_problem (α : ℝ) 
  (h1 : Real.cos (Real.pi + α) = 4/5)
  (h2 : Real.tan α > 0) :
  Real.tan α = 3/4 ∧ 
  (2 * Real.sin (Real.pi - α) + Real.sin (Real.pi/2 - α)) / (Real.cos (-α) + 4 * Real.cos (Real.pi/2 + α)) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l891_89148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l891_89152

/-- The hyperbola defined by x²/16 - y²/9 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The left focus of the hyperbola -/
noncomputable def F₁ : ℝ × ℝ := (-3, 0)

/-- The right focus of the hyperbola -/
noncomputable def F₂ : ℝ × ℝ := (3, 0)

/-- A and B are the endpoints of a chord passing through F₁ -/
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

/-- The length of the chord AB is 6 -/
axiom chord_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6

/-- The chord AB passes through F₁ -/
axiom chord_through_F₁ : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ 
  F₁ = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

/-- The perimeter of triangle ABF₂ is 28 -/
theorem triangle_perimeter : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
  Real.sqrt ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) +
  Real.sqrt ((B.1 - F₂.1)^2 + (B.2 - F₂.2)^2) = 28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l891_89152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_p_false_and_prop_q_true_l891_89138

-- Define the variance function
noncomputable def variance (ξ : ℝ → ℝ) : ℝ := sorry

-- Define the dot product
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := sorry

-- Define perpendicularity for planes
def planes_perpendicular (α β : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define proposition p
def prop_p : Prop :=
  ∀ ξ : ℝ → ℝ, variance ξ = 1 → variance (λ x ↦ 2 * ξ x + 1) = 2

-- Define proposition q
def prop_q : Prop :=
  ∀ (α β : Set (ℝ × ℝ × ℝ)) (u v : ℝ × ℝ × ℝ),
    α ≠ β →
    dot_product u v = 0 →
    planes_perpendicular α β

theorem prop_p_false_and_prop_q_true : ¬prop_p ∧ prop_q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_p_false_and_prop_q_true_l891_89138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l891_89159

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x

/-- The right focus of the ellipse -/
def right_focus : ℝ × ℝ := (1, 0)

/-- The distance formula from a point to a line -/
noncomputable def distance_point_to_line (x y : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x + B * y + C) / Real.sqrt (A^2 + B^2)

theorem distance_right_focus_to_line :
  let (x, y) := right_focus
  let A : ℝ := 1
  let B : ℝ := -Real.sqrt 3
  let C : ℝ := 0
  distance_point_to_line x y A B C = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l891_89159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l891_89166

-- Define the original and target functions
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 2 * Real.cos x

-- Define the transformation
noncomputable def transform (x : ℝ) : ℝ := x / 2 - Real.pi / 4

-- State the theorem
theorem graph_transformation :
  ∀ x : ℝ, g x = f (transform x) :=
by
  intro x
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l891_89166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_equations_l891_89192

/-- In a unit circle with center O, chords PQ and MN are parallel to radius OR.
    Chords MP, PQ, and NR are each s-1 units long, and chord MN is 2s units long. -/
def circle_config (s : ℝ) : Prop :=
  ∃ (O P Q M N R : ℝ × ℝ),
    (∀ x : ℝ × ℝ, (x.1 - O.1)^2 + (x.2 - O.2)^2 = 1) ∧
    (P.1 - Q.1 = M.1 - N.1 ∧ P.2 - Q.2 = M.2 - N.2) ∧
    (P.1 - Q.1 = O.1 - R.1 ∧ P.2 - Q.2 = O.2 - R.2) ∧
    ((M.1 - P.1)^2 + (M.2 - P.2)^2 = (s-1)^2) ∧
    ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (s-1)^2) ∧
    ((N.1 - R.1)^2 + (N.2 - R.2)^2 = (s-1)^2) ∧
    ((M.1 - N.1)^2 + (M.2 - N.2)^2 = (2*s)^2)

theorem circle_chord_equations (s : ℝ) (h : circle_config s) :
  (2*s - (s-1) = 2) ∧ ¬((2*s)*(s-1) = 2) ∧ ¬((2*s)^2 - (s-1)^2 = 3*Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_equations_l891_89192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_negative_phi_l891_89112

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem largest_negative_phi : 
  ∀ φ : ℝ, 
  (∀ x : ℝ, f (x + π/8) φ = f (-x + π/8) φ) → 
  (∀ ψ : ℝ, ψ < 0 → ψ ≤ -3*π/4) →
  φ = -3*π/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_negative_phi_l891_89112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_transformations_l891_89120

/-- Represents the axes in a 2D plane -/
inductive Axis
  | PositiveX
  | NegativeX
  | PositiveY
  | NegativeY

/-- Represents the position of an L-shape -/
structure LPosition where
  base_axis : Axis
  stem_axis : Axis
  quadrant : Nat

/-- Applies a 180° counterclockwise rotation to an L-shape -/
def rotate180CCW (l : LPosition) : LPosition :=
  sorry

/-- Reflects an L-shape in the x-axis -/
def reflectXAxis (l : LPosition) : LPosition :=
  sorry

/-- Applies a 90° clockwise rotation to an L-shape -/
def rotate90CW (l : LPosition) : LPosition :=
  sorry

/-- The initial position of the L-shape -/
def initialL : LPosition := {
  base_axis := Axis.PositiveX,
  stem_axis := Axis.PositiveY,
  quadrant := 1
}

/-- The final position of the L-shape after transformations -/
def finalL : LPosition := {
  base_axis := Axis.NegativeY,
  stem_axis := Axis.NegativeX,
  quadrant := 3
}

/-- Theorem stating that the transformations result in the expected final position -/
theorem l_transformations :
  rotate90CW (reflectXAxis (rotate180CCW initialL)) = finalL :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_transformations_l891_89120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l891_89184

theorem expansion_coefficient (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → 
    ∃ f : ℝ → ℝ, (a + 1/x) * (1 - x)^4 = -6*x + f x) ↔ 
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l891_89184
