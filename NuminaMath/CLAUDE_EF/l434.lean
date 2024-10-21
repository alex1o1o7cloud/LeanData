import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_product_equality_l434_43496

theorem tangent_product_equality (α β γ : Real) 
  (h : Real.cos γ = Real.cos α * Real.cos β) : 
  Real.tan ((γ + α) / 2) * Real.tan ((γ - α) / 2) = (Real.tan (β / 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_product_equality_l434_43496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l434_43484

/-- A circle passes through (0,1) and is tangent to y = x^2 + 1 at (3,10). Its center is (-3/7, 113/14). -/
theorem circle_center (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) : 
  (∀ (p : ℝ × ℝ), p ∈ C → (p.1 - center.1)^2 + (p.2 - center.2)^2 = (center.1)^2 + (center.2 - 1)^2) →  -- C is a circle with given center
  (0, 1) ∈ C →  -- C passes through (0,1)
  (3, 10) ∈ C →  -- C passes through (3,10)
  (∀ p : ℝ × ℝ, p ∈ C → p.2 ≥ p.1^2 + 1) →  -- C is tangent to y = x^2 + 1
  (∃ (t : ℝ), ∀ p : ℝ × ℝ, p ∈ C → p.2 - 10 = 6 * (p.1 - 3) + t * ((p.1 - 3)^2 + (p.2 - 10)^2)) →  -- Tangency condition
  center = (-3/7, 113/14) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l434_43484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_real_part_l434_43455

noncomputable def complex_options : List ℂ := [-3, -2 + Complex.I, -Real.sqrt 3 + Real.sqrt 3 * Complex.I, -2 + Real.sqrt 2 * Complex.I, 3 * Complex.I]

def max_real_part (z : ℂ) : ℝ := (z^4).re

theorem greatest_real_part :
  ∃ (z₁ z₂ : ℂ), z₁ ∈ complex_options ∧ z₂ ∈ complex_options ∧
  z₁ ≠ z₂ ∧
  ∀ (w : ℂ), w ∈ complex_options → max_real_part w ≤ max_real_part z₁ ∧
  max_real_part z₁ = max_real_part z₂ ∧
  (z₁ = -3 ∨ z₁ = 3 * Complex.I) ∧ (z₂ = -3 ∨ z₂ = 3 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_real_part_l434_43455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l434_43452

theorem trigonometric_identity (α : ℝ) (h1 : Real.cos α = 2/3) (h2 : α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) :
  Real.sin (α - 2*Real.pi) + Real.sin (-α - 3*Real.pi) * Real.cos (α - 3*Real.pi) = -5*Real.sqrt 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l434_43452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_approximation_l434_43434

-- Define an angle as a real number between 0 and 2π
def Angle : Type := {x : ℝ // 0 ≤ x ∧ x < 2 * Real.pi}

-- Define a predicate for constructible angles
def constructible : Angle → Prop := sorry

-- Theorem statement
theorem angle_approximation (α : Angle) (ε : ℝ) (h : ε > 0) :
  ∃ β : Angle, constructible β ∧ |α.val - β.val| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_approximation_l434_43434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_21952000_l434_43420

theorem cube_root_of_21952000 : (21952000 : ℝ)^(1/3) = 280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_21952000_l434_43420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_a_eq_neg_one_l434_43468

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 + (2 * a^3 - a^2) * log x - (a^2 + 2*a - 1) * x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a x : ℝ) : ℝ := x + (2 * a^3 - a^2) / x - (a^2 + 2*a - 1)

/-- Theorem stating that if x=1 is an extreme point of f(a,x), then a = -1 -/
theorem extreme_point_implies_a_eq_neg_one (a : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → |f a x - f a 1| ≤ ε * |x - 1|) →
  f_derivative a 1 = 0 →
  a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_a_eq_neg_one_l434_43468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_no_advice_formula_l434_43463

/-- The expected number of explorers who do not receive advice -/
noncomputable def expectedNoAdvice (n : ℕ) (p : ℝ) : ℝ :=
  (1 - (1 - p) ^ n) / p

/-- Theorem: The expected number of explorers who do not receive advice
    is (1 - (1-p)^n) / p given n explorers and friendship probability p -/
theorem expected_no_advice_formula (n : ℕ) (p : ℝ) 
  (h1 : 0 < p) (h2 : p < 1) :
  expectedNoAdvice n p = (1 - (1 - p) ^ n) / p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_no_advice_formula_l434_43463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concrete_volume_is_12_cubic_yards_l434_43493

/-- Represents the dimensions of the sidewalk sections -/
structure SidewalkDimensions where
  rectWidth : ℝ
  rectLength : ℝ
  rectThickness : ℝ
  trapHeight : ℝ
  trapBottomBase : ℝ
  trapTopBase : ℝ
  trapThickness : ℝ

/-- Calculates the total volume of concrete needed for the sidewalk in cubic yards -/
noncomputable def calculateConcreteVolume (d : SidewalkDimensions) : ℝ :=
  let rectVolume := d.rectWidth * d.rectLength * d.rectThickness / 27
  let trapArea := (d.trapBottomBase + d.trapTopBase) / 2 * d.trapHeight
  let trapVolume := trapArea * d.trapThickness / 27
  rectVolume + trapVolume

/-- Rounds up a real number to the nearest integer -/
noncomputable def ceilToInt (x : ℝ) : ℤ :=
  Int.ceil x

/-- The main theorem stating that the required concrete volume is 12 cubic yards -/
theorem concrete_volume_is_12_cubic_yards (d : SidewalkDimensions) 
  (h1 : d.rectWidth = 4/3)
  (h2 : d.rectLength = 100/3)
  (h3 : d.rectThickness = 1/9)
  (h4 : d.trapHeight = 100/3)
  (h5 : d.trapBottomBase = 4/3)
  (h6 : d.trapTopBase = 8/3)
  (h7 : d.trapThickness = 1/9) :
  ceilToInt (calculateConcreteVolume d) = 12 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concrete_volume_is_12_cubic_yards_l434_43493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_3x3_board_l434_43402

/-- A coloring of a 3x3 board is valid if no two squares in the same row, column, or diagonal have the same color. -/
def IsValidColoring (board : Fin 3 → Fin 3 → Nat) : Prop :=
  (∀ i j₁ j₂, j₁ ≠ j₂ → board i j₁ ≠ board i j₂) ∧ 
  (∀ i₁ i₂ j, i₁ ≠ i₂ → board i₁ j ≠ board i₂ j) ∧ 
  (∀ i₁ i₂, i₁ ≠ i₂ → board i₁ i₁ ≠ board i₂ i₂) ∧ 
  (∀ i₁ i₂, i₁ ≠ i₂ → board i₁ (2 - i₁) ≠ board i₂ (2 - i₂))

/-- The minimum number of colors required for a valid coloring of a 3x3 board is 5. -/
theorem min_colors_3x3_board :
  (∃ (board : Fin 3 → Fin 3 → Nat), IsValidColoring board ∧ (∀ i j, board i j < 5)) ∧
  (∀ n : Nat, n < 5 → ¬∃ (board : Fin 3 → Fin 3 → Nat), IsValidColoring board ∧ (∀ i j, board i j < n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_3x3_board_l434_43402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_multiple_of_ten_valid_set_l434_43431

/-- A point on a circle with circumference 15 units -/
structure CirclePoint where
  position : ℝ
  nonneg : 0 ≤ position
  lt_circ : position < 15

/-- The distance between two points on the circle -/
noncomputable def circleDistance (p q : CirclePoint) : ℝ :=
  min (((q.position - p.position) % 15 + 15) % 15)
      (((p.position - q.position) % 15 + 15) % 15)

/-- A set of points on the circle satisfying the distance conditions -/
def ValidPointSet (S : Set CirclePoint) : Prop :=
  (∀ p ∈ S, ∃! q, q ∈ S ∧ circleDistance p q = 2) ∧
  (∀ p ∈ S, ∃! q, q ∈ S ∧ circleDistance p q = 3)

theorem exists_non_multiple_of_ten_valid_set :
  ∃ (S : Set CirclePoint), ValidPointSet S ∧ Finite S ∧ ¬(∃ k, Nat.card S = 10 * k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_multiple_of_ten_valid_set_l434_43431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l434_43406

-- Define the complex number z
noncomputable def z (b : ℝ) : ℂ := 2 + b * Complex.I

-- Define the complex number ω
noncomputable def ω (b : ℝ) : ℂ := z b / (1 - Complex.I)

-- Theorem statement
theorem complex_number_properties (b : ℝ) 
  (h1 : b > 0) 
  (h2 : (z b)^2 = Complex.I * Complex.im ((z b)^2)) : 
  b = 2 ∧ Complex.abs (ω b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l434_43406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_zero_units_digit_l434_43498

def modified_fibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => modified_fibonacci n + modified_fibonacci (n + 1)

theorem first_zero_units_digit :
  (∀ k < 13, modified_fibonacci k % 10 ≠ 0) ∧
  modified_fibonacci 13 % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_zero_units_digit_l434_43498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_positions_l434_43412

-- Define the type for positions
inductive Position : Type
  | first : Position
  | second : Position
  | third : Position
  | fourth : Position

-- Define the type for students
inductive Student : Type
  | A : Student
  | B : Student
  | C : Student
  | D : Student

-- Define a function that assigns positions to students
variable (position : Student → Position)

-- Define adjacency for positions
def adjacent (p1 p2 : Position) : Prop :=
  (p1 = Position.first ∧ p2 = Position.second) ∨
  (p1 = Position.second ∧ p2 = Position.third) ∨
  (p1 = Position.third ∧ p2 = Position.fourth) ∨
  (p1 = Position.second ∧ p2 = Position.first) ∨
  (p1 = Position.third ∧ p2 = Position.second) ∨
  (p1 = Position.fourth ∧ p2 = Position.third)

-- State the theorem
theorem correct_positions :
  (position Student.A ≠ Position.first ∧ position Student.A ≠ Position.second) →
  (position Student.B ≠ Position.second ∧ position Student.B ≠ Position.third) →
  adjacent (position Student.B) (position Student.C) →
  adjacent (position Student.C) (position Student.D) →
  (∀ (s1 s2 : Student), s1 ≠ s2 → position s1 ≠ position s2) →
  (position Student.A = Position.fourth ∧
   position Student.B = Position.first ∧
   position Student.C = Position.second ∧
   position Student.D = Position.third) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_positions_l434_43412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_diagonal_relation_l434_43403

/-- Predicate to represent that a is a side length of a regular pentagon -/
def is_regular_pentagon_side (a : ℝ) : Prop := sorry

/-- Predicate to represent that d is a diagonal length of a regular pentagon with side a -/
def is_regular_pentagon_diagonal (a d : ℝ) : Prop := sorry

/-- For a regular pentagon with side length a and diagonal length d, 
    the relationship d^2 = a^2 + ad holds. -/
theorem regular_pentagon_diagonal_relation (a d : ℝ) 
  (h_a : a > 0) 
  (h_d : d > 0) 
  (h_pentagon : is_regular_pentagon_side a)
  (h_diagonal : is_regular_pentagon_diagonal a d) : 
  d^2 = a^2 + a*d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_diagonal_relation_l434_43403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_PQRS_l434_43488

/-- Represents a tetrahedron PQRS with given side lengths -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- The specific tetrahedron PQRS from the problem -/
noncomputable def PQRS : Tetrahedron where
  PQ := 6
  PR := 3.5
  PS := 4
  QR := 5
  QS := 4.5
  RS := (9/2) * Real.sqrt 2

/-- M is the midpoint of RS -/
noncomputable def M (t : Tetrahedron) : ℝ × ℝ × ℝ := sorry

/-- Length of MQ -/
noncomputable def MQ (t : Tetrahedron) : ℝ := sorry

/-- Volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of PQRS is 12 / MQ -/
theorem volume_PQRS : volume PQRS = 12 / MQ PQRS := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_PQRS_l434_43488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_with_product_one_l434_43411

theorem infinite_solutions_with_product_one :
  ∃ f : ℕ → ℤ × ℤ,
    Function.Injective f ∧
    ∀ n : ℕ,
      let (a, b) := f n
      ∃ k : ℝ,
        k ≠ (1 : ℝ) / k ∧
        k^2012 = a * k + b ∧
        (1 / k)^2012 = a * (1 / k) + b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_with_product_one_l434_43411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l434_43453

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 - 24 * x - y^2 + 6 * y - 3 = 0

/-- The distance between vertices of the hyperbola -/
noncomputable def vertex_distance : ℝ := 2 * Real.sqrt 7.5

/-- Theorem stating that the distance between vertices of the hyperbola
    defined by the given equation is 2√(7.5) -/
theorem hyperbola_vertex_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola_equation x₁ y₁ ∧
    hyperbola_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = vertex_distance^2 ∧
    ∀ (x y : ℝ), hyperbola_equation x y →
      (x - x₁)^2 + (y - y₁)^2 ≤ vertex_distance^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l434_43453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l434_43469

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

-- State the theorem
theorem f_max_value :
  ∃ (x : ℝ), f x = Real.exp 1 ∧ ∀ (y : ℝ), f y ≤ Real.exp 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l434_43469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_l434_43421

-- Define the expression as noncomputable
noncomputable def f (x y z : ℝ) : ℝ :=
  (x^2 * y^2 / ((x^2 - y*z) * (y^2 - x*z))) +
  (x^2 * z^2 / ((x^2 - y*z) * (z^2 - x*y))) +
  (y^2 * z^2 / ((y^2 - x*z) * (z^2 - x*y)))

-- State the theorem
theorem expression_equals_one
  (x y z : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (h_sum : x + y + z = -1) :
  f x y z = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_l434_43421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l434_43478

theorem sequence_existence (n : ℕ) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n ↔ ∃ (x : Fin n → ℕ), StrictMono x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l434_43478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_cube_root_denominator_l434_43477

theorem rationalize_cube_root_denominator :
  1 / (Real.rpow 2 (1/3) - 1) = Real.rpow 4 (1/3) + Real.rpow 2 (1/3) + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_cube_root_denominator_l434_43477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_upper_bound_A_upper_bound_achievable_l434_43419

/-- The expression A as a function of x, y, and z -/
noncomputable def A (x y z : ℝ) : ℝ :=
  ((x - y) * Real.sqrt (x^2 + y^2) + (y - z) * Real.sqrt (y^2 + z^2) + (z - x) * Real.sqrt (z^2 + x^2) + Real.sqrt 2) /
  ((x - y)^2 + (y - z)^2 + (z - x)^2 + 2)

/-- The theorem stating that A is bounded above by 1/√2 for positive x, y, z -/
theorem A_upper_bound (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  A x y z ≤ 1 / Real.sqrt 2 := by
  sorry

/-- The theorem stating that the upper bound 1/√2 is achievable -/
theorem A_upper_bound_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ A x y z = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_upper_bound_A_upper_bound_achievable_l434_43419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_of_divisor_and_remainder_l434_43407

theorem largest_product_of_divisor_and_remainder (m a b : ℕ) : 
  m % 720 = 83 → 
  m % a = b → 
  a > 0 →
  b > 0 →
  a * b ≤ 5112 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_of_divisor_and_remainder_l434_43407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l434_43414

theorem inequality_solution_set (a : ℝ) :
  ((3 - a) / 2 - 2 = 2) →
  (Set.Ioi 9 = { x : ℝ | (2 - a / 5) < (1 / 3) * x }) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l434_43414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_466560000_l434_43424

theorem cube_root_of_466560000 : (466560000 : ℝ) ^ (1/3 : ℝ) = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_466560000_l434_43424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_ratio_l434_43465

theorem tan_ratio_from_sin_ratio (α β : ℝ) 
  (h : Real.sin (α + β) / Real.sin (α - β) = 3) : 
  Real.tan α / Real.tan β = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_ratio_l434_43465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_without_extreme_points_l434_43485

-- Define a polynomial function type
def PolynomialFunction := ℝ → ℝ

-- Define what it means for a function to be polynomial
def IsPolynomial (f : ℝ → ℝ) : Prop := sorry

-- Define what an extreme point is
def IsExtremePoint (f : ℝ → ℝ) (x : ℝ) : Prop := sorry

theorem polynomial_without_extreme_points :
  ∃ (f : PolynomialFunction) (a b : ℝ), 
    a < b ∧ 
    IsPolynomial f ∧ 
    ∀ x ∈ Set.Ioo a b, ¬IsExtremePoint f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_without_extreme_points_l434_43485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_side_b_value_l434_43472

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C
  S : ℝ  -- Area of the triangle

-- Theorem for part (1)
theorem angle_A_value (ABC : Triangle) (h : ABC.b * ABC.c * (Real.cos ABC.A) = 2 * Real.sqrt 3 * ABC.S) :
  ABC.A = π / 6 := by
  sorry

-- Theorem for part (2)
theorem side_b_value (ABC : Triangle) 
  (h1 : Real.tan ABC.A / Real.tan ABC.B = 1 / 2 ∧ Real.tan ABC.B / Real.tan ABC.C = 2 / 3)
  (h2 : ABC.c = 1) :
  ABC.b = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_side_b_value_l434_43472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_example_l434_43479

def vector1 : Fin 3 → ℝ := ![4, -3, 5]
def vector2 : Fin 3 → ℝ := ![-6, 1, -2]

def dot_product (v1 v2 : Fin 3 → ℝ) : ℝ :=
  (v1 0) * (v2 0) + (v1 1) * (v2 1) + (v1 2) * (v2 2)

theorem dot_product_example : dot_product vector1 vector2 = -37 := by
  unfold dot_product vector1 vector2
  simp
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_example_l434_43479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_l434_43439

theorem probability_factor_of_36 : 
  (Finset.filter (λ n ↦ n ∣ 36) (Finset.range 37)).card / 36 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_l434_43439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_rounds_to_target_l434_43481

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem not_rounds_to_target : 
  round_to_hundredth 74.554 ≠ 74.56 ∧ 
  round_to_hundredth 74.559 = 74.56 ∧ 
  round_to_hundredth 74.5551 = 74.56 ∧ 
  round_to_hundredth 74.559999999 = 74.56 ∧ 
  round_to_hundredth 74.56333 = 74.56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_rounds_to_target_l434_43481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_divisible_l434_43430

theorem least_multiple_divisible (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(112 ∣ 72 * m) ∨ ¬(199 ∣ 72 * m)) ∧ 
  (112 ∣ 72 * n) ∧ 
  (199 ∣ 72 * n) → 
  n = 310 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_divisible_l434_43430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_speed_l434_43462

/-- Proves that the walking speed is 4 km/hr given the conditions of the problem -/
theorem walking_speed (total_distance : ℝ) (running_speed : ℝ) (total_time : ℝ) (walking_speed : ℝ)
  (h1 : total_distance = 12)
  (h2 : running_speed = 8)
  (h3 : total_time = 2.25)
  (h4 : total_distance / 2 / running_speed + total_distance / 2 / walking_speed = total_time) :
  walking_speed = 4 :=
by sorry

#check walking_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_speed_l434_43462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pyramid_properties_l434_43433

/-- Represents a right pyramid with a rectangular base -/
structure RightPyramid where
  baseLength : ℝ
  baseWidth : ℝ
  height : ℝ

/-- Calculates the total length of edges for a right pyramid -/
noncomputable def totalEdgeLength (p : RightPyramid) : ℝ :=
  2 * (p.baseLength + p.baseWidth) + 4 * Real.sqrt (p.height^2 + (Real.sqrt (p.baseLength^2 + p.baseWidth^2) / 2)^2)

/-- Calculates the volume of a right pyramid -/
noncomputable def volume (p : RightPyramid) : ℝ :=
  (1/3) * p.baseLength * p.baseWidth * p.height

theorem right_pyramid_properties (p : RightPyramid) 
    (h1 : p.baseLength = 12)
    (h2 : p.baseWidth = 8)
    (h3 : p.height = 15) :
    totalEdgeLength p = 40 + 4 * Real.sqrt 277 ∧ volume p = 480 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pyramid_properties_l434_43433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_unique_zero_implies_a_equals_one_lower_bound_of_m_l434_43437

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log x + a*x^2 + 2
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x - 2

-- Part I: Tangent line equation
theorem tangent_line_equation (x y : ℝ) :
  f (-1) 1 = 1 ∧ (deriv (f (-1))) 1 = -3 →
  3*x + y - 4 = 0 ↔ y = (deriv (f (-1))) 1 * (x - 1) + f (-1) 1 :=
by sorry

-- Part II: Value of a
theorem unique_zero_implies_a_equals_one (a : ℝ) :
  a > 0 ∧ (∃! x, g a x = 0) →
  a = 1 :=
by sorry

-- Part III: Range of m
theorem lower_bound_of_m (m : ℝ) :
  (∀ x, Real.exp (-2) < x → x < Real.exp 1 → g 1 x ≤ m) →
  m ≥ 2 * Real.exp 2 - 3 * Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_unique_zero_implies_a_equals_one_lower_bound_of_m_l434_43437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_truncated_cone_volume_l434_43404

/-- The volume of a truncated right circular cone -/
noncomputable def truncatedConeVolume (R r h : ℝ) : ℝ :=
  (Real.pi * h / 3) * (R^2 + r^2 + R*r)

/-- Theorem: Volume of a specific truncated cone -/
theorem specific_truncated_cone_volume :
  truncatedConeVolume 10 5 10 = (1750/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_truncated_cone_volume_l434_43404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_theta_l434_43427

theorem tan_pi_4_plus_theta (θ : ℝ) :
  Real.tan (π/4 + θ) = 3 → Real.sin (2*θ) - 2*(Real.cos θ)^2 = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_theta_l434_43427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l434_43443

/-- A hyperbola with foci on the x-axis and axis of symmetry on the coordinate axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The slope of the asymptote of the hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ := h.b / h.a

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + (h.b / h.a)^2)

/-- Theorem: The eccentricity of a hyperbola with an asymptote parallel to x + 2y - 3 = 0 is √5/2 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_asymptote : asymptote_slope h = 1/2) : 
  eccentricity h = Real.sqrt 5 / 2 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l434_43443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l434_43482

open Complex

noncomputable def parallelogramArea (z : ℂ) : ℝ :=
  ‖z - 0‖ * ‖(z + z⁻¹) - z⁻¹‖ * abs (sin (arg (z - 0) - arg ((z + z⁻¹) - z⁻¹)))

theorem min_value_theorem (z : ℂ) (h1 : z.im < 0) (h2 : parallelogramArea z = 42 / 43) :
  ∃ (d : ℝ), d^2 = 147 / 43 ∧ ∀ (w : ℂ), ‖w / 2 + 2 / w‖^2 ≥ d^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l434_43482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l434_43446

-- Define the ellipse parameters
noncomputable def a : ℝ := Real.sqrt 8
def b : ℝ := 2
def c : ℝ := 2

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 4 = 1

-- Define the focus position
def focus : ℝ × ℝ := (2, 0)

-- Define eccentricity
noncomputable def eccentricity : ℝ := c / a

-- Theorem statement
theorem ellipse_eccentricity :
  ellipse_equation 0 0 ∧ focus = (2, 0) → eccentricity = Real.sqrt 2 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l434_43446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_intervals_of_f_l434_43438

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 2*x + 1)

-- State the theorem
theorem monotonic_increasing_intervals_of_f :
  ∃ (a b : ℝ), a = -3 ∧ b = -1 ∧
  (∀ x y, x < y ∧ y < a → f x < f y) ∧
  (∀ x y, b < x ∧ x < y → f x < f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_intervals_of_f_l434_43438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_l434_43451

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the points
def A : ℝ × ℝ → Prop := λ p => circle_eq p.1 p.2
def B : ℝ × ℝ → Prop := λ p => circle_eq p.1 p.2
def C : ℝ × ℝ → Prop := λ p => circle_eq p.1 p.2
def D : ℝ × ℝ := (1, 0)
def P : ℝ × ℝ := (5, 0)

-- Define the midpoint condition
def is_midpoint (m x y : ℝ × ℝ) : Prop :=
  m.1 = (x.1 + y.1) / 2 ∧ m.2 = (x.2 + y.2) / 2

-- Define the vector sum
def vector_sum (a b c p : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 - p.1) + (b.1 - p.1) + (c.1 - p.1), (a.2 - p.2) + (b.2 - p.2) + (c.2 - p.2))

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- Theorem statement
theorem max_vector_sum :
  ∀ (a b c : ℝ × ℝ), A a → B b → C c →
  is_midpoint D b c →
  (∀ (x y z : ℝ × ℝ), A x → B y → C z →
    is_midpoint D y z →
    magnitude (vector_sum a b c P) ≥ magnitude (vector_sum x y z P)) →
  magnitude (vector_sum a b c P) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_l434_43451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosC_and_max_area_l434_43417

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the cosine rule
noncomputable def cosine_rule (t : Triangle) : ℝ := (t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)

-- Define the area of a triangle using two sides and the sine of the included angle
noncomputable def triangle_area (a b sinC : ℝ) : ℝ := (1/2) * a * b * sinC

theorem triangle_cosC_and_max_area (t : Triangle) 
  (h1 : 2 * (t.a^2 + t.b^2 - t.c^2) = 3 * t.a * t.b) :
  cosine_rule t = 3/4 ∧ 
  (t.c = 2 → ∃ (max_area : ℝ), max_area = Real.sqrt 7 ∧ 
    ∀ (a' b' : ℝ), triangle_area a' b' (Real.sqrt (1 - (3/4)^2)) ≤ max_area) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosC_and_max_area_l434_43417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l434_43416

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 2/5) : Real.cos (7*θ) = 16716/78125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l434_43416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_base_equals_result_l434_43494

def percentage : Float := 16.666666666666668
def base : Float := 480
def result : Float := 80

theorem percentage_of_base_equals_result :
  (percentage / 100 * base).round = result := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_base_equals_result_l434_43494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_trip_cost_l434_43432

/-- Represents the cost of a trip between two places -/
noncomputable def TripCost (distance : ℝ) (gasolinePrice : ℝ) (driverWage : ℝ) (speed : ℝ) : ℝ :=
  let time := distance / speed
  let fuelConsumption := (3 + speed^2 / 360) * time
  gasolinePrice * fuelConsumption + driverWage * time

theorem optimal_trip_cost :
  let distance := (120 : ℝ)
  let gasolinePrice := (6 : ℝ)
  let driverWage := (42 : ℝ)
  let costFunction (speed : ℝ) := TripCost distance gasolinePrice driverWage speed
  ∀ speed : ℝ, 50 ≤ speed → speed ≤ 100 →
    costFunction speed ≥ 240 ∧
    (costFunction speed = 240 ↔ speed = 60) :=
by sorry

#check optimal_trip_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_trip_cost_l434_43432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_progression_l434_43450

noncomputable def a : ℝ := 5 + 2 * Real.sqrt 6
noncomputable def c : ℝ := 5 - 2 * Real.sqrt 6

theorem arithmetic_and_geometric_progression :
  (∀ b : ℝ, (b - a = c - b) → b = 5) ∧
  (∀ b : ℝ, (b^2 = a * c) → (b = 1 ∨ b = -1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_progression_l434_43450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_rent_expense_l434_43441

-- Define the given conditions
noncomputable def monthly_salary : ℝ := 5000
noncomputable def tax_rate : ℝ := 0.1
def late_payments : ℕ := 2
noncomputable def salary_fraction : ℝ := 3/5

-- Define the theorem
theorem monthly_rent_expense :
  let after_tax_salary := monthly_salary * (1 - tax_rate)
  let total_late_payment := after_tax_salary * salary_fraction
  total_late_payment / late_payments = 1350 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_rent_expense_l434_43441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l434_43495

theorem expression_simplification (x : ℝ) 
  (hx_nonzero : x ≠ 0) 
  (hx_pos1 : x^2 - 3*x + 2 > 0) 
  (hx_pos2 : x^2 + 3*x + 2 > 0) : 

  (((x^2 - 3*x + 2)^(-(1/2 : ℝ)) - (x^2 + 3*x + 2)^(-(1/2 : ℝ))) / 
  ((x^2 - 3*x + 2)^(-(1/2 : ℝ)) + (x^2 + 3*x + 2)^(-(1/2 : ℝ)))) - 1 + 
  ((x^4 - 5*x^2 + 4)^((1/2 : ℝ))) / (3*x) = (x^2 - 3*x + 2) / (3*x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l434_43495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_single_root_multiplicity_l434_43400

variable (R : Type*) [CommRing R] [IsDomain R]
variable (n p q : ℕ) (f : Polynomial R)

/-- A polynomial f of degree n such that f^p is divisible by f'^q has a single root of multiplicity n -/
theorem polynomial_single_root_multiplicity
  (hdeg : Polynomial.degree f = n)
  (hdiv : (f ^ p) ∣ (Polynomial.derivative f ^ q))
  (hp : p > 0)
  (hq : q > 0) :
  (Polynomial.derivative f ∣ f) ∧ ∃ (α : R) (c : R), f = c • (Polynomial.X - Polynomial.C α) ^ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_single_root_multiplicity_l434_43400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_side_length_proof_l434_43444

/-- The length of the side of a regular decagon inscribed in a circle of radius 1 -/
noncomputable def decagon_side_length : ℝ := (Real.sqrt 5 - 1) / 2

/-- Theorem stating that the length of the side of a regular decagon inscribed in a circle of radius 1 is (√5 - 1) / 2 -/
theorem decagon_side_length_proof :
  let r : ℝ := 1 -- radius of the circle
  let α : ℝ := 2 * Real.pi / 10 -- central angle for each side of the decagon
  let l : ℝ := 2 * r * Real.sin (α / 2) -- length of the side
  l = decagon_side_length :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_side_length_proof_l434_43444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l434_43435

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

theorem odd_function_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 2), f (x + a) ≥ 2 * f x) →
  a ≥ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l434_43435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sequence_sum_l434_43448

theorem product_sequence_sum (a : ℚ) : 
  (∃ (n : ℕ), 
    (Finset.prod (Finset.range n) (λ i => ((i + 2 : ℚ) / (i + 1)))) * 
    ((2 * a + 1) / a) * 
    (Finset.prod (Finset.range n) (λ i => ((i + n + 2 : ℚ) / (i + n + 1)))) * 
    (7 * a / (a - 1)) = 45 / 2) →
  7 * a + (a - 1) = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sequence_sum_l434_43448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hygiene_disease_relationship_l434_43456

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  !![40, 60; 10, 90]

-- Define the formula for K²
noncomputable def K_squared (n : ℕ) (a b c d : ℕ) : ℝ :=
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% confidence
def critical_value : ℝ := 6.635

-- Define the total number of surveyed individuals
def total_surveyed : ℕ := 200

-- Theorem statement
theorem hygiene_disease_relationship :
  -- There is a 99% confidence in the difference of hygiene habits
  K_squared total_surveyed
    (contingency_table 0 0) (contingency_table 0 1)
    (contingency_table 1 0) (contingency_table 1 1) > critical_value ∧
  -- The risk level R can be expressed as given
  ∃ R : ℝ, R = (contingency_table 0 0 / (contingency_table 0 0 + contingency_table 0 1) : ℝ) /
              (contingency_table 0 1 / (contingency_table 0 0 + contingency_table 0 1) : ℝ) *
              (contingency_table 1 1 / (contingency_table 1 0 + contingency_table 1 1) : ℝ) /
              (contingency_table 1 0 / (contingency_table 1 0 + contingency_table 1 1) : ℝ) ∧
  -- The estimated value of R is 6
  R = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hygiene_disease_relationship_l434_43456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_g100_l434_43415

-- Define g₀
def g₀ (x : ℝ) : ℝ := x + |x - 150| - |x + 50|

-- Define gₙ recursively
def g : ℕ → ℝ → ℝ
  | 0 => g₀
  | n + 1 => fun x => |g n x| - 2

-- Theorem statement
theorem unique_solution_g100 :
  ∃! x : ℝ, g 100 x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_g100_l434_43415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l434_43457

-- Define the function g(x) as noncomputable
noncomputable def g (x : ℝ) : ℝ := (3*x - 9)*(x - 4)/x

-- Define the solution set
def solution_set : Set ℝ := Set.Ioc 0 3 ∪ Set.Ici 4

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | g x ≥ 0 ∧ x ≠ 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l434_43457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_toppings_l434_43409

theorem pizza_toppings (total_slices : ℕ) (onion_slices : ℕ) (olive_slices : ℕ) 
  (h1 : total_slices = 18)
  (h2 : onion_slices = 10)
  (h3 : olive_slices = 10)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ onion_slices ∨ slice ≤ olive_slices)) :
  ∃ both_toppings : ℕ, both_toppings = onion_slices + olive_slices - total_slices ∧ both_toppings = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_toppings_l434_43409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_position_of_VBRGJ_l434_43459

def letters : List Char := ['B', 'G', 'J', 'R', 'V']

def target_word : List Char := ['V', 'B', 'R', 'G', 'J']

def is_alphabetical (w1 w2 : List Char) : Bool :=
  match w1, w2 with
  | [], [] => true
  | [], _ => true
  | _, [] => false
  | h1::t1, h2::t2 => if h1 < h2 then true
                      else if h1 > h2 then false
                      else is_alphabetical t1 t2

def count_permutations_before (word : List Char) (all_perms : List (List Char)) : Nat :=
  (all_perms.filter (fun p => is_alphabetical p word)).length

theorem position_of_VBRGJ :
  let all_permutations := letters.permutations
  count_permutations_before target_word all_permutations + 1 = 115 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_position_of_VBRGJ_l434_43459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_point_properties_l434_43489

/-- The ellipse E in the Cartesian coordinate system xOy -/
structure Ellipse where
  center : ℝ × ℝ
  major_axis : ℝ
  equation : ℝ → ℝ → Prop

/-- The circle C in the Cartesian coordinate system xOy -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A point P on the ellipse E -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line passing through point P -/
structure Line where
  slope : ℝ
  point : Point

theorem ellipse_and_point_properties
  (E : Ellipse)
  (C : Circle)
  (P : Point)
  (l₁ l₂ : Line) :
  E.center = (0, 0) →
  E.major_axis = 8 →
  C.equation = fun x y ↦ x^2 + y^2 - 4*x + 2 = 0 →
  (∃ x y, C.equation x y ∧ (x = 2 ∧ y = 0)) →
  P.x < 0 →
  E.equation P.x P.y →
  l₁.point = P →
  l₂.point = P →
  l₁.slope * l₂.slope = 1/2 →
  (∀ x y, C.equation x y → (y - P.y ≠ l₁.slope * (x - P.x) ∧ y - P.y ≠ l₂.slope * (x - P.x))) →
  (E.equation = fun x y ↦ x^2/16 + y^2/12 = 1) ∧
  ((P.x, P.y) = (-2, 3) ∨ (P.x, P.y) = (-2, -3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_point_properties_l434_43489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteen_digit_multiple_of_2_13_l434_43418

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n]
  else (n % 10) :: digits (n / 10)

theorem thirteen_digit_multiple_of_2_13 : ∃ N : ℕ,
  (N ≥ 10^12 ∧ N < 10^13) ∧  -- 13-digit integer
  (N % 2^13 = 0) ∧           -- divisible by 2^13
  (∀ d, d ∈ digits N → d = 8 ∨ d = 9) ∧  -- digits are 8 or 9
  N = 8888888888888 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteen_digit_multiple_of_2_13_l434_43418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_identical_digits_l434_43449

theorem consecutive_sum_identical_digits : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0) ∧ 
    (Finset.card S = 2) ∧
    (∀ n ∈ S, 
      ∃ d : ℕ, d ∈ Finset.range 10 ∧ 
      n * (n + 1) / 2 = 11 * d ∧
      1 ≤ d ∧ d ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_identical_digits_l434_43449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_intervals_l434_43436

/-- The function f(x) = -1/2 x² + 13/2 -/
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 13/2

/-- Theorem stating the intervals [a, b] where f has min value 2a and max value 2b -/
theorem f_min_max_intervals :
  ∀ a b : ℝ, a < b →
    (∀ x ∈ Set.Icc a b, 2*a ≤ f x ∧ f x ≤ 2*b) ∧
    (∃ x ∈ Set.Icc a b, f x = 2*a) ∧
    (∃ x ∈ Set.Icc a b, f x = 2*b) →
    ((a = 1 ∧ b = 3) ∨ (a = -2 - Real.sqrt 17 ∧ b = 13/4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_intervals_l434_43436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_nonzero_terms_l434_43426

/-- The expansion of (x-5)(3x^2-2x+8) - 2(x^3 + 3x^2 - 4x) results in a polynomial with exactly 4 nonzero terms -/
theorem expansion_nonzero_terms :
  let p : Polynomial ℚ := X - 5
  let q : Polynomial ℚ := 3 * X^2 - 2 * X + 8
  let r : Polynomial ℚ := X^3 + 3 * X^2 - 4 * X
  let expansion := p * q - 2 * r
  (expansion.support.filter (λ n => expansion.coeff n ≠ 0)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_nonzero_terms_l434_43426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_good_point_l434_43445

/-- A point in the circular arrangement -/
structure Point where
  value : Int
  deriving Repr

/-- The circular arrangement of points -/
def CircularArrangement := List Point

/-- Check if a point is good in the given arrangement -/
def isGoodPoint (arrangement : CircularArrangement) (startIndex : Nat) : Prop :=
  ∀ (direction : Bool) (distance : Nat),
    let sequence := if direction
                    then List.drop startIndex arrangement ++ List.take startIndex arrangement
                    else List.reverse (List.drop startIndex arrangement ++ List.take startIndex arrangement)
    (List.foldl (λ sum p => sum + p.value) 0 (List.take distance sequence)) > 0

/-- The main theorem -/
theorem existence_of_good_point
  (arrangement : CircularArrangement)
  (total_points : arrangement.length = 1985)
  (negative_points : (arrangement.filter (λ p => p.value = -1)).length < 662) :
  ∃ (i : Nat), i < arrangement.length ∧ isGoodPoint arrangement i :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_good_point_l434_43445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_15_percent_l434_43491

/-- Calculates the profit percentage given the selling price and cost price -/
noncomputable def profit_percentage (selling_price cost_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that the profit percentage is 15% for the given selling price and cost price -/
theorem profit_percentage_is_15_percent :
  let selling_price : ℝ := 100
  let cost_price : ℝ := 86.95652173913044
  profit_percentage selling_price cost_price = 15 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_15_percent_l434_43491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_difference_per_sq_inch_l434_43401

-- Define the properties of the TVs
noncomputable def first_tv_width : ℝ := 24
noncomputable def first_tv_height : ℝ := 16
noncomputable def first_tv_cost : ℝ := 672

noncomputable def new_tv_width : ℝ := 48
noncomputable def new_tv_height : ℝ := 32
noncomputable def new_tv_cost : ℝ := 1152

-- Calculate areas
noncomputable def first_tv_area : ℝ := first_tv_width * first_tv_height
noncomputable def new_tv_area : ℝ := new_tv_width * new_tv_height

-- Calculate cost per square inch
noncomputable def first_tv_cost_per_sq_inch : ℝ := first_tv_cost / first_tv_area
noncomputable def new_tv_cost_per_sq_inch : ℝ := new_tv_cost / new_tv_area

-- Theorem to prove
theorem cost_difference_per_sq_inch : 
  first_tv_cost_per_sq_inch - new_tv_cost_per_sq_inch = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_difference_per_sq_inch_l434_43401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dormitory_to_city_distance_l434_43425

/-- The distance from the dormitory to the city in kilometers -/
noncomputable def total_distance : ℝ := 24

/-- The fraction of the total distance traveled by foot -/
noncomputable def foot_fraction : ℝ := 1/2

/-- The fraction of the total distance traveled by bus -/
noncomputable def bus_fraction : ℝ := 1/4

/-- The distance traveled by car in kilometers -/
noncomputable def car_distance : ℝ := 6

theorem dormitory_to_city_distance :
  foot_fraction * total_distance + bus_fraction * total_distance + car_distance = total_distance := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dormitory_to_city_distance_l434_43425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l434_43466

theorem triangle_third_side_length (a b c : ℝ) : 
  a = 5 ∧ b = 8 ∧ c = 6 → 
  a + b > c ∧ b + c > a ∧ a + c > b :=
by
  intro h
  have ha : a = 5 := h.left
  have hb : b = 8 := h.right.left
  have hc : c = 6 := h.right.right
  rw [ha, hb, hc]
  simp
  exact ⟨by norm_num, by norm_num, by norm_num⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l434_43466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ellipse_perimeter_l434_43440

/-- Given a rectangle EFGH and an ellipse passing through E and G with foci at F and H,
    this theorem proves that if the rectangle has area 4032 and the ellipse has area 4032π,
    then the perimeter of the rectangle is 8√2016. -/
theorem rectangle_ellipse_perimeter (E F G H : ℝ × ℝ) :
  let rectangle_area := 4032
  let ellipse_area := 4032 * Real.pi
  rectangle_area = abs ((F.1 - E.1) * (G.2 - E.2)) →
  ellipse_area = Real.pi * (dist E G / 2) * Real.sqrt ((dist E G / 2)^2 - (dist F H / 2)^2) →
  dist E F + dist F G = dist E G →
  dist E H + dist H G = dist E G →
  2 * (dist E F + dist F G) = 8 * Real.sqrt 2016 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ellipse_perimeter_l434_43440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_problem_l434_43413

-- Define the ellipse G
def ellipse_G (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define our custom circle (to avoid conflict with existing 'circle' definition)
def our_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def line_l (x y m : ℝ) : Prop := y = x + m

-- Define the isosceles triangle condition
def isosceles_triangle (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 + x2) / 2 = -2 ∧ (y1 + y2) / 2 = 1/3

theorem ellipse_and_line_problem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y, ellipse_G x y a b → our_circle x y) →
  (a^2 = 8 ∧ b^2 = 4) ∧
  ∃ m, line_l (-3) 2 m ∧
       ∀ x1 y1 x2 y2, 
         ellipse_G x1 y1 a b → 
         ellipse_G x2 y2 a b → 
         line_l x1 y1 m → 
         line_l x2 y2 m → 
         isosceles_triangle x1 y1 x2 y2 → 
         m = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_problem_l434_43413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_exponential_function_l434_43499

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := 2^x + c

theorem point_on_exponential_function (c : ℝ) :
  f c 2 = 5 → c = 1 := by
  intro h
  simp [f] at h
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_exponential_function_l434_43499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_is_twelve_l434_43410

/-- Represents the dimensions and constraints of the box packaging problem. -/
structure BoxProblem where
  base_length : ℚ
  base_width : ℚ
  total_volume : ℚ
  cost_per_box : ℚ
  min_spend : ℚ

/-- Calculates the maximum height of the box given the problem constraints. -/
noncomputable def max_box_height (p : BoxProblem) : ℚ :=
  min (p.total_volume / (p.base_length * p.base_width)) 
      ((p.cost_per_box * p.total_volume) / (p.base_length * p.base_width * p.min_spend))

/-- Theorem stating that the maximum box height for the given problem is 12 inches. -/
theorem box_height_is_twelve : 
  let p : BoxProblem := {
    base_length := 20,
    base_width := 20,
    total_volume := 2400000,
    cost_per_box := 1/2,
    min_spend := 250
  }
  max_box_height p = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_height_is_twelve_l434_43410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_shift_max_value_of_function_l434_43405

-- Problem 1
theorem function_shift (f : ℝ → ℝ) :
  (∀ x, f (x + 1) = x^2 - 2*x) → (∀ x, f x = x^2 - 4*x + 3) := by
  sorry

-- Problem 2
theorem max_value_of_function :
  ∃ M, M = 4/3 ∧ ∀ x, 1 / (1 - x*(1-x)) ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_shift_max_value_of_function_l434_43405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petrov_class_vowel_probability_l434_43474

/-- Represents a class of students with unique initials -/
structure PetrovClass where
  total_students : ℕ
  vowels : Finset Char
  has_unique_initials : Bool

/-- The probability of selecting a student with vowel initials -/
def vowel_initial_probability (c : PetrovClass) : ℚ :=
  c.vowels.card / c.total_students

/-- Mr. Petrov's history class -/
def petrov_class : PetrovClass where
  total_students := 30
  vowels := {'A', 'E', 'I', 'O', 'U', 'X'}
  has_unique_initials := true

/-- Theorem stating the probability of selecting a student with vowel initials in Mr. Petrov's class -/
theorem petrov_class_vowel_probability :
  vowel_initial_probability petrov_class = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petrov_class_vowel_probability_l434_43474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_six_digit_divisible_tens_digit_l434_43470

theorem smallest_six_digit_divisible_tens_digit : ∃ n : ℕ,
  (100000 ≤ n) ∧ 
  (n < 1000000) ∧
  (∀ k : ℕ, k ∈ ({10, 11, 12, 13, 14, 15} : Set ℕ) → n % k = 0) ∧
  (∀ m : ℕ, 100000 ≤ m ∧ m < n → ∃ k : ℕ, k ∈ ({10, 11, 12, 13, 14, 15} : Set ℕ) ∧ m % k ≠ 0) ∧
  ((n / 10) % 10 = 2) := by
  sorry

#check smallest_six_digit_divisible_tens_digit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_six_digit_divisible_tens_digit_l434_43470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_average_speed_l434_43480

/-- Represents James' bike ride --/
structure BikeRide where
  total_time : ℝ
  second_hour_distance : ℝ
  second_hour_increase : ℝ
  third_hour_increase : ℝ
  wind_resistance_increase : ℝ
  elevation_gain : ℝ

/-- Calculates the average speed of the bike ride --/
noncomputable def average_speed (ride : BikeRide) : ℝ :=
  let first_hour_distance := ride.second_hour_distance / (1 + ride.second_hour_increase)
  let third_hour_distance := ride.second_hour_distance * (1 + ride.third_hour_increase)
  let total_distance := first_hour_distance + ride.second_hour_distance + third_hour_distance
  total_distance / ride.total_time

/-- Theorem stating that James' average speed is 18.5 miles per hour --/
theorem james_average_speed :
  let ride : BikeRide := {
    total_time := 3,
    second_hour_distance := 18,
    second_hour_increase := 0.2,
    third_hour_increase := 0.25,
    wind_resistance_increase := 0.1,
    elevation_gain := 500
  }
  average_speed ride = 18.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_average_speed_l434_43480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l434_43447

noncomputable def f (a b x : ℝ) : ℝ := (x + a) / (x^2 + b*x + 1)

theorem odd_function_properties (a b : ℝ) 
  (h_odd : ∀ x, f a b (-x) = -(f a b x)) :
  (a = 0 ∧ b = 0) ∧ 
  (∀ x y, 1 < x → x < y → f 0 0 x > f 0 0 y) ∧
  (∀ k, k < 0 → (∀ t, f 0 0 (t^2 - 2*t + 3) + f 0 0 (k - 1) < 0) → -1 < k) := by
  sorry

#check odd_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l434_43447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_distances_is_two_l434_43422

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 3 / 2) * t, 1 + (1 / 2) * t)

-- Define the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem product_of_distances_is_two :
  ∃ (t1 t2 : ℝ),
    on_circle (line_l t1).1 (line_l t1).2 ∧
    on_circle (line_l t2).1 (line_l t2).2 ∧
    t1 ≠ t2 ∧
    (distance point_P (line_l t1)) * (distance point_P (line_l t2)) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_distances_is_two_l434_43422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l434_43442

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the tangent line condition
def is_tangent_line (l : ℝ → ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀ > 0 ∧ l x₀ = f x₀ ∧ (∀ x, x ≠ x₀ → l x ≠ f x)

-- Theorem statement
theorem tangent_line_equation :
  ∀ l : ℝ → ℝ,
  (l 0 = -1) →  -- Line passes through (0, -1)
  (is_tangent_line l f) →  -- Line is tangent to f
  (∀ x y, l x = y ↔ x - y - 1 = 0) :=  -- Equation of line l
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l434_43442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l434_43458

open Real

-- Define the circle O
def circle_O (ρ θ : ℝ) : Prop := ρ = cos θ + sin θ

-- Define the line l
def line_l (ρ θ : ℝ) : Prop := ρ * sin (θ - π/4) = sqrt 2/2

-- Define the constraints
def constraints (ρ θ : ℝ) : Prop := ρ ≥ 0 ∧ 0 ≤ θ ∧ θ ≤ 2*π

-- Theorem statement
theorem intersection_point :
  ∀ ρ θ : ℝ,
  circle_O ρ θ →
  line_l ρ θ →
  constraints ρ θ →
  0 < θ →
  θ < π →
  ρ = 1 ∧ θ = π/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l434_43458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dual_expr_product_fraction_value_equation_solution_l434_43492

-- Define the dual expressions
noncomputable def dual_expr (a b : ℝ) : ℝ × ℝ := (Real.sqrt a + Real.sqrt b, Real.sqrt a - Real.sqrt b)

-- Theorem 1: (2+√3)(2-√3) = 1
theorem dual_expr_product : (2 + Real.sqrt 3) * (2 - Real.sqrt 3) = 1 := by sorry

-- Define x and y
noncomputable def x : ℝ := 1 / (Real.sqrt 5 - 2)
noncomputable def y : ℝ := 1 / (Real.sqrt 5 + 2)

-- Theorem 2: (x-y)/(x²y+xy²) = 2√5/5
theorem fraction_value : (x - y) / (x^2 * y + x * y^2) = (2 * Real.sqrt 5) / 5 := by sorry

-- Theorem 3: The solution to √(24-x) - √(8-x) = 2 is x = -1
theorem equation_solution : ∃ (x : ℝ), Real.sqrt (24 - x) - Real.sqrt (8 - x) = 2 ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dual_expr_product_fraction_value_equation_solution_l434_43492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l434_43423

/-- The function f(x) = sin(ωx + π/6) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

/-- The theorem stating that ω = 1/3 is the unique value satisfying the given conditions -/
theorem omega_value :
  ∃! ω : ℝ,
    (∀ x : ℝ, f ω x ≤ f ω Real.pi) ∧
    (∀ x y : ℝ, -Real.pi/6 ≤ x ∧ x < y ∧ y ≤ Real.pi/6 → f ω x < f ω y) ∧
    ω = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l434_43423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_rotation_volume_l434_43460

/-- A function representing the volume of a cube rotated around its face diagonal -/
noncomputable def VolumeOfRotatedCube (a : ℝ) : ℝ :=
  2 * Real.pi * ∫ x in (0)..(a / Real.sqrt 2), (x^2 + a^2)

/-- The volume of a solid obtained by rotating a cube around its face diagonal -/
theorem cube_rotation_volume (a : ℝ) (ha : a > 0) :
  ∃ V : ℝ, V = (7 * Real.pi * a^3 * Real.sqrt 2) / 6 ∧
  V = VolumeOfRotatedCube a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_rotation_volume_l434_43460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_number_l434_43483

theorem imaginary_part_of_complex_number (b : ℝ) (z : ℂ) : 
  z = 1 + b * Complex.I → Complex.abs z = 2 → b = Real.sqrt 3 ∨ b = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_number_l434_43483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_volume_is_46pi_l434_43486

/-- A shaded region in the first quadrant with specific properties -/
structure ShadedRegion where
  rectangle1_height : ℝ
  rectangle1_width : ℝ
  rectangle2_height : ℝ
  rectangle2_width : ℝ
  triangle_base : ℝ
  triangle_height : ℝ

/-- The volume of the solid formed by rotating the shaded region about the x-axis -/
noncomputable def rotated_volume (region : ShadedRegion) : ℝ :=
  let cylinder1_volume := Real.pi * region.rectangle1_height^2 * region.rectangle1_width
  let cylinder2_volume := Real.pi * region.rectangle2_height^2 * region.rectangle2_width
  let cone_volume := (1/3) * Real.pi * region.triangle_height^2 * region.triangle_base
  cylinder1_volume + cylinder2_volume + cone_volume

/-- Theorem stating that the volume of the rotated solid is 46π cubic units -/
theorem rotated_volume_is_46pi (region : ShadedRegion) 
  (h1 : region.rectangle1_height = 4)
  (h2 : region.rectangle1_width = 1)
  (h3 : region.rectangle2_height = 3)
  (h4 : region.rectangle2_width = 3)
  (h5 : region.triangle_base = 1)
  (h6 : region.triangle_height = 3) :
  rotated_volume region = 46 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_volume_is_46pi_l434_43486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l434_43461

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := u.1 * v.1 + u.2 * v.2
  let norm_sq := u.1 * u.1 + u.2 * u.2
  (dot / norm_sq * u.1, dot / norm_sq * u.2)

theorem vector_satisfies_projections : 
  let v : ℝ × ℝ := (8, 6)
  proj (3, 2) v = (45/13, 30/13) ∧ 
  proj (1, 4) v = (32/17, 128/17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l434_43461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l434_43464

noncomputable section

-- Define the points
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (-2, -1)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def OC : ℝ × ℝ := C

-- Define dot product
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector addition
def add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define scalar multiplication
def scale (t : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (t * v.1, t * v.2)

theorem vector_problem :
  (magnitude (add AB AC) = 2 * Real.sqrt 10) ∧
  (∃ t : ℝ, t = -11/5 ∧ dot (add AB (scale (-t) OC)) OC = 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l434_43464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l434_43454

def point1 : ℝ × ℝ × ℝ := (4, -2, 3)
def point2 (lambda : ℝ) : ℝ × ℝ × ℝ := (2, 4, -7 + lambda)

theorem distance_between_points (lambda : ℝ) :
  Real.sqrt ((point2 lambda).1 - point1.1)^2 + ((point2 lambda).2.1 - point1.2.1)^2 + ((point2 lambda).2.2 - point1.2.2)^2 =
  Real.sqrt (40 + (lambda - 10)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l434_43454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_set_l434_43429

-- Define the four sets
def set1 : Set ℝ := {x | x = 1}
def set2 : Set ℝ := {y | (y - 1)^2 = 0}
def set3 : Set (Set ℝ) := {{x | x = 1}}
def set4 : Set ℝ := {1}

-- Theorem statement
theorem different_set : 
  (set1 = set2) ∧ (set2 = set4) ∧ (set1 = set4) ∧ 
  (set3 ≠ {set1}) ∧ (set3 ≠ {set2}) ∧ (set3 ≠ {set4}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_set_l434_43429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_proof_l434_43487

/-- Given vectors a, b, c in ℝ² and l ∈ ℝ satisfying certain conditions, prove that l + m = 6 -/
theorem vector_equation_proof (a b c : ℝ × ℝ) (l : ℝ) (m : ℝ) 
  (ha : a = (2, 1)) 
  (hb : b = (3, 4)) 
  (hc : c = (1, m)) 
  (heq : a + b = l • c) : 
  l + m = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_proof_l434_43487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cleaning_time_per_week_l434_43408

-- Define the cleaning times and room sizes
noncomputable def richard_time : ℝ := 22
noncomputable def richard_size : ℝ := 120
noncomputable def cory_size : ℝ := 140
noncomputable def blake_size : ℝ := 160
noncomputable def evie_size : ℝ := 180

-- Define the relationships between cleaning times
noncomputable def cory_time : ℝ := richard_time + 3
noncomputable def blake_time : ℝ := cory_time - 4
noncomputable def evie_time : ℝ := (richard_time + blake_time) / 2

-- Define the number of times they clean per week
def cleanings_per_week : ℕ := 2

-- Theorem statement
theorem total_cleaning_time_per_week :
  (richard_time + cory_time + blake_time + evie_time) * (cleanings_per_week : ℝ) = 179 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cleaning_time_per_week_l434_43408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l434_43476

/-- Parabola definition -/
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Point on the parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : Parabola x y
  not_origin : x ≠ 0 ∨ y ≠ 0

/-- Line parallel to PQ that intersects the parabola at exactly one point -/
structure TangentLine (P : PointOnParabola) where
  b : ℝ
  eq : ℝ → ℝ → Prop
  tangent : ∃! E : ℝ × ℝ, Parabola E.1 E.2 ∧ eq E.1 E.2

/-- Helper function for triangle area -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Helper function to check if a point is on a line -/
def on_line (A B C : ℝ × ℝ) : Prop := sorry

/-- The theorem to be proved -/
theorem parabola_properties (P : PointOnParabola) (l : TangentLine P) :
  (∃ (min_area : ℝ), min_area = 2 ∧ 
    ∀ E : ℝ × ℝ, Parabola E.1 E.2 → l.eq E.1 E.2 → 
      area_triangle (0, 0) (P.x, P.y) E ≥ min_area) ∧
  (∀ E : ℝ × ℝ, Parabola E.1 E.2 → l.eq E.1 E.2 → 
    on_line (P.x, P.y) E Focus) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l434_43476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_hair_growth_rate_l434_43467

/-- Calculates the monthly hair growth rate given initial length, current length, and time period in years. -/
noncomputable def monthly_hair_growth_rate (initial_length current_length : ℝ) (years : ℕ) : ℝ :=
  let total_growth := current_length - initial_length
  let months := (years * 12 : ℝ)
  total_growth / months

/-- Theorem stating that given the specified conditions, the monthly hair growth rate is 0.5 inches. -/
theorem bob_hair_growth_rate :
  let initial_length : ℝ := 6
  let current_length : ℝ := 36
  let years : ℕ := 5
  monthly_hair_growth_rate initial_length current_length years = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_hair_growth_rate_l434_43467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_yard_conversion_l434_43473

/-- Conversion factor from yards to meters -/
def yard_to_meter : ℝ := 0.9144

/-- The number of cubic meters in one cubic yard -/
def cubic_yard_to_cubic_meter : ℝ := yard_to_meter ^ 3

/-- Theorem stating that the number of cubic meters in one cubic yard is approximately 0.7636 -/
theorem cubic_yard_conversion :
  abs (cubic_yard_to_cubic_meter - 0.7636) < 0.0001 := by
  sorry

#eval cubic_yard_to_cubic_meter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_yard_conversion_l434_43473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_people_count_over_three_days_l434_43497

/-- The total number of people counted over three days given specific conditions -/
theorem people_count_over_three_days (r : ℝ) (h_r_pos : 0 < r) (h_r_lt_one : r < 1) : 
  (2 * 500) + 500 + (r * 500) = 1500 + 500 * r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_people_count_over_three_days_l434_43497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corridor_width_2_functions_l434_43490

-- Define the functions
def f1 (x : ℝ) : ℝ := x^2

noncomputable def f2 (x : ℝ) : ℝ := Real.sqrt (4 - x^2)

noncomputable def f3 (x : ℝ) : ℝ := if x ≤ 0 then Real.exp x - 1 else 1 - Real.exp (-x)

noncomputable def f4 (x : ℝ) : ℝ := 2 / x

-- Define the domain conditions
def D1 : Set ℝ := { x | x ≥ 0 }
def D2 : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }
def D3 : Set ℝ := Set.univ
def D4 : Set ℝ := { x | |x| ≥ 4 }

-- Define what it means for a function to have a corridor of width 2
def has_corridor_of_width_2 (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∃ (k m₁ m₂ : ℝ), ∀ x ∈ D, k * x + m₁ ≤ f x ∧ f x ≤ k * x + m₂ ∧ m₂ - m₁ = 2

-- State the theorem
theorem corridor_width_2_functions :
  ¬(has_corridor_of_width_2 f1 D1) ∧
  (has_corridor_of_width_2 f2 D2) ∧
  (has_corridor_of_width_2 f3 D3) ∧
  (has_corridor_of_width_2 f4 D4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corridor_width_2_functions_l434_43490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_l434_43471

open Real

/-- Curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 4 = 0

/-- Curve C₂ in Cartesian coordinates -/
def C₂ (x y : ℝ) : Prop := ∃ θ : ℝ, x = cos θ ∧ y = 1 + sin θ

/-- Curve C₃ in Cartesian coordinates -/
def C₃ (x y : ℝ) (α : ℝ) : Prop := ∃ t : ℝ, t > 0 ∧ x = t * cos α ∧ y = t * sin α

/-- Point A is the intersection of C₁ and C₃ -/
noncomputable def A (α : ℝ) : ℝ × ℝ := 
  (4 * cos α / (Real.sqrt 3 * cos α + sin α), 4 * sin α / (Real.sqrt 3 * cos α + sin α))

/-- Point B is the intersection of C₂ and C₃ -/
noncomputable def B (α : ℝ) : ℝ × ℝ := (2 * sin α * cos α, 2 * sin α^2)

/-- The ratio |OB|/|OA| -/
noncomputable def ratio (α : ℝ) : ℝ := 
  let (xb, yb) := B α
  let (xa, ya) := A α
  Real.sqrt (xb^2 + yb^2) / Real.sqrt (xa^2 + ya^2)

theorem max_ratio :
  ∀ α : ℝ, 0 < α → α < π/2 → ratio α ≤ 3/4 ∧ 
  (ratio (π/3) = 3/4 ∧ ∀ β : ℝ, 0 < β → β < π/2 → β ≠ π/3 → ratio β < 3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_l434_43471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_1_to_9999_l434_43475

/-- Sum of digits function for a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits for a range of natural numbers -/
def sumOfDigitsRange (start finish : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits from numbers 1 to 9999 is 194445 -/
theorem sum_of_digits_1_to_9999 : sumOfDigitsRange 1 9999 = 194445 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_1_to_9999_l434_43475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_worker_time_on_DE_l434_43428

/-- Represents a worker paving a path -/
structure Worker where
  speed : ℝ
  path_length : ℝ

/-- Represents the park scenario -/
structure ParkScenario where
  worker1 : Worker
  worker2 : Worker
  total_time : ℝ
  speed_ratio : ℝ

/-- Calculates the time spent on a specific segment -/
noncomputable def time_on_segment (w : Worker) (segment_length : ℝ) : ℝ :=
  segment_length / w.speed

/-- Main theorem: The second worker spends 45 minutes on segment D-E -/
theorem second_worker_time_on_DE (park : ParkScenario) 
  (h1 : park.total_time = 9)
  (h2 : park.speed_ratio = 1.2)
  (h3 : park.worker1.path_length = park.worker1.speed * park.total_time)
  (h4 : park.worker2.path_length = park.worker2.speed * park.total_time)
  (h5 : park.worker2.speed = park.speed_ratio * park.worker1.speed)
  (h6 : park.worker2.path_length = 1.2 * park.worker1.path_length)
  (h7 : ∃ (de : ℝ), park.worker2.path_length = park.worker1.path_length + 2 * de) :
  ∃ (de : ℝ), time_on_segment park.worker2 de = 45 / 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_worker_time_on_DE_l434_43428
