import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shekar_weighted_average_l568_56881

-- Define the scores and weightages
noncomputable def scores : List ℝ := [76, 65, 82, 67, 95, 89, 79]
noncomputable def weightages : List ℝ := [0.10, 0.15, 0.10, 0.15, 0.20, 0.10, 0.20]

-- Define the weighted average calculation function
noncomputable def weighted_average (scores : List ℝ) (weightages : List ℝ) : ℝ :=
  (List.sum (List.zipWith (·*·) scores weightages)) / (List.sum weightages)

-- Theorem statement
theorem shekar_weighted_average :
  weighted_average scores weightages = 79.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shekar_weighted_average_l568_56881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l568_56840

/-- A hyperbola with focus on the x-axis and asymptotes y = ±(√5/2)x has eccentricity 3/2 -/
theorem hyperbola_eccentricity (C : Set (ℝ × ℝ)) (a b : ℝ) :
  (∃ (f : ℝ), (f, 0) ∈ C) →  -- focus on x-axis
  (∀ (x y : ℝ), y = (Real.sqrt 5 / 2) * x ∨ y = -(Real.sqrt 5 / 2) * x → (x, y) ∈ C) →  -- asymptotes
  (b / a = Real.sqrt 5 / 2) →  -- relation between semi-axes derived from asymptotes
  Real.sqrt (1 + (b/a)^2) = 3/2  -- eccentricity formula
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l568_56840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specialSystem_fraction_equivalence_l568_56874

-- Define a custom numeral system
structure CustomNumeralSystem where
  base : ℕ
  toStandard : ℕ → ℕ
  fromStandard : ℕ → ℕ

-- Define the specific numeral system where 1/4 of 20 equals 6
def specialSystem : CustomNumeralSystem :=
  { base := 12,
    toStandard := λ n ↦ n * 12 / 10,
    fromStandard := λ n ↦ n * 10 / 12 }

-- Define the operation of taking a fraction of a number in the custom system
def fractionOf (n : ℕ) (d : ℕ) (x : ℕ) (sys : CustomNumeralSystem) : ℚ :=
  (sys.toStandard x * n) / (d * sys.base)

-- Theorem statement
theorem specialSystem_fraction_equivalence :
  fractionOf 1 4 20 specialSystem = 6 ∧
  fractionOf 1 5 10 specialSystem = 29 / 12 := by
  sorry

-- Note: 29/12 is equivalent to 2 5/12 in standard decimal notation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specialSystem_fraction_equivalence_l568_56874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_total_length_l568_56817

-- Define the grid
def grid_width : ℕ := 12
def grid_height : ℕ := 4

-- Define the segments of X
noncomputable def x_diagonal_length : ℝ := Real.sqrt 8

-- Define the segments of Y
def y_vertical_length : ℝ := 2
noncomputable def y_diagonal_length : ℝ := Real.sqrt 2

-- Define the segments of Z
def z_horizontal_length : ℝ := 3
def z_vertical_length : ℝ := 2

-- Theorem statement
theorem xyz_total_length :
  2 * x_diagonal_length + y_vertical_length + y_diagonal_length +
  2 * z_horizontal_length + z_vertical_length = 10 + 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_total_length_l568_56817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_playground_fundraising_l568_56860

/-- Represents the fundraising results and calculations for Tree Elementary School's new playground --/
theorem playground_fundraising (johnson_class : ℝ) (sutton_class : ℝ) (rollin_class : ℝ) 
  (edward_class : ℝ) (andrea_class : ℝ) (thompson_class : ℝ) :
  let total_raised := johnson_class + sutton_class + rollin_class + edward_class + andrea_class + thompson_class
  let admin_fee := total_raised * 0.02
  let maintenance_expense := total_raised * 0.05
  let school_band_allocation := total_raised * 0.07
  let total_deductions := admin_fee + maintenance_expense + school_band_allocation
  let amount_left := total_raised - total_deductions
  amount_left = 28681 :=
by
  have h1 : johnson_class = 2300 := by sorry
  have h2 : sutton_class = johnson_class / 2 := by sorry
  have h3 : rollin_class = sutton_class * 8 := by sorry
  have h4 : rollin_class = (johnson_class + sutton_class + rollin_class) / 3 := by sorry
  have h5 : edward_class = rollin_class * 0.75 := by sorry
  have h6 : andrea_class = edward_class * 1.5 := by sorry
  have h7 : thompson_class = johnson_class * 1.5 := by sorry
  sorry

#check playground_fundraising

end NUMINAMATH_CALUDE_ERRORFEEDBACK_playground_fundraising_l568_56860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_range_of_f_l568_56814

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x + 2 * Real.sin x * Real.cos x + 2

-- Statement 1: Negation of the universal quantifier
theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

-- Statement 2: Range of the function f
theorem range_of_f :
  Set.range f = Set.Icc (3/4) (3 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_range_of_f_l568_56814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_neither_arithmetic_nor_geometric_l568_56819

/-- A sequence {a_n} with sum of first n terms S_n = a^n - 1 -/
def S (a : ℝ) (n : ℕ) : ℝ := a^n - 1

/-- The n-th term of the sequence -/
def a_n (a : ℝ) (n : ℕ) : ℝ := S a n - S a (n-1)

/-- Arithmetic sequence general term -/
def isArithmetic (f : ℕ → ℝ) : Prop :=
  ∃ d c : ℝ, ∀ n : ℕ, f n = d * n + c

/-- Geometric sequence general term -/
def isGeometric (f : ℕ → ℝ) : Prop :=
  ∃ a r : ℝ, ∀ n : ℕ, f n = a * r^(n-1)

theorem sequence_neither_arithmetic_nor_geometric (a : ℝ) (h : a ≠ 0) :
  ¬ isArithmetic (a_n a) ∧ ¬ isGeometric (a_n a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_neither_arithmetic_nor_geometric_l568_56819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_theorem_l568_56841

-- Define the line l
def line (k : ℝ) (x : ℝ) : ℝ := k * (x - 1) + 2

-- Define the circle C
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 5

-- Define the condition for the shortest chord
def shortest_chord_condition (k x y : ℝ) : Prop :=
  circle_equation x y ∧ y = line k x ∧ k * ((y - 1) / (x - 2)) = -1

-- Theorem statement
theorem shortest_chord_theorem (k : ℝ) :
  (∃ x y : ℝ, shortest_chord_condition k x y) ↔
  (∀ x y : ℝ, circle_equation x y ∧ y = line k x → k * ((y - 1) / (x - 2)) = -1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_theorem_l568_56841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_in_third_quadrant_l568_56885

noncomputable def f (α : Real) : Real :=
  (Real.cos (Real.pi/2 + α) * Real.cos (2*Real.pi - α) * Real.sin (-α + 3*Real.pi/2)) /
  (Real.sin (-Real.pi - α) * Real.sin (3*Real.pi/2 + α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by
  sorry

theorem f_value_in_third_quadrant (α : Real) 
  (h1 : Real.pi < α ∧ α < 3*Real.pi/2) -- α is in the third quadrant
  (h2 : Real.cos (α - 3*Real.pi/2) = 1/5) :
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_in_third_quadrant_l568_56885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l568_56880

/-- Calculates the final amount of an investment with compound interest -/
def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_growth : 
  let principal : ℝ := 15000
  let rate : ℝ := 0.045
  let time : ℕ := 7
  let final_amount := compoundInterest principal rate time
  ⌊final_amount⌋ = 20144 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l568_56880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l568_56827

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-- The foci of a hyperbola -/
def Hyperbola.foci (h : Hyperbola a b) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- A point on the right branch of a hyperbola -/
def Hyperbola.rightBranchPoint (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The distance from a point to a line in ℝ² -/
def distanceToLine (p q r : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
noncomputable def Hyperbola.eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The length of the real axis of a hyperbola -/
def Hyperbola.realAxisLength (h : Hyperbola a b) : ℝ := sorry

theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) :
  let (f₁, f₂) := h.foci
  let p := h.rightBranchPoint
  distance p f₂ = distance f₁ f₂ →
  distanceToLine f₂ p f₁ = h.realAxisLength →
  h.eccentricity = 5/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l568_56827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l568_56830

-- Define the triangle ABC
structure Triangle where
  A : Real  -- Angle A
  B : Real  -- Angle B
  C : Real  -- Angle C
  AB : Real -- Side AB
  BC : Real -- Side BC
  AC : Real -- Side AC

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.AB > 0 ∧ t.BC > 0 ∧ t.AC > 0

-- Define the arithmetic sequence property for angles
def anglesFormArithmeticSequence (t : Triangle) : Prop :=
  2 * t.B = t.A + t.C

-- Helper function to calculate the area of a triangle
noncomputable def area (t : Triangle) : Real :=
  1/2 * t.AB * t.AC * Real.sin t.B

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : isValidTriangle t)
  (h2 : anglesFormArithmeticSequence t)
  (h3 : t.AC = 3)
  (h4 : Real.cos t.C = Real.sqrt 6 / 3) :
  t.AB = 2 ∧ 
  (∀ s : Triangle, isValidTriangle s → anglesFormArithmeticSequence s → s.AC = 3 → 
    area s ≤ 9 * Real.sqrt 3 / 4) ∧
  ∃ s : Triangle, isValidTriangle s ∧ anglesFormArithmeticSequence s ∧ s.AC = 3 ∧ 
    area s = 9 * Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l568_56830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l568_56872

/-- Represents the side lengths of seven squares in ascending order -/
structure SquareSides where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+
  g : ℕ+

/-- Conditions for the square sides -/
def satisfies_conditions (s : SquareSides) : Prop :=
  s.a + s.b + s.c = s.d ∧
  s.d + s.e = s.g ∧
  s.b + s.c = s.f ∧
  s.c + s.f = s.g

/-- The width of the rectangle -/
def width (s : SquareSides) : ℕ+ :=
  s.a + s.b + s.g

/-- The height of the rectangle -/
def rectangle_height (s : SquareSides) : ℕ+ :=
  s.d + s.e

/-- Theorem stating that the perimeter of the rectangle is 40 -/
theorem rectangle_perimeter (s : SquareSides) 
  (h_conditions : satisfies_conditions s)
  (h_coprime : Nat.Coprime (width s).val (rectangle_height s).val) : 
  2 * ((width s).val + (rectangle_height s).val) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l568_56872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_positive_range_l568_56886

-- Define the properties of the function f
def is_valid_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (f (-1) = 0) ∧
  (∀ x > 0, x * (deriv f x) - f x < 0)

-- State the theorem
theorem function_positive_range (f : ℝ → ℝ) (hf : is_valid_function f) :
  {x : ℝ | f x > 0} = Set.Ioi (-1) \ {-1} ∪ Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_positive_range_l568_56886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_l_shape_l568_56820

/-- Represents a chessboard as a function from coordinates to a boolean value (shaded or not) -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Counts the number of shaded squares on the chessboard -/
def count_shaded (board : Chessboard) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 8)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 8)) fun j =>
      if board i j then 1 else 0)

/-- Checks if a 2x2 subgrid has at least 3 unshaded cells -/
def has_three_unshaded (board : Chessboard) (i j : Fin 7) : Bool :=
  let subgrid := [board i j, board i j.succ, board i.succ j, board i.succ j.succ]
  (subgrid.filter (· = false)).length ≥ 3

/-- Main theorem: If a chessboard has 31 shaded squares, 
    then there exists a 2x2 subgrid with at least 3 unshaded cells -/
theorem chessboard_l_shape (board : Chessboard) 
    (h : count_shaded board = 31) :
    ∃ i j : Fin 7, has_three_unshaded board i j := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_l_shape_l568_56820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l568_56897

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l568_56897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_l568_56805

open Real

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + cos (x - π / 3) ^ 2 - 1

noncomputable def g (x : ℝ) : ℝ := f (x - π / 3)

theorem f_properties :
  -- 1. Smallest positive period
  (∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧
  -- 2. Monotonic decrease intervals
  (∀ k : ℤ, ∀ x y : ℝ, k * π + π / 6 ≤ x ∧ x < y ∧ y ≤ k * π + 2 * π / 3 → f y < f x) ∧
  -- 3. Expression for g
  (∀ x : ℝ, g x = -1 / 2 * cos (2 * x)) ∧
  -- 4. Maximum value in [-π/4, π/3]
  (∀ x : ℝ, -π / 4 ≤ x ∧ x ≤ π / 3 → f x ≤ 1 / 2) ∧
  (∃ x : ℝ, -π / 4 ≤ x ∧ x ≤ π / 3 ∧ f x = 1 / 2) ∧
  -- 5. Minimum value in [-π/4, π/3]
  (∀ x : ℝ, -π / 4 ≤ x ∧ x ≤ π / 3 → f x ≥ -sqrt 3 / 4) ∧
  (∃ x : ℝ, -π / 4 ≤ x ∧ x ≤ π / 3 ∧ f x = -sqrt 3 / 4) :=
by sorry

-- Separate theorem for the period
theorem f_period : ∃ T : ℝ, T = π ∧ T > 0 ∧ ∀ x, f (x + T) = f x ∧
  ∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_l568_56805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l568_56829

def a (n : ℕ+) : ℝ := 2 * (n : ℝ) - 1

def S (n : ℕ+) : ℝ := (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

noncomputable def b (n : ℕ+) : ℝ := (a n + 1) * (2 : ℝ)^(a n)

noncomputable def T (n : ℕ+) : ℝ := (Finset.range n.val).sum (λ i => b ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_properties (n : ℕ+) :
  (∀ m : ℕ+, a m > 0) ∧
  (∀ m : ℕ+, 2 * Real.sqrt (S m) = a m + 1) →
  (a n = 2 * (n : ℝ) - 1) ∧
  (T n = 4/9 + (3 * (n : ℝ) - 1)/9 * 4^((n : ℝ) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l568_56829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_of_line_m_l568_56882

/-- The directed distance from a point to a line -/
noncomputable def directedDistance (x₀ y₀ a b c : ℝ) : ℝ :=
  (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

theorem slope_range_of_line_m (a b c : ℝ) :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (3, 0)
  3 * a + c = 0 →  -- Line m passes through P(3,0)
  ∃ θ : ℝ,
    let C : ℝ × ℝ := (9 * Real.cos θ, 18 + 9 * Real.sin θ)
    (C.1)^2 + (C.2 - 18)^2 = 81 →  -- C is on the circle
    directedDistance A.1 A.2 a b c + directedDistance B.1 B.2 a b c + directedDistance C.1 C.2 a b c = 0 →
    -3 ≤ -a / b ∧ -a / b ≤ -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_of_line_m_l568_56882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l568_56873

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem smallest_positive_period_of_f :
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l568_56873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_with_root_property_l568_56852

/-- The cube root of unity -/
noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

/-- The polynomial type we're considering -/
def P (a b c : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + 2400

/-- The property that if r is a root, so is ω * r -/
def HasRootProperty (a b c : ℝ) : Prop :=
  ∀ r : ℂ, P a b c r = 0 → P a b c (ω * r) = 0

/-- The main theorem: there is exactly one polynomial with the required property -/
theorem unique_polynomial_with_root_property :
  ∃! (a b c : ℝ), HasRootProperty a b c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_with_root_property_l568_56852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l568_56816

theorem sufficient_not_necessary :
  (∀ x : ℝ, |x + 1| < 1 → x⁻¹ < (1 / 2)) ∧
  (∃ x : ℝ, x⁻¹ < (1 / 2) ∧ |x + 1| ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l568_56816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equality_l568_56865

theorem sin_cos_equality (x : ℝ) : 
  x = π / 18 → Real.sin (4 * x) * Real.sin (5 * x) = Real.cos (4 * x) * Real.cos (5 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equality_l568_56865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_functions_bisect_circle_f1_is_odd_f2_is_odd_f3_is_even_l568_56853

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the three functions
noncomputable def f1 (x : ℝ) : ℝ := x * Real.cos x
noncomputable def f2 (x : ℝ) : ℝ := Real.tan x
noncomputable def f3 (x : ℝ) : ℝ := x * Real.sin x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem stating that exactly two of the given functions are odd
theorem two_functions_bisect_circle :
  (is_odd f1 ∧ is_odd f2 ∧ ¬is_odd f3) :=
by
  sorry

-- Additional theorems to prove individual properties
theorem f1_is_odd : is_odd f1 :=
by
  sorry

theorem f2_is_odd : is_odd f2 :=
by
  sorry

theorem f3_is_even : ¬is_odd f3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_functions_bisect_circle_f1_is_odd_f2_is_odd_f3_is_even_l568_56853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_student_seat_number_l568_56812

def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (selectedNumbers : List ℕ) : Prop :=
  ∃ (step : ℕ), 
    step > 0 ∧
    selectedNumbers.length + 1 = sampleSize ∧
    ∀ (i j : ℕ), i < j → i < selectedNumbers.length → j < selectedNumbers.length →
      selectedNumbers.get ⟨j, by sorry⟩ - selectedNumbers.get ⟨i, by sorry⟩ = (j - i) * step

theorem fifth_student_seat_number 
  (totalStudents : ℕ) (sampleSize : ℕ) (selectedNumbers : List ℕ) :
  totalStudents = 55 →
  sampleSize = 5 →
  selectedNumbers = [5, 16, 27, 49] →
  systematicSample totalStudents sampleSize selectedNumbers →
  ∃ (fifthNumber : ℕ), fifthNumber = 38 ∧ 
    systematicSample totalStudents sampleSize (selectedNumbers ++ [fifthNumber]) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_student_seat_number_l568_56812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_concentric_circles_l568_56833

/-- Two concentric circles with center O -/
structure ConcentricCircles where
  center : Real × Real
  small_radius : Real
  large_radius : Real
  h_positive_small : small_radius > 0
  h_positive_large : large_radius > 0
  h_concentric : small_radius < large_radius

/-- An arc on a circle -/
structure CircleArc where
  radius : Real
  angle : Real  -- in degrees

/-- The length of an arc -/
noncomputable def arc_length (arc : CircleArc) : Real :=
  (arc.angle / 360) * (2 * Real.pi * arc.radius)

theorem area_ratio_of_concentric_circles
  (circles : ConcentricCircles)
  (small_arc : CircleArc)
  (large_arc : CircleArc)
  (h_small_arc : small_arc.radius = circles.small_radius)
  (h_large_arc : large_arc.radius = circles.large_radius)
  (h_small_angle : small_arc.angle = 60)
  (h_large_angle : large_arc.angle = 72)
  (h_equal_length : arc_length small_arc = arc_length large_arc) :
  (Real.pi * circles.small_radius^2) / (Real.pi * circles.large_radius^2) = 36 / 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_concentric_circles_l568_56833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_l568_56867

-- Define the circumference of the sphere
noncomputable def sphere_circumference : ℝ := 24 * Real.pi

-- Define the number of wedges
def num_wedges : ℕ := 6

-- Theorem statement
theorem wedge_volume (r : ℝ) (h1 : 2 * Real.pi * r = sphere_circumference) : 
  (4 / 3 * Real.pi * r^3) / num_wedges = 384 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_l568_56867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_travel_time_l568_56818

-- Define the speeds of K and M
noncomputable def speed_K (x : ℝ) := x
noncomputable def speed_M (x : ℝ) := x - (1/2)

-- Define the time taken by K and M to travel 45 miles
noncomputable def time_K (x : ℝ) := 45 / speed_K x
noncomputable def time_M (x : ℝ) := 45 / speed_M x

-- State the theorem
theorem K_travel_time (x : ℝ) (h1 : x > 1/2) :
  time_M x - time_K x = 3/4 →
  time_K x = 45 / x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_travel_time_l568_56818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l568_56808

theorem two_integers_sum (a b : ℕ+) : 
  a * b + a + b = 119 → 
  Nat.Coprime a b → 
  a < 25 → 
  b < 25 → 
  a + b = 27 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l568_56808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l568_56844

open Real

noncomputable def f (x : ℝ) := 3 * sin (2 * x + π / 6)

theorem f_properties :
  (∀ x, f (π / 3 - x) = f (π / 3 + x)) ∧
  (∀ x, f x ≤ 3) ∧
  (∃ x, f x = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l568_56844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lune_area_l568_56889

/-- The area of a lune formed by two semicircles -/
theorem lune_area (r : ℝ) (h : r > 0) : 
  (1/2 * Real.pi * r^2 + 2 * r^2) - (1/6 * Real.pi * (3*r)^2) = (2 - Real.pi) * r^2 := by
  -- Expand the left side of the equation
  have h1 : (1/2 * Real.pi * r^2 + 2 * r^2) - (1/6 * Real.pi * (3*r)^2) 
           = (1/2 * Real.pi * r^2 + 2 * r^2) - (3/2 * Real.pi * r^2) := by
    ring
  
  -- Simplify the expression
  have h2 : (1/2 * Real.pi * r^2 + 2 * r^2) - (3/2 * Real.pi * r^2)
           = 2 * r^2 - Real.pi * r^2 := by
    ring
  
  -- Show that this is equal to (2 - Real.pi) * r^2
  have h3 : 2 * r^2 - Real.pi * r^2 = (2 - Real.pi) * r^2 := by
    ring
  
  -- Combine the steps to prove the theorem
  rw [h1, h2, h3]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lune_area_l568_56889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_analogical_reasoning_correctness_l568_56806

-- Define a custom ordering for complex numbers based on their real parts
instance : LE ℂ where
  le a b := a.re ≤ b.re

instance : LT ℂ where
  lt a b := a.re < b.re

theorem analogical_reasoning_correctness : 
  (∃! n : ℕ, n = 2 ∧ 
    (((∀ a b : ℝ, a^2 < b^2 → a < b) = false) ∧
     ((∀ a b c : ℝ, c ≠ 0 → (a + b) / c = a / c + b / c) = true) ∧
     ((∀ a b : ℂ, a - b = 0 → a = b) = true) ∧
     ((∀ a b : ℂ, (a - b).re > 0 → a > b) = false))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_analogical_reasoning_correctness_l568_56806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_roots_condition_l568_56800

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := x^2 + a*x + b*(Real.cos x)

-- Define the theorem
theorem identical_roots_condition (a b : ℝ) :
  (∃ x : ℝ, f a b x = 0) ∧
  (∀ x : ℝ, f a b x = 0 ↔ f a b (f a b x) = 0) ↔
  b = 0 ∧ 0 ≤ a ∧ a < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_roots_condition_l568_56800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newspaper_subscription_cost_l568_56822

/-- The monthly cost of a newspaper subscription without discount, given that:
    1) An annual subscription has a 20% discount on the total bill.
    2) The discounted annual subscription costs $96. -/
def monthly_cost : ℚ := 10

/-- The discounted annual cost of the subscription -/
def discounted_annual_cost : ℚ := 96

/-- The discount rate for an annual subscription -/
def discount_rate : ℚ := 1/5

/-- The number of months in a year -/
def months_per_year : ℕ := 12

theorem newspaper_subscription_cost :
  monthly_cost * (1 - discount_rate) * months_per_year = discounted_annual_cost :=
by
  -- The proof goes here
  sorry

#eval monthly_cost * (1 - discount_rate) * months_per_year

end NUMINAMATH_CALUDE_ERRORFEEDBACK_newspaper_subscription_cost_l568_56822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_probable_sum_is_127_l568_56811

/-- Represents a card with a power of 2 as its number -/
structure Card where
  power : Nat
  number : Nat
  number_def : number = 2^power
deriving DecidableEq

/-- The set of cards in the hat -/
def cards : Finset Card :=
  Finset.image (fun k => ⟨k - 1, 2^(k - 1), rfl⟩) (Finset.range 7)

/-- The sum of numbers on all cards -/
def total_sum : Nat := Finset.sum cards fun c => c.number

/-- A subset of cards represents a possible pick -/
def is_valid_pick (pick : Finset Card) : Prop :=
  pick ⊆ cards ∧ (Finset.sum pick fun c => c.number) > 124

/-- The sum of a valid pick -/
def pick_sum (pick : Finset Card) : Nat :=
  Finset.sum pick fun c => c.number

/-- The probability of a specific pick -/
noncomputable def pick_probability (pick : Finset Card) : ℝ :=
  1 / (Finset.card cards).choose (Finset.card pick)

/-- The theorem to be proved -/
theorem most_probable_sum_is_127 :
  ∀ (pick : Finset Card), is_valid_pick pick →
    pick_sum pick = total_sum ∨
    pick_probability pick < pick_probability cards :=
by
  sorry

#eval total_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_probable_sum_is_127_l568_56811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_construction_l568_56839

/-- A right-angled triangle with a point on the angle bisector of the right angle -/
structure RightTriangleWithBisectorPoint where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ
  is_right_angle : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0
  P_on_bisector : (P.1 - C.1) * (B.2 - C.2) = (P.2 - C.2) * (B.1 - C.1)

/-- Angles formed by PA and PB with AC and BC respectively -/
noncomputable def angle_with_leg (t : RightTriangleWithBisectorPoint) (X : ℝ × ℝ) : ℝ :=
  Real.arccos ((X.1 - t.C.1) * (t.P.1 - t.C.1) + (X.2 - t.C.2) * (t.P.2 - t.C.2)) /
    (Real.sqrt ((X.1 - t.C.1)^2 + (X.2 - t.C.2)^2) * Real.sqrt ((t.P.1 - t.C.1)^2 + (t.P.2 - t.C.2)^2))

theorem unique_construction
  (t : RightTriangleWithBisectorPoint)
  (φ ψ : ℝ)
  (h_φ : π/4 < φ ∧ φ < 3*π/4)
  (h_ψ : π/4 < ψ ∧ ψ < 3*π/4)
  (h_angles : angle_with_leg t t.A = φ ∧ angle_with_leg t t.B = ψ) :
  ∃! (t' : RightTriangleWithBisectorPoint), t' = t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_construction_l568_56839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inverse_l568_56807

-- Define the power function
noncomputable def f (n : ℝ) : ℝ → ℝ := fun x ↦ x^n

-- Define the inverse function
def g : ℝ → ℝ := fun x ↦ x^2

-- Theorem statement
theorem power_function_inverse 
  (n : ℝ) 
  (h1 : f n 2 = Real.sqrt 2) 
  (h2 : ∀ x : ℝ, x ≥ 0 → g (f n x) = x ∧ f n (g x) = x) :
  n = 1/2 ∧ ∀ x : ℝ, x ≥ 0 → g x = x^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inverse_l568_56807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_ineffective_approx_l568_56809

/-- The number of patients in the trial -/
def n : ℕ := 10

/-- The threshold number of recoveries for the drug to be considered effective -/
def threshold : ℕ := 5

/-- The probability of recovery if the drug is effective -/
def p : ℝ := 0.8

/-- The probability of determining that the drug is ineffective when it actually is effective -/
noncomputable def prob_ineffective : ℝ :=
  (Finset.range threshold).sum (λ k => (n.choose k) * (p^k) * ((1 - p)^(n - k)))

/-- Theorem stating that the probability of determining the drug is ineffective
    when it actually increases the cure rate to 80% is approximately 0.006 -/
theorem prob_ineffective_approx :
  |prob_ineffective - 0.006| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_ineffective_approx_l568_56809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_quadratic_l568_56868

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + 2*x*y + 3*y^2 - 6*x - 2*y ≥ -11 ∧ 
  ∃ (x y : ℝ), x^2 + 2*x*y + 3*y^2 - 6*x - 2*y = -11 := by
  sorry

#check min_value_quadratic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_quadratic_l568_56868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_square_rectangle_perimeter_l568_56877

/-- Represents a rectangle divided into nine non-overlapping squares -/
structure NineSquareRectangle where
  width : ℕ+
  height : ℕ+
  squares : Fin 9 → ℕ+
  sum_smallest_three_eq_largest : squares 0 + squares 1 + squares 2 = squares 8
  width_height_coprime : Nat.Coprime width.val height.val
  valid_decomposition : 
    width = squares 6 + squares 8 ∧
    height = squares 7 + squares 8 ∧
    squares 0 + squares 1 = squares 2 ∧
    squares 0 + squares 2 = squares 3 ∧
    squares 2 + squares 3 = squares 4 ∧
    squares 3 + squares 4 = squares 5 ∧
    squares 1 + squares 2 + squares 4 = squares 6 ∧
    squares 1 + squares 6 = squares 7 ∧
    squares 0 + squares 3 + squares 5 = squares 8 ∧
    squares 5 + squares 8 = squares 6 + squares 7

/-- The perimeter of a rectangle with the given properties is 260 -/
theorem nine_square_rectangle_perimeter (r : NineSquareRectangle) : 
  2 * (r.width + r.height) = 260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_square_rectangle_perimeter_l568_56877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_12th_innings_l568_56891

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  notOutCount : Nat

/-- Calculates the batting average -/
def battingAverage (stats : BatsmanStats) : ℚ :=
  (stats.totalRuns : ℚ) / (stats.innings - stats.notOutCount : ℚ)

/-- Theorem about a batsman's average after 12 innings -/
theorem batsman_average_after_12th_innings 
  (stats : BatsmanStats) 
  (h1 : stats.innings = 12)
  (h2 : stats.notOutCount = 0)
  (h3 : battingAverage stats - battingAverage { stats with 
    innings := stats.innings - 1, 
    totalRuns := stats.totalRuns - 70 
  } = 3)
  : battingAverage stats = 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_12th_innings_l568_56891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_has_card13_l568_56878

-- Define the possible card types
inductive Card : Type
| card12 : Card
| card13 : Card
| card23 : Card

-- Define the people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a function that assigns a card to each person
def card_assignment := Person → Card

-- Define the statements made by each person
def A_statement (assignment : card_assignment) : Prop :=
  ¬(assignment Person.B = Card.card12 ∧ assignment Person.A = Card.card12) ∧
  ¬(assignment Person.B = Card.card23 ∧ assignment Person.A = Card.card23)

def B_statement (assignment : card_assignment) : Prop :=
  ¬(assignment Person.C = Card.card12 ∧ assignment Person.B = Card.card12) ∧
  ¬(assignment Person.C = Card.card13 ∧ assignment Person.B = Card.card13)

def C_statement (assignment : card_assignment) : Prop :=
  assignment Person.C ≠ Card.card23

-- The main theorem
theorem A_has_card13 :
  ∀ (assignment : card_assignment),
    (∀ c : Card, ∃! p : Person, assignment p = c) →
    A_statement assignment →
    B_statement assignment →
    C_statement assignment →
    assignment Person.A = Card.card13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_has_card13_l568_56878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l568_56854

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => sequence_a n / (sequence_a n ^ 2 + 1)

theorem sequence_a_properties :
  (∀ n : ℕ, n > 0 → sequence_a n < sequence_a (n - 1)) ∧
  (∀ n : ℕ, 0 < sequence_a n ∧ sequence_a n ≤ 1) ∧
  (1 / 11 < sequence_a 50 ∧ sequence_a 50 < 1 / 10) := by
  sorry

#check sequence_a_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l568_56854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_distance_to_sides_l568_56832

noncomputable section

-- Define the rectangle dimensions
def rectangle_length : ℝ := 12
def rectangle_width : ℝ := 8

-- Define the lemming's movements
def diagonal_move : ℝ := 7.5
def first_turn_move : ℝ := 3
def second_turn_move : ℝ := 2

-- Function to calculate the final position of the lemming
noncomputable def lemming_final_position : ℝ × ℝ :=
  let diagonal := Real.sqrt (rectangle_length^2 + rectangle_width^2)
  let scale_factor := diagonal_move / diagonal
  let x1 := rectangle_length * scale_factor
  let y1 := rectangle_width * scale_factor
  let x2 := x1 + first_turn_move
  let y2 := y1 - second_turn_move
  (x2, y2)

-- Function to calculate distances to each side
def distances_to_sides (pos : ℝ × ℝ) : List ℝ :=
  let (x, y) := pos
  [x, y, rectangle_length - x, rectangle_width - y]

-- Theorem statement
theorem average_distance_to_sides :
  let pos := lemming_final_position
  let distances := distances_to_sides pos
  (distances.sum / distances.length) = 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_distance_to_sides_l568_56832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sqrt_count_l568_56869

theorem ceiling_sqrt_count : 
  (Finset.filter (fun x : ℤ => ⌈Real.sqrt (x : ℝ)⌉ = 20) (Finset.Icc 361 399)).card = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sqrt_count_l568_56869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_longest_chord_l568_56831

/-- Given a circle where the longest chord has length 16, prove that its radius is 8. -/
theorem circle_radius_from_longest_chord (O : ℝ → ℝ → Prop) :
  (∃ (chord : ℝ × ℝ → ℝ), 
    (∀ c : ℝ × ℝ, chord c ≤ 16) ∧ 
    (∃ c : ℝ × ℝ, chord c = 16)) →
  (∃ (center : ℝ × ℝ) (radius : ℝ), 
    radius = 8 ∧ 
    ∀ p : ℝ × ℝ, O p.1 p.2 ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_longest_chord_l568_56831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_is_36_l568_56866

noncomputable def profit_calculation (natasha_money : ℝ) (natasha_carla_ratio : ℝ) (carla_cosima_ratio : ℝ) (selling_price_ratio : ℝ) : ℝ :=
  let carla_money := natasha_money / natasha_carla_ratio
  let cosima_money := carla_money / carla_cosima_ratio
  let total_money := natasha_money + carla_money + cosima_money
  let selling_price := total_money * selling_price_ratio
  selling_price - total_money

theorem profit_is_36 :
  profit_calculation 60 3 2 (7/5) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_is_36_l568_56866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l568_56859

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x) - 1

-- State the theorem
theorem f_properties :
  -- The smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- The intervals of monotonic increase on [0, π]
  (∀ (x y : ℝ), (0 ≤ x ∧ x < y ∧ y ≤ π / 8) → f x < f y) ∧
  (∀ (x y : ℝ), (5 * π / 8 ≤ x ∧ x < y ∧ y ≤ π) → f x < f y) ∧
  (∀ (x y : ℝ), (π / 8 < x ∧ x < y ∧ y < 5 * π / 8) → f x > f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l568_56859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_factorization_l568_56801

/-- Determines if an expression can be factored using the difference of squares formula -/
def is_difference_of_squares (expr : ℝ → ℝ → ℝ) : Prop :=
  ∃ (f g : ℝ → ℝ → ℝ), ∀ a b, expr a b = (f a b)^2 - (g a b)^2

theorem difference_of_squares_factorization :
  is_difference_of_squares (λ a b => -4*a^2 + b^2) ∧
  ¬is_difference_of_squares (λ a b => a^2 + b^2) ∧
  ¬is_difference_of_squares (λ a b => -a^2 - b^2) ∧
  ¬is_difference_of_squares (λ a c => a^2 - c^2 - 2*a*c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_factorization_l568_56801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_viewers_count_l568_56883

/-- Represents a movie in the cinema --/
inductive Movie
| ToyStory
| IceAge
| Shrek
| MonkeyKing

/-- Returns the price of a movie ticket --/
def ticket_price (m : Movie) : ℕ :=
  match m with
  | Movie.ToyStory => 50
  | Movie.IceAge => 55
  | Movie.Shrek => 60
  | Movie.MonkeyKing => 65

/-- Represents a combination of movies a viewer can watch --/
structure MovieCombination where
  first : Movie
  second : Option Movie
  valid : second.isNone ∨ 
    (first ≠ Movie.IceAge ∨ second ≠ some Movie.Shrek) ∧
    (first ≠ Movie.Shrek ∨ second ≠ some Movie.IceAge)

/-- Calculates the total price for a movie combination --/
def combination_price (c : MovieCombination) : ℕ :=
  ticket_price c.first + match c.second with
  | none => 0
  | some m => ticket_price m

/-- The theorem to be proved --/
theorem min_viewers_count :
  ∃ (combinations : List MovieCombination),
    (∀ c ∈ combinations, ∃ n : ℕ, combination_price c = n) ∧
    (∀ n : ℕ, (∃ c ∈ combinations, combination_price c = n) →
              (∀ c ∈ combinations, combination_price c = n)) ∧
    combinations.length = 9 ∧
    (199 * combinations.length + 1 = 1792) := by
  sorry

#eval 199 * 9 + 1  -- This should output 1792

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_viewers_count_l568_56883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l568_56825

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a^2 * x

theorem a_range (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → |f a x₁ - f a x₂| ≤ 1) →
  a ∈ Set.Icc (-(2 * Real.sqrt 3)/3) ((2 * Real.sqrt 3)/3) := by
  sorry

#check a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l568_56825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_on_interval_l568_56851

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)
noncomputable def g (x : ℝ) : ℝ := 2^(x - 1)
noncomputable def h (x : ℝ) : ℝ := Real.sqrt (x - 1)
noncomputable def k (x : ℝ) : ℝ := Real.log (x - 1)

theorem decreasing_function_on_interval :
  (∀ x y, 1 < x ∧ x < y → f y < f x) ∧
  (∃ x y, 1 < x ∧ x < y ∧ g x ≤ g y) ∧
  (∃ x y, 1 < x ∧ x < y ∧ h x ≤ h y) ∧
  (∃ x y, 1 < x ∧ x < y ∧ k x ≤ k y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_on_interval_l568_56851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l568_56875

theorem circle_equation (C : (ℝ × ℝ) → Prop) :
  (∃ a : ℝ, C = λ (p : ℝ × ℝ) => (p.1 - a)^2 + p.2^2 = 5) →  -- center on x-axis, radius √5
  (∀ p : ℝ × ℝ, C p → p.1 < 0) →  -- located on the left side of y-axis
  (∃ p : ℝ × ℝ, C p ∧ p.1 + 2*p.2 = 0) →  -- tangent to x + 2y = 0
  C = λ (p : ℝ × ℝ) => (p.1 + 5)^2 + p.2^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l568_56875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_l568_56845

theorem coefficient_x_cubed : 
  (Polynomial.coeff ((X - 2) * (X + 1)^5) 3) = -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_l568_56845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_l568_56815

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.cos x * (a * Real.sin x - Real.cos x) + (Real.cos (Real.pi / 2 - x))^2

theorem range_of_f_on_interval (a : ℝ) :
  (f a (-Real.pi/3) = f a 0 ∧ a = 2 * Real.sqrt 3) →
  Set.range (fun x => f a x) ∩ Set.Ioo 0 (Real.pi/3) = Set.Ioc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_l568_56815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_A_contrapositive_true_l568_56824

-- Define the propositions using more general terms
def proposition_A : Prop := ∀ (α β : ℝ), α + β + 90 = 180 → α + β = 90
def proposition_B : Prop := ∀ (a b c : ℝ), a = b → (a = 3 ∧ b = 7) → a + b + c = 17
def proposition_C : Prop := ∀ (A1 A2 : ℝ), A1 = A2 → A1 = A2
def proposition_D : Prop := ∀ (x : ℝ), x = 1 → x^2 = 1

-- Define a function to check if a proposition's contrapositive is true
def contrapositive_is_true (p : Prop) : Prop := ¬p → ¬(p → False)

-- Theorem stating that only proposition A's contrapositive is true
theorem only_A_contrapositive_true :
  contrapositive_is_true proposition_A ∧
  ¬contrapositive_is_true proposition_B ∧
  ¬contrapositive_is_true proposition_C ∧
  ¬contrapositive_is_true proposition_D :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_A_contrapositive_true_l568_56824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_numbers_sum_product_l568_56847

theorem six_numbers_sum_product (a b c d e f : ℕ) :
  a + b + c + d + e + f = a * b * c * d * e * f →
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 ∧ e = 0 ∧ f = 0) ∨
  (List.count 1 [a, b, c, d, e, f] = 4 ∧
   List.count 2 [a, b, c, d, e, f] = 1 ∧
   List.count 6 [a, b, c, d, e, f] = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_numbers_sum_product_l568_56847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_probability_l568_56848

noncomputable def f (a b x : ℝ) : ℝ := a * x + b * Real.sin x

def valid_pair (a b : ℤ) : Prop :=
  a ≠ b ∧ a ∈ ({-2, 0, 1, 2} : Set ℤ) ∧ b ∈ ({-2, 0, 1, 2} : Set ℤ)

def non_negative_slope (a b : ℝ) : Prop :=
  ∀ x, 0 < x → x < Real.pi / 2 → a + b * Real.cos x ≥ 0

def count_non_negative_pairs : ℕ := 7

def total_valid_pairs : ℕ := 12

theorem tangent_slope_probability :
  (count_non_negative_pairs : ℚ) / total_valid_pairs =
  (7 : ℚ) / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_probability_l568_56848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_fraction_representation_l568_56894

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + (x^2 - 1) * Real.sqrt (x^2 - 4)

theorem f_fraction_representation (x : ℝ) :
  (x ≥ 2 → (f x - 2) / (f x + 2) = ((x + 1) * Real.sqrt (x - 2)) / ((x - 1) * Real.sqrt (x + 2))) ∧
  (x ≤ -2 → (f x - 2) / (f x + 2) = (-(x + 1) * Real.sqrt (-x + 2)) / ((x - 1) * Real.sqrt (-x - 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_fraction_representation_l568_56894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_on_interval_l568_56838

open Real

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := sin (2 * x + π / 6)

-- State the theorem
theorem g_increasing_on_interval :
  StrictMonoOn g (Set.Ioo (-π/3) (π/6)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_on_interval_l568_56838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parabola_intersection_midpoint_l568_56835

/-- The line equation: y = x - 1 -/
def line (x y : ℝ) : Prop := y = x - 1

/-- The parabola equation: y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The midpoint coordinates of the line segment -/
def midpoint_coords : ℝ × ℝ := (3, 2)

/-- Theorem: The midpoint of the line segment cut by the line y = x - 1 on the parabola y^2 = 4x is (3, 2) -/
theorem line_parabola_intersection_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line x₁ y₁ ∧ parabola x₁ y₁ ∧
    line x₂ y₂ ∧ parabola x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    midpoint_coords = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parabola_intersection_midpoint_l568_56835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_length_for_8_boys_l568_56843

-- Define the work rate for 6 boys
noncomputable def work_rate_6 (wall_length : ℝ) (days : ℝ) : ℝ := wall_length / days

-- Define the work rate for 8 boys based on the work rate of 6 boys
noncomputable def work_rate_8 (rate_6 : ℝ) : ℝ := rate_6 * (8 / 6)

-- Theorem statement
theorem wall_length_for_8_boys 
  (wall_length_6 : ℝ) 
  (days_6 : ℝ) 
  (days_8 : ℝ) 
  (h1 : wall_length_6 = 60) 
  (h2 : days_6 = 5) 
  (h3 : days_8 = 3.125) : 
  work_rate_8 (work_rate_6 wall_length_6 days_6) * days_8 = 50 := by
  sorry

#check wall_length_for_8_boys

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_length_for_8_boys_l568_56843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l568_56810

/-- Line l in the Cartesian plane -/
def line_l (x y : ℝ) : Prop := x - y = 1

/-- Circle C in the Cartesian plane -/
def circle_C (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 1

/-- Point A on the x-axis -/
def point_A : ℝ × ℝ := (4, 0)

/-- Point B on the x-axis -/
def point_B : ℝ × ℝ := (6, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Perimeter of triangle PAB -/
noncomputable def triangle_perimeter (P : ℝ × ℝ) : ℝ :=
  distance P point_A + distance P point_B + distance point_A point_B

theorem min_perimeter_triangle :
  ∃ (P : ℝ × ℝ), line_l P.1 P.2 ∧
  (∀ (Q : ℝ × ℝ), line_l Q.1 Q.2 → triangle_perimeter P ≤ triangle_perimeter Q) ∧
  triangle_perimeter P = 2 + Real.sqrt 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l568_56810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l568_56850

noncomputable def h (x : ℝ) : ℝ := (2 * x^2 + 3 * x - 1) / (x^2 + x - 12)

theorem domain_of_h :
  {x : ℝ | ∃ y, h x = y} = {x | x < -4 ∨ (-4 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l568_56850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_male_college_degree_count_l568_56857

theorem male_college_degree_count :
  let total_employees : ℕ := 148
  let total_females : ℕ := 92
  let advanced_degrees : ℕ := 78
  let females_advanced : ℕ := 53
  let males : ℕ := total_employees - total_females
  let college_only : ℕ := total_employees - advanced_degrees
  let females_college : ℕ := total_females - females_advanced
  let males_advanced : ℕ := advanced_degrees - females_advanced
  let males_college : ℕ := males - males_advanced
  males_college = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_male_college_degree_count_l568_56857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_in_sphere_l568_56870

-- Define the sphere
noncomputable def sphere_surface_area : ℝ := 4 * Real.pi

-- Define the pyramid
structure Pyramid where
  base_side_length : ℝ
  height : ℝ

-- Define the inscribed pyramid
noncomputable def inscribed_pyramid : Pyramid :=
  { base_side_length := 1,
    height := Real.sqrt 2 / 2 }

-- Theorem statement
theorem pyramid_volume_in_sphere :
  (1/3 : ℝ) * inscribed_pyramid.base_side_length ^ 2 * inscribed_pyramid.height = Real.sqrt 2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_in_sphere_l568_56870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_odd_function_l568_56899

noncomputable def f (x : ℝ) : ℝ := (1 * Real.cos (2 * x)) - (Real.sqrt 3 * Real.sin (2 * x))

noncomputable def g (t : ℝ) (x : ℝ) : ℝ := f (x + t)

theorem min_t_for_odd_function : 
  ∀ t : ℝ, t > 0 → (∀ x : ℝ, g t x = -g t (-x)) → t ≥ π / 12 := by
  sorry

#check min_t_for_odd_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_odd_function_l568_56899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l568_56892

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 7 - y^2 / 3 = 1

-- Define the focal length
noncomputable def focal_length (h : ℝ → ℝ → Prop) : ℝ := 2 * Real.sqrt 10

-- Theorem statement
theorem hyperbola_focal_length :
  focal_length hyperbola = 2 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l568_56892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_for_horizontal_asymptote_l568_56890

/-- Given a rational function f(x) = p(x) / q(x) where
    p(x) = 3x^8 + 5x^5 - 2x^2 and q(x) is a polynomial,
    prove that the smallest possible degree of q(x) for f(x)
    to have a horizontal asymptote is 8. -/
theorem smallest_degree_for_horizontal_asymptote :
  let p (x : ℝ) := 3 * x^8 + 5 * x^5 - 2 * x^2
  ∀ q : ℝ → ℝ, (∃ (L : ℝ), ∀ ε > 0, ∃ M, ∀ x > M, |p x / q x - L| < ε) →
    (∃ n : ℕ, ∀ x, q x = x^n) →
    (∀ n : ℕ, (∀ x, q x = x^n) → n ≥ 8) ∧
    (∃ q : ℝ → ℝ, (∀ x, q x = x^8) ∧
      ∃ (L : ℝ), ∀ ε > 0, ∃ M, ∀ x > M, |p x / q x - L| < ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_for_horizontal_asymptote_l568_56890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l568_56876

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

-- Define the equation
def equation (a b n : ℕ) : Prop := 2^a + 5^b + 1 = factorial n

-- State the theorem
theorem unique_solution :
  ∀ a b n : ℕ, equation a b n → (a = 2 ∧ b = 0 ∧ n = 3) :=
by
  sorry

-- Example to show the equation holds for the solution
example : equation 2 0 3 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l568_56876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_eight_term_ap_with_reversed_ap_l568_56864

/-- The function that reverses the binary representation of an odd positive integer. -/
noncomputable def r (n : ℕ) : ℕ := sorry

/-- Predicate to check if a list of natural numbers forms an arithmetic progression. -/
def is_arithmetic_progression (l : List ℕ) : Prop := sorry

/-- Predicate to check if a list of natural numbers is strictly increasing. -/
def is_strictly_increasing (l : List ℕ) : Prop := sorry

/-- Predicate to check if a natural number is odd. -/
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem no_eight_term_ap_with_reversed_ap :
  ¬ ∃ (a : Fin 8 → ℕ),
    (∀ i, is_odd (a i)) ∧
    is_strictly_increasing (List.ofFn a) ∧
    is_arithmetic_progression (List.ofFn a) ∧
    is_arithmetic_progression (List.ofFn (r ∘ a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_eight_term_ap_with_reversed_ap_l568_56864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_mod_8_l568_56893

/-- Sequence defined by the last two digits of the sum of the previous two terms -/
def a : ℕ → ℕ
  | 0 => 21  -- Adding this case to handle Nat.zero
  | 1 => 21
  | 2 => 90
  | n + 3 => (a (n + 2) + a (n + 1)) % 100

/-- Sum of squares of the first n terms of sequence a -/
def sum_squares (n : ℕ) : ℕ :=
  (List.range n).map (fun i => (a (i + 1))^2) |>.sum

/-- The remainder of the sum of squares of the first 20 terms when divided by 8 is 5 -/
theorem sum_squares_mod_8 : sum_squares 20 % 8 = 5 := by
  sorry

#eval sum_squares 20 % 8  -- Optional: to check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_mod_8_l568_56893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximation_l568_56804

/-- The area of the region inside a rectangle but outside three quarter circles --/
noncomputable def area_outside_circles (cd da ra rb rc : ℝ) : ℝ :=
  cd * da - (Real.pi / 4) * (ra^2 + rb^2 + rc^2)

/-- Theorem stating the approximate area of the region --/
theorem area_approximation :
  let cd := (4 : ℝ)
  let da := (6 : ℝ)
  let ra := (2 : ℝ)
  let rb := (3 : ℝ)
  let rc := (4 : ℝ)
  ∃ ε > 0, |area_outside_circles cd da ra rb rc - 1.235| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximation_l568_56804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l568_56802

/-- Represents the time taken for a train to cross a signal pole -/
noncomputable def time_to_cross_pole (train_length : ℝ) (platform_length : ℝ) (time_to_cross_platform : ℝ) : ℝ :=
  train_length / ((train_length + platform_length) / time_to_cross_platform)

/-- Theorem stating that a 300m train crossing a 200m platform in 30s takes about 18s to cross a signal pole -/
theorem train_crossing_time :
  let train_length : ℝ := 300
  let platform_length : ℝ := 200
  let time_to_cross_platform : ℝ := 30
  abs (time_to_cross_pole train_length platform_length time_to_cross_platform - 18) < 0.1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l568_56802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_partition_l568_56803

/-- A partition of a set into three subsets -/
def Partition (S : Set (ℝ × ℝ)) : Type :=
  { p : (ℝ × ℝ → Fin 3) // ∀ x ∈ S, True }

/-- The unit disc in ℝ² -/
def UnitDisc : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 1}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that it's impossible to partition the unit disc
    into three subsets without two points at distance 1 in the same subset -/
theorem no_valid_partition :
  ¬ ∃ (p : Partition UnitDisc),
    ∀ (x y : ℝ × ℝ), x ∈ UnitDisc → y ∈ UnitDisc →
      p.val x = p.val y → distance x y ≠ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_partition_l568_56803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_distribution_l568_56846

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1020) (h2 : pencils = 860) :
  (Nat.gcd pens pencils) = (
    Finset.sup (Finset.filter (fun n => pens % n = 0 ∧ pencils % n = 0 ∧ n > 0) (Finset.range (min pens pencils + 1)))
      (fun x => x)
  ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_distribution_l568_56846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_l568_56884

theorem rectangle_breadth (square_area : ℝ) (rectangle_area : ℝ) :
  square_area = 1225 →
  rectangle_area = 140 →
  let square_side := Real.sqrt square_area
  let rectangle_length := (2 / 5) * square_side
  let rectangle_breadth := rectangle_area / rectangle_length
  rectangle_breadth = 10 := by
  intro h1 h2
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_l568_56884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l568_56826

noncomputable section

open Real

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  c = 3 ∧ 
  C = π / 3 ∧
  sin B = 2 * sin A →
  -- Law of Sines
  a / sin A = b / sin B ∧
  b / sin B = c / sin C ∧
  c / sin C = a / sin A →
  -- Law of Cosines
  c^2 = a^2 + b^2 - 2*a*b*(cos C) →
  -- Conclusion
  a = sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l568_56826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l568_56887

-- Define the function f as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

-- State the theorem
theorem power_function_properties :
  ∃ α : ℝ, (f α 2 = Real.sqrt 2) ∧ 
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f α x < f α y) ∧
  (¬ ∀ x : ℝ, f α (-x) = f α x) ∧
  (¬ ∀ x : ℝ, f α (-x) = -(f α x)) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l568_56887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_rExpansions_l568_56879

/-- r-expansion of a number x -/
def rExpansion (r : ℝ) (x : ℝ) : (ℕ → ℕ) → Prop :=
  λ a => x = ∑' n, (a n : ℝ) / r^(n+1) ∧ ∀ n, 0 ≤ a n ∧ a n < Int.floor r

/-- The existence of at least one r-expansion for numbers in [0, n/(r-1)] -/
axiom exists_rExpansion (r : ℝ) (n : ℕ) (x : ℝ) (h : 0 ≤ x ∧ x ≤ n / (r - 1)) :
  ∃ a, rExpansion r x a

/-- The main theorem: If r is not an integer, then there exists a number in [0, n/(r-1)]
    with infinitely many r-expansions -/
theorem infinite_rExpansions (r : ℝ) (n : ℕ) (hr : ¬ (∃ m : ℤ, r = m)) :
  ∃ p, 0 ≤ p ∧ p ≤ n / (r - 1) ∧ ∃ f : ℕ → (ℕ → ℕ), (∀ i j, i ≠ j → f i ≠ f j) ∧ 
    ∀ i, rExpansion r p (f i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_rExpansions_l568_56879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_l568_56898

noncomputable def f (x : ℝ) : ℝ := (x^2 - 9) / (x - 3)

theorem no_solution : ¬ ∃ x : ℝ, f x = x + 1 := by
  intro h
  cases' h with x hx
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check no_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_l568_56898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrews_balloons_l568_56821

theorem andrews_balloons (blue_total : ℕ) (purple_total : ℕ)
  (h_blue : blue_total = 303)
  (h_purple : purple_total = 453) :
  (2 * blue_total) / 3 + (3 * purple_total) / 5 = 473 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrews_balloons_l568_56821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l568_56862

/-- The function f(x) = 1/x + 2x -/
noncomputable def f (x : ℝ) : ℝ := 1/x + 2*x

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := 2 - 1/(x^2)

/-- The slope of the tangent line at x = 1 -/
noncomputable def m : ℝ := f' 1

/-- The y-coordinate of the point where the tangent touches the curve at x = 1 -/
noncomputable def y₁ : ℝ := f 1

/-- The equation of the tangent line at x = 1 -/
def tangent_line (x y : ℝ) : Prop := x - y + 2 = 0

theorem tangent_line_at_one :
  tangent_line 1 y₁ ∧ ∀ x y, tangent_line x y ↔ y = m * (x - 1) + y₁ := by
  sorry

#check tangent_line_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l568_56862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_f_g_l568_56836

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x - 3)
noncomputable def g (x : ℝ) : ℝ := 1/4 + Real.log (x/2)

theorem min_difference_f_g :
  ∀ m n : ℝ, f m = g n → 
  (∀ m' n' : ℝ, f m' = g n' → n' - m' ≥ 1/2 + Real.log 2) ∧
  (∃ m₀ n₀ : ℝ, f m₀ = g n₀ ∧ n₀ - m₀ = 1/2 + Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_f_g_l568_56836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposed_parallelogram_definition_is_incorrect_l568_56895

/-- A polygon is a closed broken line (allowing for self-intersections) -/
structure Polygon where
  is_closed : Bool
  allows_self_intersections : Bool

/-- A quadrilateral is a polygon with four sides -/
structure Quadrilateral extends Polygon where
  sides : Fin 4 → ℝ × ℝ  -- Representing sides as vectors

/-- Definition of parallel sides -/
def are_parallel (s1 s2 : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), s1 = k • s2 ∨ s2 = k • s1

/-- Definition of equal sides -/
def are_equal (s1 s2 : ℝ × ℝ) : Prop :=
  s1 = s2

/-- The proposed definition of a parallelogram -/
def is_parallelogram_proposed (q : Quadrilateral) : Prop :=
  (are_equal (q.sides 0) (q.sides 2) ∧ are_parallel (q.sides 0) (q.sides 2)) ∨
  (are_equal (q.sides 1) (q.sides 3) ∧ are_parallel (q.sides 1) (q.sides 3))

/-- The correct definition of a parallelogram -/
def is_parallelogram_correct (q : Quadrilateral) : Prop :=
  are_parallel (q.sides 0) (q.sides 2) ∧ are_parallel (q.sides 1) (q.sides 3)

/-- Theorem stating that the proposed definition is not equivalent to the correct definition -/
theorem proposed_parallelogram_definition_is_incorrect :
  ∃ (q : Quadrilateral), is_parallelogram_proposed q ≠ is_parallelogram_correct q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposed_parallelogram_definition_is_incorrect_l568_56895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weekly_payment_l568_56888

/-- The weekly payment for employee Y in rupees -/
def payment_Y : ℚ := 318.1818181818182

/-- The percentage of Y's wage that X is paid -/
def percentage_X : ℚ := 120

/-- Rounds a rational number to two decimal places -/
def round_to_two_decimals (x : ℚ) : ℚ :=
  (x * 100).floor / 100

/-- Theorem stating the total weekly payment for both employees -/
theorem total_weekly_payment :
  round_to_two_decimals (payment_Y + payment_Y * percentage_X / 100) = 700 := by
  sorry

#eval round_to_two_decimals (payment_Y + payment_Y * percentage_X / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weekly_payment_l568_56888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_field_area_l568_56849

/-- Represents the properties of a football field -/
structure FootballField where
  total_fertilizer : ℚ
  partial_fertilizer : ℚ
  partial_area : ℚ

/-- Calculates the total area of a football field given its properties -/
noncomputable def calculate_total_area (field : FootballField) : ℚ :=
  field.total_fertilizer * field.partial_area / field.partial_fertilizer

/-- Theorem stating that the total area of the given football field is 7200 square yards -/
theorem football_field_area (field : FootballField)
  (h1 : field.total_fertilizer = 1200)
  (h2 : field.partial_fertilizer = 600)
  (h3 : field.partial_area = 3600) :
  calculate_total_area field = 7200 := by
  sorry

#eval (1200 : ℚ) * (3600 : ℚ) / (600 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_field_area_l568_56849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bianca_first_day_miles_l568_56823

/-- Represents the number of miles Bianca ran on the first day -/
def first_day_miles : ℕ := by sorry

/-- Represents the number of miles Bianca ran on the second day -/
def second_day_miles : ℕ := 4

/-- Represents the total number of miles Bianca ran over the two days -/
def total_miles : ℕ := 12

/-- Theorem stating that Bianca ran 8 miles on the first day -/
theorem bianca_first_day_miles :
  first_day_miles = 8 ∧ first_day_miles + second_day_miles = total_miles := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bianca_first_day_miles_l568_56823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_is_cos_l568_56863

open Real

noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => sin
  | (n + 1) => deriv (f n)

theorem f_2017_is_cos : f 2017 = cos := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_is_cos_l568_56863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solvability_l568_56858

noncomputable def integer_part (x : ℝ) : ℤ := Int.floor x

theorem system_solvability (p : ℝ) :
  (∃ (x y : ℝ), 2 * (integer_part x) + y = 3/2 ∧ 3 * (integer_part x) - 2 * y = p) ↔
  (∃ k : ℤ, p = 7 * k - 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solvability_l568_56858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_two_l568_56856

def seven_numbers : List ℕ := [1870, 1995, 2020, 2026, 2110, 2124, 2500]

theorem mean_of_remaining_two (numbers : List ℕ) (h1 : numbers = seven_numbers) 
  (h2 : ∃ (five : List ℕ), five.length = 5 ∧ (five.sum : ℚ) / 5 = 2100 ∧ five.toFinset ⊆ numbers.toFinset) 
  (h3 : numbers.sum = 14645) :
  let remaining := numbers.filter (λ x => x ∉ (numbers.filter (λ y => y ∈ (Classical.choose h2).toFinset)).toFinset)
  (remaining.sum : ℚ) / 2 = 2072.5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_two_l568_56856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_permutations_l568_56896

def digits : List Nat := [1, 2, 3, 4, 5]

def is_permutation (l : List Nat) : Prop :=
  l.length = digits.length ∧ l.toFinset = digits.toFinset

def digit_sum (l : List Nat) : Nat :=
  l.sum

theorem no_prime_permutations :
  ∀ l : List Nat, is_permutation l →
    ∃ n : Nat, n > 1 ∧ (l.foldl (fun acc d ↦ acc * 10 + d) 0) % n = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_permutations_l568_56896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_results_in_loss_l568_56842

/-- The selling price of the house -/
def house_selling_price : ℚ := 10000

/-- The selling price of the store -/
def store_selling_price : ℚ := 15000

/-- The loss percentage on the house -/
def house_loss_percent : ℚ := 25 / 100

/-- The gain percentage on the store -/
def store_gain_percent : ℚ := 25 / 100

/-- The cost price of the house -/
def house_cost_price : ℚ := house_selling_price / (1 - house_loss_percent)

/-- The cost price of the store -/
def store_cost_price : ℚ := store_selling_price / (1 + store_gain_percent)

/-- The total cost price -/
def total_cost_price : ℚ := house_cost_price + store_cost_price

/-- The total selling price -/
def total_selling_price : ℚ := house_selling_price + store_selling_price

/-- The transaction loss -/
def transaction_loss : ℚ := total_cost_price - total_selling_price

theorem transaction_results_in_loss :
  ∃ (n : ℤ), (transaction_loss * 3).num = n ∧ (transaction_loss * 3).den = 1 ∧ n = 1000 := by
  sorry

#eval (transaction_loss * 3).num
#eval (transaction_loss * 3).den

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_results_in_loss_l568_56842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_k_rightAngled_iff_k_l568_56813

-- Define the vectors AB and AC
def AB (k : ℝ) : ℝ × ℝ := (2 - k, -1)
def AC (k : ℝ) : ℝ × ℝ := (1, k)

-- Define collinearity condition
def collinear (k : ℝ) : Prop :=
  ∃ t : ℝ, AB k = t • AC k

-- Define right-angled triangle condition
def rightAngled (k : ℝ) : Prop :=
  let BC := ((AC k).1 - (AB k).1, (AC k).2 - (AB k).2)
  (AB k).1 * (AC k).1 + (AB k).2 * (AC k).2 = 0 ∨
  (AB k).1 * BC.1 + (AB k).2 * BC.2 = 0 ∨
  BC.1 * (AC k).1 + BC.2 * (AC k).2 = 0

-- Theorem statements
theorem collinear_iff_k (k : ℝ) :
  collinear k ↔ k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2 :=
sorry

theorem rightAngled_iff_k (k : ℝ) :
  rightAngled k ↔ k = 1 ∨ k = -1 + Real.sqrt 2 ∨ k = -1 - Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_k_rightAngled_iff_k_l568_56813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_l568_56861

/-- An arithmetic sequence with first term -9 and S_3 = S_7 -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = -9
  sum_equality : (Finset.range 3).sum a = (Finset.range 7).sum a

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (Finset.range n).sum seq.a

/-- The theorem stating that the smallest sum occurs when n = 5 -/
theorem smallest_sum (seq : ArithmeticSequence) :
  ∀ n ∈ ({4, 5, 6, 7} : Finset ℕ), sum_n seq 5 ≤ sum_n seq n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_l568_56861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_satisfies_conditions_l568_56828

-- Define sets A and B
def A : Set ℝ := {x | 3 * x^2 - 8 * x + 4 > 0}
def B (a : ℝ) : Set ℝ := {x | -2 / (x^2 - a*x - 2*a^2) < 0}

-- Define the range of a
def range_of_a : Set ℝ := Set.Iic (-2) ∪ Set.Ici 1

-- Theorem statement
theorem range_of_a_satisfies_conditions (a : ℝ) :
  (a ∈ range_of_a) ↔
  ((Aᶜ ∩ B a = ∅) ∧ (B a ⊂ A)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_satisfies_conditions_l568_56828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_51_equals_5151_l568_56871

/-- Definition of the original sequence a_n -/
def a (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate to check if a number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 ≠ 0

/-- Definition of the sequence b_n -/
def b (n : ℕ) : ℕ := 
  let k := 2 * n - 1
  a k

/-- Theorem stating that the 51st term of b_n is 5151 -/
theorem b_51_equals_5151 : b 51 = 5151 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_51_equals_5151_l568_56871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_y_intercept_l568_56855

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ :=
  let m := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)
  let b := l.y₁ - m * l.x₁
  b

theorem line_y_intercept :
  let l : Line := { x₁ := 3, y₁ := 27, x₂ := -7, y₂ := -5 }
  y_intercept l = 17.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_y_intercept_l568_56855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_supplies_cost_decrease_l568_56834

/-- Represents the cost decrease percentages and relative costs of painting supplies -/
structure PaintingSupplies where
  canvas_decrease : ℝ
  paint_decrease : ℝ
  brushes_decrease : ℝ
  paint_to_canvas_ratio : ℝ
  brushes_to_canvas_ratio : ℝ

/-- Calculates the total cost decrease percentage for painting supplies -/
noncomputable def total_cost_decrease (supplies : PaintingSupplies) : ℝ :=
  let original_total := 1 + supplies.paint_to_canvas_ratio + supplies.brushes_to_canvas_ratio
  let new_canvas := 1 - supplies.canvas_decrease
  let new_paint := supplies.paint_to_canvas_ratio * (1 - supplies.paint_decrease)
  let new_brushes := supplies.brushes_to_canvas_ratio * (1 - supplies.brushes_decrease)
  let new_total := new_canvas + new_paint + new_brushes
  (original_total - new_total) / original_total * 100

/-- Theorem stating that the total cost decrease for the given painting supplies is 36.875% -/
theorem painting_supplies_cost_decrease :
  let supplies := PaintingSupplies.mk 0.35 0.50 0.20 4 3
  total_cost_decrease supplies = 36.875 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_supplies_cost_decrease_l568_56834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_converges_to_four_fifths_two_fifths_l568_56837

/-- Represents the position of the bug -/
structure BugPosition where
  x : ℝ
  y : ℝ

/-- Represents a single movement of the bug -/
structure BugMovement where
  distance : ℝ
  angle : ℝ

/-- The initial position of the bug -/
def initial_position : BugPosition := ⟨0, 0⟩

/-- The sequence of bug movements -/
noncomputable def bug_movements : ℕ → BugMovement
  | 0 => ⟨1, 0⟩
  | n + 1 => ⟨(1/2)^(n+1), (Real.pi/2) * (n+1)⟩

/-- The position of the bug after n movements -/
noncomputable def bug_position (n : ℕ) : BugPosition :=
  sorry

/-- The limit of the bug's position as n approaches infinity -/
noncomputable def bug_limit : BugPosition :=
  sorry

theorem bug_converges_to_four_fifths_two_fifths :
  bug_limit = ⟨4/5, 2/5⟩ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_converges_to_four_fifths_two_fifths_l568_56837
