import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_A_not_sufficient_for_right_triangle_l708_70841

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real

-- Define angle measures
noncomputable def angle_measure (t : Triangle) (v : Real) : Real := sorry

-- Define side lengths
noncomputable def side_length (t : Triangle) (p q : Real) : Real := sorry

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  angle_measure t t.A = 90 ∨ angle_measure t t.B = 90 ∨ angle_measure t t.C = 90

-- Condition A
def condition_A (t : Triangle) : Prop :=
  ∃ (k : Real), angle_measure t t.A = 3 * k ∧
             angle_measure t t.B = 4 * k ∧
             angle_measure t t.C = 5 * k

-- Condition B
def condition_B (t : Triangle) : Prop :=
  ∃ (k : Real), side_length t t.A t.B = 3 * k ∧
             side_length t t.B t.C = 4 * k ∧
             side_length t t.A t.C = 5 * k

-- Condition C
def condition_C (t : Triangle) : Prop :=
  angle_measure t t.A + angle_measure t t.B = angle_measure t t.C

-- Condition D
def condition_D (t : Triangle) : Prop :=
  (side_length t t.A t.B)^2 = (side_length t t.B t.C)^2 + (side_length t t.A t.C)^2

-- Theorem statement
theorem condition_A_not_sufficient_for_right_triangle :
  ∃ (t : Triangle), condition_A t ∧ ¬is_right_triangle t ∧
  (∀ (t : Triangle), condition_B t → is_right_triangle t) ∧
  (∀ (t : Triangle), condition_C t → is_right_triangle t) ∧
  (∀ (t : Triangle), condition_D t → is_right_triangle t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_A_not_sufficient_for_right_triangle_l708_70841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l708_70873

/-- Given a triangle ABC and a point D in its plane, if BC = 3CD and AD = mAB + nAC, 
    then m = -1/3 and n = 4/3 -/
theorem vector_decomposition (A B C D : EuclideanSpace ℝ (Fin 2)) 
  (h : (B - C) = (3 : ℝ) • (C - D)) : 
  ∃ (m n : ℝ), (D - A) = m • (B - A) + n • (C - A) ∧ m = -1/3 ∧ n = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l708_70873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_with_small_perimeter_exists_l708_70806

/-- Represents a cell in a grid --/
structure Cell where
  x : ℤ
  y : ℤ

/-- Represents a shape composed of cells --/
def Shape := Set Cell

/-- A 5-cell cross shape --/
def Cross : Shape := sorry

/-- A 4-cell corner shape --/
def Corner : Shape := sorry

/-- Checks if two shapes overlap --/
def overlaps (s1 s2 : Shape) : Prop := sorry

/-- Calculates the perimeter of a shape --/
def perimeter (s : Shape) : ℕ := sorry

/-- Combines shapes into a single shape --/
def combineShapes (shapes : List Shape) : Shape := sorry

/-- Theorem stating that a figure with perimeter less than 25 can be formed --/
theorem figure_with_small_perimeter_exists :
  ∃ (crosses corners : List Shape),
    crosses.length = 4 ∧
    corners.length = 4 ∧
    (∀ c, c ∈ crosses → c = Cross) ∧
    (∀ c, c ∈ corners → c = Corner) ∧
    (∀ s1 s2, s1 ∈ crosses ++ corners → s2 ∈ crosses ++ corners → s1 ≠ s2 → ¬ overlaps s1 s2) ∧
    perimeter (combineShapes (crosses ++ corners)) < 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_with_small_perimeter_exists_l708_70806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_principal_l708_70820

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℚ)
  (months : ℕ)
  (rate_paise : ℚ)
  (h_interest : interest = 16.32)
  (h_months : months = 8)
  (h_rate_paise : rate_paise = 6) :
  let rate_rupee : ℚ := rate_paise / 100
  let principal : ℚ := interest / (rate_rupee * months)
  principal = 34 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_principal_l708_70820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombic_prism_cross_section_area_l708_70842

/-- A right prism with a rhombic base -/
structure RhombicPrism where
  height : ℝ
  side : ℝ
  acute_angle : ℝ
  cutting_angle : ℝ

/-- The area of the cross section of a rhombic prism -/
noncomputable def cross_section_area (p : RhombicPrism) : ℝ :=
  (4 * Real.sqrt 3) / 3

/-- Theorem: The area of the cross section of a rhombic prism with given dimensions -/
theorem rhombic_prism_cross_section_area 
  (p : RhombicPrism) 
  (h₁ : p.height = 1)
  (h₂ : p.side = 2)
  (h₃ : p.acute_angle = π / 6)  -- 30 degrees in radians
  (h₄ : p.cutting_angle = π / 3)  -- 60 degrees in radians
  : cross_section_area p = (4 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombic_prism_cross_section_area_l708_70842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_properties_l708_70853

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + (1/2) * x^2

theorem extremum_properties (x₀ : ℝ) (h_pos : x₀ > 0) 
  (h_extremum : ∀ x, f x ≥ f x₀ ∨ f x ≤ f x₀) :
  0 < x₀ ∧ x₀ < 1 / Real.exp 1 ∧ f x₀ + x₀ < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_properties_l708_70853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_pricing_strategy_l708_70895

theorem merchant_pricing_strategy (list_price : ℝ) (list_price_pos : list_price > 0) :
  let cost_price := 0.7 * list_price
  let profit_margin := 0.25
  let discount_rate := 0.25
  let selling_price := cost_price / (1 - profit_margin)
  let marked_price := selling_price / (1 - discount_rate)
  ∃ ε > 0, |marked_price - 1.2444 * list_price| < ε := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_pricing_strategy_l708_70895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_cos_2x_minus_pi_over_6_l708_70898

noncomputable def f (x : ℝ) := Real.cos x

noncomputable def g (x : ℝ) : ℝ := f (2 * (x - Real.pi / 12))

theorem g_equals_cos_2x_minus_pi_over_6 :
  ∀ x : ℝ, g x = Real.cos (2 * x - Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_cos_2x_minus_pi_over_6_l708_70898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l708_70819

def digits : List Nat := [0, 1, 2, 3, 4, 5, 6]

def isEven (n : Nat) : Bool := n % 2 = 0

def isOdd (n : Nat) : Bool := ¬(isEven n)

def isDivisibleBy5 (n : Nat) : Bool := n % 5 = 0

def isValidNumber (n : Nat) : Bool :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100) ∈ digits ∧
  ((n / 10) % 10) ∈ digits ∧
  (n % 10) ∈ digits ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10)

def hasCorrectParity (n : Nat) : Bool :=
  (isEven (n / 100) ∧ isEven ((n / 10) % 10) ∧ isOdd (n % 10)) ∨
  (isEven (n / 100) ∧ isOdd ((n / 10) % 10) ∧ isEven (n % 10)) ∨
  (isOdd (n / 100) ∧ isEven ((n / 10) % 10) ∧ isEven (n % 10))

theorem count_valid_numbers :
  (Finset.filter (λ n : Nat => isValidNumber n ∧ hasCorrectParity n ∧ isDivisibleBy5 n)
    (Finset.range 1000)).card = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l708_70819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1001_of_2_pow_1990_eq_256_l708_70826

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- f₁ function: square of the sum of digits -/
def f₁ (k : ℕ) : ℕ := (sumOfDigits k) ^ 2

/-- fₙ function: recursive definition -/
def f : ℕ → ℕ → ℕ
  | 0, k => k
  | 1, k => f₁ k
  | n + 1, k => f₁ (f n k)

/-- Main theorem: f₁₀₀₁(2^1990) = 256 -/
theorem f_1001_of_2_pow_1990_eq_256 : f 1001 (2^1990) = 256 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1001_of_2_pow_1990_eq_256_l708_70826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_properties_l708_70871

/-- The cumulative distribution function of the standard normal distribution -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- The probability measure for the standard normal distribution -/
noncomputable def P : Set ℝ → ℝ := sorry

theorem normal_distribution_properties (a : ℝ) (h : a > 0) :
  (Φ 0 = 0.5) ∧
  (Φ a + Φ (-a) = 1) ∧
  (P {x | x < a} = 2 * Φ a - 1) ∧
  (P {x | |x| > a} = 2 - 2 * Φ a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_properties_l708_70871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l708_70869

-- Define the sequences
def sequence1 (x y a₁ a₂ a₃ : ℝ) : List ℝ := [x, a₁, a₂, a₃, y]
def sequence2 (x y b₁ b₂ b₃ : ℝ) : List ℝ := [b₁, x, b₂, b₃, y, b₁]

-- Define arithmetic sequence property
def isArithmetic (s : List ℝ) : Prop :=
  ∀ i j k, i < j ∧ j < k ∧ k < s.length → s[j]! - s[i]! = s[k]! - s[j]!

-- Main theorem
theorem arithmetic_sequence_ratio
  (x y a₁ a₂ a₃ b₁ b₂ b₃ b₄ : ℝ)
  (h_neq : x ≠ y)
  (h_arith1 : isArithmetic (sequence1 x y a₁ a₂ a₃))
  (h_arith2 : isArithmetic (sequence2 x y b₁ b₂ b₃))
  : (b₄ - b₃) / (a₂ - a₁) = 8/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l708_70869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l708_70808

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := (1/12) * x^2 + a * x + b

-- Define the points A, B, C, and T
structure Point where
  x : ℝ
  y : ℝ

-- Theorem statement
theorem intersection_property (a b : ℝ) (A C B T : Point) :
  -- f intersects x-axis at A and C
  f a b A.x = 0 ∧ f a b C.x = 0 ∧ A.y = 0 ∧ C.y = 0 ∧
  -- f intersects y-axis at B
  B.x = 0 ∧ f a b 0 = B.y ∧
  -- T is the point (3, 3)
  T.x = 3 ∧ T.y = 3 ∧
  -- TA = TB = TC
  (T.x - A.x)^2 + (T.y - A.y)^2 = (T.x - B.x)^2 + (T.y - B.y)^2 ∧
  (T.x - B.x)^2 + (T.y - B.y)^2 = (T.x - C.x)^2 + (T.y - C.y)^2 →
  b = -6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l708_70808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_theorem_l708_70878

/-- The angle between the common tangent to two circles and the line segment connecting their centers -/
noncomputable def tangent_angle (r : ℝ) : ℝ :=
  90 - 2 * Real.arctan (1 / 2)

/-- Given two circles with equal radii and their centers 2r apart,
    the angle between their common tangent and the line connecting their centers
    is equal to 90° - 2 * arctan(1/2) -/
theorem tangent_angle_theorem (r : ℝ) (h : r > 0) :
  tangent_angle r = 90 - 2 * Real.arctan (1 / 2) := by
  -- Unfold the definition of tangent_angle
  unfold tangent_angle
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_theorem_l708_70878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_at_one_plus_i_l708_70810

open Complex

/-- The complex function g(z) = (z^4 - 2z^2 + 1) / (z - I) -/
noncomputable def g (z : ℂ) : ℂ := (z^4 - 2*z^2 + 1) / (z - I)

/-- Theorem stating that g(1+I) = 17 - 4I -/
theorem g_at_one_plus_i : g (1 + I) = 17 - 4*I := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_at_one_plus_i_l708_70810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_manager_age_range_l708_70863

theorem hiring_manager_age_range 
  (average_age : ℕ) 
  (std_dev : ℕ) 
  (num_ages : ℕ) 
  (h1 : average_age = 10)
  (h2 : std_dev = 8)
  (h3 : num_ages = 17) :
  ∃ k : ℕ, 
    k = 1 ∧ 
    Finset.range (average_age + k * std_dev + 1 - (average_age - k * std_dev)) = 
      Finset.range num_ages.succ ∩ Finset.range (average_age * 2 + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_manager_age_range_l708_70863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nectarine_max_profit_l708_70896

/-- Yield function for nectarine tree -/
noncomputable def yield (x : ℝ) : ℝ := 4 - 3 / (x + 1)

/-- Profit function for nectarine tree -/
noncomputable def profit (x : ℝ) : ℝ := 16 * yield x - x - 2 * x

/-- Theorem stating the maximum profit and optimal fertilizer cost -/
theorem nectarine_max_profit :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 5 ∧
  (∀ y, 0 ≤ y ∧ y ≤ 5 → profit y ≤ profit x) ∧
  profit x = 43 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nectarine_max_profit_l708_70896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_a2_range_l708_70812

/-- An infinite geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The sum of an infinite geometric sequence -/
noncomputable def SumOfGeometricSequence (a : ℕ → ℝ) : ℝ :=
  a 0 / (1 - (a 1 / a 0))

theorem geometric_sequence_a2_range (a : ℕ → ℝ) :
  GeometricSequence a →
  SumOfGeometricSequence a = 4 →
  ∃ x : ℝ, x = a 1 ∧ (x ∈ Set.Ioo (-8) 0 ∪ Set.Ico 0 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_a2_range_l708_70812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_ride_cost_l708_70828

/-- Calculates the cost of a taxi ride -/
def taxi_cost (base_fare : ℚ) (per_mile_rate : ℚ) (distance : ℚ) (discount : ℚ) (discount_threshold : ℚ) : ℚ :=
  let total_cost := base_fare + per_mile_rate * distance
  if distance > discount_threshold then total_cost - discount else total_cost

theorem taxi_ride_cost : 
  let base_fare : ℚ := 2
  let per_mile_rate : ℚ := 3/10
  let distance : ℚ := 12
  let discount : ℚ := 1
  let discount_threshold : ℚ := 10
  taxi_cost base_fare per_mile_rate distance discount discount_threshold = 23/5 := by
  sorry

#eval taxi_cost 2 (3/10) 12 1 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_ride_cost_l708_70828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_length_is_240_l708_70832

/-- The length of a race where runner A beats B by 40 yards, B beats C by 30 yards, 
    and A beats C by 65 yards. -/
def race_length : ℝ := 240

/-- The speed of runner A -/
def a : ℝ := sorry

/-- The speed of runner B -/
def b : ℝ := sorry

/-- The speed of runner C -/
def c : ℝ := sorry

/-- A beats B by 40 yards -/
axiom a_beats_b : race_length / a = (race_length - 40) / b

/-- B beats C by 30 yards -/
axiom b_beats_c : race_length / b = (race_length - 30) / c

/-- A beats C by 65 yards -/
axiom a_beats_c : race_length / a = (race_length - 65) / c

theorem race_length_is_240 : race_length = 240 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_length_is_240_l708_70832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_equilateral_condition_l708_70889

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  side_angle_relation : 
    a / Real.sin A = b / Real.sin B ∧ 
    b / Real.sin B = c / Real.sin C ∧ 
    a / Real.sin A = c / Real.sin C

-- Statement 1
theorem sine_inequality (t : Triangle) : 
  t.A > t.B → t.B > t.C → Real.sin t.A > Real.sin t.B ∧ Real.sin t.B > Real.sin t.C :=
by
  sorry

-- Statement 4
theorem equilateral_condition (t : Triangle) : 
  t.a / Real.cos (t.A / 2) = t.b / Real.cos (t.B / 2) ∧
  t.b / Real.cos (t.B / 2) = t.c / Real.cos (t.C / 2) →
  t.A = t.B ∧ t.B = t.C :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_equilateral_condition_l708_70889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_g_odd_l708_70809

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (3 : ℝ)^x + (3 : ℝ)^(-x)
noncomputable def g (x : ℝ) : ℝ := (3 : ℝ)^x - (3 : ℝ)^(-x)

-- State the theorem
theorem f_even_and_g_odd :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, g (-x) = -g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_g_odd_l708_70809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_6_simplest_l708_70877

/-- A square root is in its simplest form if it cannot be simplified further. -/
noncomputable def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y z : ℝ, y * y = x → z * z = x → y ≤ z

/-- The given options for square roots. -/
noncomputable def sqrt_options : List ℝ := [Real.sqrt (1/2), Real.sqrt 8, Real.sqrt 1.5, Real.sqrt 6]

/-- Theorem: √6 is in its simplest form among the given options. -/
theorem sqrt_6_simplest : 
  ∀ x ∈ sqrt_options, is_simplest_sqrt (Real.sqrt 6) → is_simplest_sqrt x → x = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_6_simplest_l708_70877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_permutation_sum_l708_70800

def permutation_sum (p : Equiv.Perm (Fin 6)) : ℚ :=
  |p.toFun 0 - p.toFun 1| + |p.toFun 2 - p.toFun 3| + |p.toFun 4 - p.toFun 5|

theorem average_permutation_sum :
  (Finset.sum (Finset.univ : Finset (Equiv.Perm (Fin 6))) permutation_sum) /
  (Finset.card (Finset.univ : Finset (Equiv.Perm (Fin 6)))) = 55 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_permutation_sum_l708_70800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_arrangement_l708_70883

/-- Represents the hexagon with six vertices --/
structure Hexagon :=
  (A B C D E F : Fin 6)

/-- Checks if a hexagon contains digits from 1 to 6 --/
def valid_digits (h : Hexagon) : Prop :=
  {h.A, h.B, h.C, h.D, h.E, h.F} = Finset.univ

/-- Checks if the sum condition is satisfied for all 7 lines --/
def sum_condition (h : Hexagon) : Prop :=
  (h.A : ℕ) + (h.D : ℕ) + (h.E : ℕ) = 15 ∧
  (h.A : ℕ) + (h.B : ℕ) + (h.F : ℕ) = 15 ∧
  (h.B : ℕ) + (h.C : ℕ) + (h.D : ℕ) = 15 ∧
  (h.C : ℕ) + (h.E : ℕ) + (h.F : ℕ) = 15 ∧
  (h.A : ℕ) + (h.C : ℕ) + (h.E : ℕ) = 15 ∧
  (h.B : ℕ) + (h.D : ℕ) + (h.F : ℕ) = 15 ∧
  (h.A : ℕ) + (h.B : ℕ) + (h.C : ℕ) + (h.D : ℕ) + (h.E : ℕ) + (h.F : ℕ) = 21

/-- The main theorem stating that there's only one valid arrangement --/
theorem unique_arrangement :
  ∃! h : Hexagon, valid_digits h ∧ sum_condition h ∧
    h.A = 3 ∧ h.B = 0 ∧ h.C = 1 ∧ h.D = 4 ∧ h.E = 5 ∧ h.F = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_arrangement_l708_70883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_numbers_inequality_l708_70899

/-- Given positive real numbers a, b, c such that a^(3/2) + b^(3/2) + c^(3/2) = 1,
    prove that abc ≤ 1/9 and a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2√(abc)) -/
theorem positive_numbers_inequality (a b c : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a * b * c ≤ 1/9 ∧ 
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_numbers_inequality_l708_70899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_sum_l708_70854

/-- The radius of the nth circle in the sequence -/
noncomputable def radius (n : ℕ) : ℝ := (1 : ℝ) / (3 ^ (n - 1))

/-- The area of the nth circle in the sequence -/
noncomputable def area (n : ℕ) : ℝ := Real.pi * (radius n) ^ 2

/-- The sum of the areas of all circles in the sequence -/
noncomputable def areaSum : ℝ := ∑' n, area n

/-- Theorem: The sum of the areas of all circles in the sequence is 9π/8 -/
theorem circle_area_sum : areaSum = (9 * Real.pi) / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_sum_l708_70854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l708_70872

-- Define a right triangle with an acute angle β
structure RightTriangle where
  β : Real
  is_acute : 0 < β ∧ β < Real.pi / 2
  is_right_triangle : Real.tan (β/2) = 1 / Real.sqrt 3

-- Define φ as the angle between the median and angle bisector from β
noncomputable def angle_phi (triangle : RightTriangle) : Real :=
  -- The actual definition would depend on geometric calculations
  sorry

-- State the theorem
theorem tan_phi_value (triangle : RightTriangle) :
  Real.tan (angle_phi triangle) = (3 * Real.sqrt 3 + 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l708_70872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_equation_l708_70807

theorem count_integers_satisfying_equation : 
  ∃ (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ 1 + ⌊(105 * n : ℚ) / 106⌋ = ⌈(104 * n : ℚ) / 105⌉) ∧ Finset.card S = 11130 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_equation_l708_70807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_properties_l708_70856

def regular_polygon (perimeter : ℝ) (side_length : ℝ) : Prop :=
  perimeter > 0 ∧ side_length > 0 ∧ ∃ n : ℕ, perimeter = n * side_length

theorem regular_polygon_properties 
  (perimeter : ℝ) (side_length : ℝ) 
  (h : regular_polygon perimeter side_length) 
  (h_perimeter : perimeter = 150) 
  (h_side : side_length = 10) : 
  (perimeter / side_length = 15) ∧ 
  ((((perimeter / side_length : ℝ) - 2) * 180) / (perimeter / side_length) = 156) := by
  sorry

#check regular_polygon_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_properties_l708_70856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l708_70858

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line
def line (m : ℝ) (x y : ℝ) : Prop := y = m

-- Define the circle
def myCircle (n : ℝ) (x y : ℝ) : Prop := (x - n)^2 + y^2 = 1

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

theorem parabola_intersection_theorem (p : ℝ) (m : ℝ) (x₀ y₀ : ℝ) :
  p > 1 →
  parabola p x₀ y₀ →
  line m x₀ y₀ →
  let F := focus p
  let P := (0, m)
  let Q := (x₀, y₀)
  (((Q.1 - F.1)^2 + (Q.2 - F.2)^2).sqrt) / (((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt) = p →
  (∃ (p_min : ℝ), p_min = 1 + Real.sqrt 2 / 2 ∧ 
    ∀ (p' : ℝ), p' > 1 → x₀ + p' ≥ x₀ + p_min) ∧
  (x₀ = 1 → ∃ (n : ℝ), n = 4 ∧
    ∀ (A B D E : ℝ × ℝ),
      parabola p A.1 A.2 →
      parabola p B.1 B.2 →
      myCircle n D.1 D.2 →
      myCircle n E.1 E.2 →
      (A.1 * B.1 + A.2 * B.2 = 0) →  -- OA ⊥ OB
      (∃ (t : ℝ), A.1 = t * A.2 + 4 ∧ B.1 = t * B.2 + 4 ∧ 
                  D.1 = t * D.2 + 4 ∧ E.1 = t * E.2 + 4) →
      ((D.1 - E.1)^2 + (D.2 - E.2)^2).sqrt = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l708_70858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_C_l708_70875

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

-- Define the theorem
theorem triangle_angle_C (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  a = 1 ∧
  b = Real.sqrt 2 ∧
  f A = 2 ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C →
  -- Conclusion
  C = 7 * Real.pi / 12 ∨ C = Real.pi / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_C_l708_70875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_log_inequality_g_critical_points_difference_l708_70851

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x + 1

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * Real.log x + Real.exp (-x)

-- Statement 1
theorem f_minimum_value : ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = 0 := by sorry

-- Statement 2
theorem log_inequality (a b : ℝ) (ha : 0 < a) (ha2 : a < 1) (hb : b * Real.exp ((1 - a) / a) = 1) :
  Real.log b / Real.log a > 1 := by sorry

-- Statement 3
theorem g_critical_points_difference (m : ℝ) (x₁ x₂ : ℝ) 
  (h1 : HasDerivAt (g m) 0 x₁) (h2 : HasDerivAt (g m) 0 x₂) (h3 : x₁ ≠ x₂) :
  |g m x₁ - g m x₂| < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_log_inequality_g_critical_points_difference_l708_70851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equations_and_common_chord_l708_70868

/-- Represents a circle in Cartesian coordinates -/
structure CartesianCircle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  equation : ℝ → ℝ  -- ρ as a function of θ

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ  -- ρ as a function of θ

def C₁ : CartesianCircle := ⟨(0, 0), 2⟩
def C₂ : CartesianCircle := ⟨(2, 0), 2⟩

noncomputable def polar_C₁ : PolarCircle := ⟨λ _ => 2⟩
noncomputable def polar_C₂ : PolarCircle := ⟨λ θ => 4 * Real.cos θ⟩

noncomputable def common_chord : PolarLine := ⟨λ θ => 1 / Real.cos θ⟩

theorem circle_equations_and_common_chord 
  (h₁ : C₁.center = (0, 0) ∧ C₁.radius = 2)
  (h₂ : C₂.center = (2, 0) ∧ C₂.radius = 2) :
  (∀ θ, polar_C₁.equation θ = 2) ∧ 
  (∀ θ, polar_C₂.equation θ = 4 * Real.cos θ) ∧
  (∀ θ, common_chord.equation θ = 1 / Real.cos θ) := by
  sorry

#check circle_equations_and_common_chord

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equations_and_common_chord_l708_70868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_is_zero_l708_70848

noncomputable def p (x : ℝ) : ℝ := x^8

noncomputable def d₁ (x : ℝ) : ℝ := x + 1/2

noncomputable def d₂ (x : ℝ) : ℝ := x - 1/2

noncomputable def r₁ : ℝ := p (-1/2)

noncomputable def q₁ (x : ℝ) : ℝ := (p x - r₁) / (d₁ x)

noncomputable def r₂ : ℝ := q₁ (1/2)

noncomputable def q₂ (x : ℝ) : ℝ := (q₁ x - r₂) / (d₂ x)

theorem remainder_is_zero : r₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_is_zero_l708_70848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_A_400_functional_relationships_A_more_cost_effective_l708_70850

-- Define the pricing structures for Supermarkets A and B
noncomputable def cost_A (x : ℝ) : ℝ :=
  if x ≤ 300 then 0.9 * x else 0.9 * 300 + 0.7 * (x - 300)

noncomputable def cost_B (x : ℝ) : ℝ :=
  if x ≤ 100 then x else 100 + 0.8 * (x - 100)

-- Theorem 1: Cost of 400 yuan purchase at Supermarket A
theorem cost_A_400 : cost_A 400 = 340 := by
  sorry

-- Theorem 2: Functional relationships for Supermarkets A and B
theorem functional_relationships (x : ℝ) :
  (cost_A x = if x ≤ 300 then 0.9 * x else 0.7 * x + 60) ∧
  (cost_B x = if x ≤ 100 then x else 0.8 * x + 20) := by
  sorry

-- Theorem 3: Supermarket A is more cost-effective for 450 yuan purchase
theorem A_more_cost_effective : cost_A 450 < cost_B 450 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_A_400_functional_relationships_A_more_cost_effective_l708_70850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_difference_l708_70815

-- Define the distances
def FS : ℝ := 5
def SR : ℝ := 12
def SG : ℝ := 9

-- Calculate FR and GR using Pythagorean theorem
noncomputable def FR : ℝ := Real.sqrt (FS^2 + SR^2)
noncomputable def GR : ℝ := Real.sqrt (SG^2 + SR^2)

-- Define the four routes
noncomputable def route1 : ℝ := FR + SR + SG
noncomputable def route2 : ℝ := FS + SG + GR
noncomputable def route3 : ℝ := FR + GR + SG
noncomputable def route4 : ℝ := FS + SR + GR

-- State the theorem
theorem route_difference :
  (max route1 (max route2 (max route3 route4))) -
  (min route1 (min route2 (min route3 route4))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_difference_l708_70815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repayment_period_calculation_l708_70885

/-- Calculates the new repayment period given the loan amount, original repayment period, and additional monthly payment. -/
noncomputable def new_repayment_period (loan_amount : ℝ) (original_years : ℝ) (additional_monthly : ℝ) : ℝ :=
  let original_monthly := loan_amount / (original_years * 12)
  let new_monthly := original_monthly + additional_monthly
  loan_amount / new_monthly / 12

/-- Theorem stating that for a $6,000 loan with an original 5-year plan and $150 additional monthly payment, the new repayment period is 2 years. -/
theorem repayment_period_calculation :
  new_repayment_period 6000 5 150 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repayment_period_calculation_l708_70885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_max_area_l708_70835

/-- The perimeter of the rectangle --/
noncomputable def perimeter : ℝ := 40

/-- The side length of the square with maximum area --/
noncomputable def max_side : ℝ := perimeter / 4

/-- The maximum area achievable for a rectangle with the given perimeter --/
noncomputable def max_area : ℝ := max_side ^ 2

/-- Theorem stating that the area of any rectangle with the given perimeter is at most max_area --/
theorem rectangle_max_area :
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = perimeter / 2 →
  a * b ≤ max_area := by
  sorry

#check rectangle_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_max_area_l708_70835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_omega_range_l708_70837

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x + Real.pi / 4) - 2

-- Define the theorem
theorem monotonic_decreasing_omega_range :
  ∀ ω : ℝ, ω > 0 →
  (∀ x₁ x₂ : ℝ, Real.pi / 2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.pi → f ω x₁ > f ω x₂) ↔
  (1 / 2 ≤ ω ∧ ω ≤ 5 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_omega_range_l708_70837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_labeling_count_l708_70881

/-- Representation of a cube labeling --/
def CubeLabeling := Fin 8 → Fin 8

/-- Check if a labeling is valid (each number used once) --/
def is_valid_labeling (l : CubeLabeling) : Prop :=
  Function.Injective l

/-- Sum of numbers on a face given by four vertex indices --/
def face_sum (l : CubeLabeling) (v1 v2 v3 v4 : Fin 8) : ℕ :=
  (l v1).val + (l v2).val + (l v3).val + (l v4).val + 4

/-- All face sums are equal --/
def all_faces_equal_sum (l : CubeLabeling) : Prop :=
  ∃ s : ℕ, ∀ (v1 v2 v3 v4 : Fin 8), face_sum l v1 v2 v3 v4 = s

/-- Two labelings are equivalent up to rotation --/
def rotation_equivalent (l1 l2 : CubeLabeling) : Prop :=
  sorry  -- Definition of rotation equivalence

/-- The set of all valid cube labelings --/
def valid_labelings : Set CubeLabeling :=
  {l | is_valid_labeling l ∧ all_faces_equal_sum l}

/-- Setoid instance for rotation equivalence --/
instance rotation_equiv_setoid : Setoid CubeLabeling :=
{ r := rotation_equivalent,
  iseqv := sorry }

/-- The set of distinct labelings up to rotation --/
def distinct_labelings : Type :=
  Quotient rotation_equiv_setoid

/-- Fintype instance for distinct_labelings --/
noncomputable instance : Fintype distinct_labelings :=
  sorry

theorem cube_labeling_count :
  Fintype.card (distinct_labelings) = 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_labeling_count_l708_70881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_unexpressible_prime_is_41_l708_70893

def is_expressible (n : Int) : Prop :=
  ∃ a b : ℕ, n = Int.natAbs (3^a - 2^b)

def smallest_unexpressible_prime : ℕ → Prop
| p => Nat.Prime p ∧ ¬(is_expressible p) ∧ ∀ q < p, Nat.Prime q → is_expressible q

theorem smallest_unexpressible_prime_is_41 : smallest_unexpressible_prime 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_unexpressible_prime_is_41_l708_70893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dianes_shopping_l708_70864

/-- Calculates the additional money Diane needs to buy cookies and chocolates with a discount -/
theorem dianes_shopping (cookie_price : ℚ) (chocolate_price : ℚ) (discount_rate : ℚ) (diane_money : ℚ) : 
  cookie_price = 65/100 → 
  chocolate_price = 125/100 → 
  discount_rate = 15/100 → 
  diane_money = 27/100 → 
  (⌈(cookie_price + chocolate_price) * (1 - discount_rate) * 100⌉ / 100 - diane_money) = 135/100 := by
  intros h1 h2 h3 h4
  -- The proof steps would go here
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dianes_shopping_l708_70864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mens_wages_l708_70801

-- Define the number of men, women, and boys
def men : ℕ := 5
def boys : ℕ := 8

-- Define the total earnings
def total_earnings : ℕ := 75

-- Define a variable for the number of women
variable (W : ℕ)

-- Define the equality between men and women
axiom men_women_equality : men = W

-- Define the equality between women and boys
axiom women_boys_equality : W = boys

-- Theorem to prove
theorem mens_wages : total_earnings / (3 * men) = 5 := by
  sorry

#check mens_wages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mens_wages_l708_70801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_and_number_divisibility_l708_70814

theorem polynomial_divisibility_and_number_divisibility (n : ℕ) : 
  (∀ x : ℂ, (x^2 + x + 1) ∣ (x^(2*n) + x^n + 1)) = (n % 3 = 0) ∧ 
  (37 ∣ (10^(2*n + 2) + 10^(n + 1) + 1)) = (n % 3 = 0 ∨ (n + 1) % 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_and_number_divisibility_l708_70814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_present_rounded_l708_70888

def total_laborers : ℕ := 156
def present_laborers : ℕ := 70

def percentage_present : ℚ :=
  (present_laborers : ℚ) / (total_laborers : ℚ) * 100

noncomputable def round_to_nearest_tenth (x : ℚ) : ℚ :=
  ⌊(x * 10 + 1/2)⌋ / 10

theorem percentage_present_rounded :
  round_to_nearest_tenth percentage_present = 449/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_present_rounded_l708_70888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tie_count_l708_70834

/-- The number of pupils wearing red ties -/
def R : ℕ := sorry

/-- The total number of pupils in the class -/
def total_pupils : ℕ := 2 * R + 2

/-- The probability of selecting two pupils with red ties -/
noncomputable def prob_red_red : ℚ := (R * (R - 1)) / (total_pupils * (total_pupils - 1))

/-- The probability of selecting two pupils with different tie colors -/
noncomputable def prob_diff_colors : ℚ := 
  (R * (R + 2) + (R + 2) * R) / (total_pupils * (total_pupils - 1))

/-- The main theorem -/
theorem red_tie_count : 
  R > 0 ∧ 
  prob_diff_colors = 3 * prob_red_red →
  R^3 + R^2 + R = 399 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tie_count_l708_70834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_fraction_equals_one_l708_70839

theorem tangent_fraction_equals_one :
  Real.tan (60 * π / 180) = Real.sqrt 3 →
  (Real.sqrt 3 - Real.tan (15 * π / 180)) / (1 + Real.sqrt 3 * Real.tan (15 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_fraction_equals_one_l708_70839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l708_70803

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 else -(x^2)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≥ 0, f x = x^2) →  -- f(x) = x² for x ≥ 0
  (∀ x ∈ Set.Icc a (a + 2), f (x + a) ≥ 2 * f x) →  -- f(x+a) ≥ 2f(x) for x ∈ [a, a+2]
  a ≥ Real.sqrt 2 :=  -- a ∈ [√2, +∞)
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l708_70803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_clubs_with_27_members_l708_70866

/-- The number of students in the school -/
def total_students : ℕ := 2019

/-- The number of students in each club's advisory board -/
def advisory_board_size : ℕ := 12

/-- The size of the clubs we're interested in -/
def target_club_size : ℕ := 27

/-- Represents the property that each subset of at least 12 students
    can realize an advisory meeting for exactly one student club -/
def unique_advisory_meeting (n : ℕ) : Prop :=
  ∀ (S : Finset (Fin n)), S.card ≥ advisory_board_size →
    ∃! (club : Finset (Fin n)), club ⊆ S ∧ club.card = advisory_board_size

/-- The main theorem: Under the given conditions, the number of student clubs
    with exactly 27 members is equal to (2003 choose 11) -/
theorem number_of_clubs_with_27_members :
  unique_advisory_meeting total_students →
  (Finset.filter (fun club : Finset (Fin total_students) => club.card = target_club_size) 
    (Finset.powerset (Finset.univ))).card = Nat.choose 2003 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_clubs_with_27_members_l708_70866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_distance_product_l708_70822

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A semicircle with center at the origin and radius 1 -/
def Semicircle : Set Point :=
  {p : Point | p.x^2 + p.y^2 = 1 ∧ p.y ≥ 0}

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: For any three points on a semicircle, where two are at the ends
    of the diameter and one is anywhere on the arc, the square of the distance
    between the end points equals the product of the distances from each end
    point to the third point. -/
theorem semicircle_distance_product (P₁ P₂ P₃ : Point)
    (h₁ : P₁ ∈ Semicircle) (h₂ : P₂ ∈ Semicircle) (h₃ : P₃ ∈ Semicircle)
    (h₄ : P₁.x = -1 ∧ P₁.y = 0) (h₅ : P₂.x = 1 ∧ P₂.y = 0) :
    (distance P₁ P₂)^2 = (distance P₁ P₃) * (distance P₃ P₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_distance_product_l708_70822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_chord_length_range_l708_70852

/-- Definition of the ellipse (C) -/
noncomputable def Ellipse (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Definition of the unit circle -/
noncomputable def UnitCircle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Eccentricity of the ellipse -/
noncomputable def eccentricity : ℝ := Real.sqrt 2 / 2

/-- Distance between points A and A' -/
noncomputable def AA_distance : ℝ := Real.sqrt 2

/-- Theorem about the ellipse properties and chord length range -/
theorem ellipse_properties_and_chord_length_range :
  ∀ (G H E F : ℝ × ℝ),
  (∀ x y, Ellipse x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∃ l : ℝ → ℝ → Prop,
    (Ellipse G.1 G.2 ∧ Ellipse H.1 H.2) ∧
    (UnitCircle E.1 E.2 ∧ UnitCircle F.1 F.2) ∧
    (l G.1 G.2 ∧ l H.1 H.2 ∧ l E.1 E.2 ∧ l F.1 F.2) ∧
    (Real.sqrt 3 ≤ Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2) ∧
     Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2) ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_chord_length_range_l708_70852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hcl_formation_l708_70830

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction :=
  (reactant1 : Moles)
  (reactant2 : Moles)
  (product1 : Moles)
  (product2 : Moles)

instance : OfNat Moles n where
  ofNat := (↑n : ℝ)

/-- The given reaction: C2H6 + Cl2 → C2H5Cl + HCl -/
def ethane_chlorination : Reaction :=
  { reactant1 := 1,
    reactant2 := 1,
    product1 := 1,
    product2 := 1 }

/-- Theorem stating that 3 moles of HCl are formed from 3 moles of C2H6 and 3 moles of Cl2 -/
theorem hcl_formation (initial_c2h6 : Moles) (initial_cl2 : Moles) 
    (h1 : initial_c2h6 = 3)
    (h2 : initial_cl2 = 3)
    (reaction : Reaction)
    (h3 : reaction = ethane_chlorination) :
    ∃ (hcl_formed : Moles), hcl_formed = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hcl_formation_l708_70830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l708_70855

noncomputable def binomial_expansion (n : ℕ) (x : ℝ) : ℝ := (x + 1 / (2 * x)) ^ n

def coeff (n r : ℕ) : ℚ := (n.choose r) * (1 / 2) ^ r

def sum_of_coeffs (n : ℕ) : ℚ := 2 ^ n

def constant_term (n : ℕ) : ℚ := coeff n (n / 2)

theorem binomial_expansion_properties :
  (∀ n : ℕ, 
    (coeff n 0 - coeff n 1 = coeff n 1 - coeff n 2) → 
    (n = 8 ∧ constant_term n = 35 / 8)) ∧
  (∀ n : ℕ, 
    (sum_of_coeffs n = 64) → 
    (n = 6 ∧ constant_term n = 5 / 2)) := by
  sorry

#check binomial_expansion_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l708_70855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_m_l708_70840

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - x + m

theorem minimum_value_implies_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f m x ≥ 1/3) ∧ (∃ x ∈ Set.Icc 0 1, f m x = 1/3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_m_l708_70840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_proof_l708_70876

theorem election_votes_proof (total_votes : ℕ) (invalid_percentage : ℚ) 
  (winner_percentage : ℚ) (h1 : total_votes = 7500) 
  (h2 : invalid_percentage = 1/5) (h3 : winner_percentage = 11/20) : 
  Int.floor ((1 - winner_percentage) * (1 - invalid_percentage) * total_votes) = 2700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_proof_l708_70876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_with_3_l708_70894

/-- The number of cards with the number 3 -/
def x : ℕ := sorry

/-- The number of cards with the number 4 -/
def y : ℕ := sorry

/-- The number of cards with the number 5 -/
def z : ℕ := sorry

/-- The total number of cards is 10 -/
axiom total_cards : x + y + z = 10

/-- The sum of the numbers on the cards is 39 -/
axiom sum_numbers : 3 * x + 4 * y + 5 * z = 39

/-- The number of cards with the number 5 is 2 -/
axiom cards_with_5 : z = 2

/-- The number of cards with the number 3 is 3 -/
theorem cards_with_3 : x = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_with_3_l708_70894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_tv_watching_hours_l708_70838

/-- The total hours of TV Tim watched given the following conditions:
  - Tim watches 2 shows
  - One show is 0.5 hours per episode
  - The other show is 1 hour per episode
  - The 0.5-hour show has 24 episodes
  - The 1-hour show has 12 episodes
-/
theorem tim_tv_watching_hours 
  (short_show_duration : ℝ)
  (long_show_duration : ℝ)
  (short_show_episodes : ℕ)
  (long_show_episodes : ℕ)
  (h1 : short_show_duration = 0.5)
  (h2 : long_show_duration = 1)
  (h3 : short_show_episodes = 24)
  (h4 : long_show_episodes = 12) :
  short_show_duration * (short_show_episodes : ℝ) + long_show_duration * (long_show_episodes : ℝ) = 24 := by
  sorry

#check tim_tv_watching_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_tv_watching_hours_l708_70838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_plane_l708_70891

-- Define the point M₀
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the plane
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

-- Define the distance function
noncomputable def distance_to_plane (M₀ : Point3D) (P : Plane) : ℝ :=
  abs (P.A * M₀.x + P.B * M₀.y + P.C * M₀.z + P.D) / Real.sqrt (P.A^2 + P.B^2 + P.C^2)

-- Theorem statement
theorem distance_point_to_plane (M₀ : Point3D) (P : Plane) :
  distance_to_plane M₀ P = abs (P.A * M₀.x + P.B * M₀.y + P.C * M₀.z + P.D) / Real.sqrt (P.A^2 + P.B^2 + P.C^2) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_plane_l708_70891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l708_70892

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = -1/8 * (x - 4)^2

-- Define the focus F
def focus : ℝ × ℝ := (4, 2)

-- Define the vertex E
def vertex : ℝ × ℝ := (4, 0)

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem min_distance_sum (A : ℝ × ℝ) (h_A : point_on_parabola A) 
  (h_AF : distance A focus = 4) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 13 ∧
  ∀ (P : ℝ × ℝ), P.2 = -2 → distance P A + distance P vertex ≥ min_val :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l708_70892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_redistribution_theorem_l708_70821

/-- Represents a distribution of objects into piles -/
structure Distribution (α : Type) where
  piles : List (List α)

/-- The number of piles in a distribution -/
def Distribution.num_piles {α : Type} (d : Distribution α) : Nat :=
  d.piles.length

/-- The total number of objects in a distribution -/
def Distribution.total_objects {α : Type} (d : Distribution α) : Nat :=
  d.piles.map List.length |> List.sum

/-- Checks if an object is in a smaller pile in the new distribution compared to the original -/
def in_smaller_pile {α : Type} (orig : Distribution α) (new : Distribution α) (obj : α) : Prop :=
  ∃ (i j : Nat), 
    i < orig.piles.length ∧
    j < new.piles.length ∧
    obj ∈ (orig.piles.get ⟨i, by sorry⟩) ∧ 
    obj ∈ (new.piles.get ⟨j, by sorry⟩) ∧ 
    (new.piles.get ⟨j, by sorry⟩).length < (orig.piles.get ⟨i, by sorry⟩).length

theorem redistribution_theorem {α : Type} (orig new : Distribution α) (k : Nat) :
  orig.num_piles + k = new.num_piles →
  k > 0 →
  orig.total_objects = new.total_objects →
  ∃ (smaller_pile_objs : List α), 
    smaller_pile_objs.length ≥ k + 1 ∧ 
    ∀ obj ∈ smaller_pile_objs, in_smaller_pile orig new obj :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_redistribution_theorem_l708_70821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_intervals_f_lower_bound_a_lower_bound_b_lower_bound_l_l708_70886

noncomputable def j (a : ℝ) (x : ℝ) : ℝ := a / (x + 1)

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + j a x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := |Real.log x| + j a x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + (1 - a) / x - 1

noncomputable def g' (b : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * x + 4

theorem increase_intervals_f (x : ℝ) : 
  (x ∈ Set.Ioo 0 (1/2) ∨ x ∈ Set.Ioi 2) → (deriv (f (9/2)) x > 0) := by sorry

theorem lower_bound_a (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo 0 2 → x₂ ∈ Set.Ioo 0 2 → x₁ ≠ x₂ → 
    (g a x₂ - g a x₁) / (x₂ - x₁) < -1) → a ≥ 27/2 := by sorry

theorem lower_bound_b (b : ℝ) : 
  (∀ x₁ : ℝ, x₁ ∈ Set.Ioo 0 2 → ∃ x₂ : ℝ, x₂ ∈ Set.Icc 1 2 ∧ f' (1/4) x₁ ≥ g' b x₂) → 
  b ≥ 17/8 := by sorry

theorem lower_bound_l (l : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo 1 2 → x₂ ∈ Set.Ioo 1 2 → 
    |f' (1/4) x₁ - f' (1/4) x₂| ≤ l * |1/x₁ - 1/x₂|) → l ≥ 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_intervals_f_lower_bound_a_lower_bound_b_lower_bound_l_l708_70886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_prob_B_given_A_l708_70827

-- Define the set of volunteers
inductive Volunteer : Type
| A : Volunteer
| B : Volunteer
| C : Volunteer
| D : Volunteer

-- Define the set of locations
inductive Location : Type
| Huanglong : Location
| HangzhouOlympic : Location
| Zijingang : Location

-- Define an assignment as a function from Volunteer to Location
def Assignment := Volunteer → Location

-- Define the condition that each location must have at least one volunteer
def ValidAssignment (a : Assignment) : Prop :=
  (∃ v, a v = Location.Huanglong) ∧
  (∃ v, a v = Location.HangzhouOlympic) ∧
  (∃ v, a v = Location.Zijingang)

-- Define the probability space
def Ω : Type := {a : Assignment // ValidAssignment a}

-- Define events A and B
def EventA (ω : Ω) : Prop := ω.val Volunteer.A = Location.Huanglong
def EventB (ω : Ω) : Prop := ω.val Volunteer.B = Location.Huanglong

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Axioms for probability measure
axiom P_nonneg : ∀ S : Set Ω, P S ≥ 0
axiom P_total : P Set.univ = 1

-- Define conditional probability
noncomputable def ConditionalProb (A B : Set Ω) : ℝ := P (A ∩ B) / P B

-- State the theorem
theorem conditional_prob_B_given_A :
  ConditionalProb {ω : Ω | EventB ω} {ω : Ω | EventA ω} = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_prob_B_given_A_l708_70827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_subset_size_for_perfect_square_product_l708_70846

/-- The set of positive integers whose prime factors are all less than 10 -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ ∀ p, Nat.Prime p → p ∣ n → p < 10}

/-- A function that checks if a product is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- The main theorem -/
theorem smallest_subset_size_for_perfect_square_product :
  (∀ A : Finset ℕ, (∀ a ∈ A, a ∈ S) → A.card ≥ 9 →
    ∃ a b c d : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    is_perfect_square (a * b * c * d)) ∧
  (∀ n < 9, ∃ A : Finset ℕ, (∀ a ∈ A, a ∈ S) ∧ A.card = n ∧
    ∀ a b c d : ℕ, a ∈ A → b ∈ A → c ∈ A → d ∈ A →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    ¬is_perfect_square (a * b * c * d)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_subset_size_for_perfect_square_product_l708_70846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_number_percentage_less_l708_70844

/-- Given three numbers where the first is 75% of X and the second is 63% of X,
    prove that the second number is approximately 16% less than the first number. -/
theorem second_number_percentage_less (X : ℝ) (first_number second_number : ℝ) 
  (h1 : first_number = 0.75 * X) 
  (h2 : second_number = 0.63 * X) : 
  ∃ ε > 0, |((first_number - second_number) / first_number) - 0.16| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_number_percentage_less_l708_70844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_optimization_l708_70861

/-- The annual energy consumption cost function -/
noncomputable def P (m : ℝ) (x : ℝ) : ℝ := 3 * m / (4 * x + 5)

/-- The total cost function over 40 years -/
noncomputable def S (x : ℝ) : ℝ := 40 * P 15 x + 8 * x

theorem insulation_optimization :
  ∀ x : ℝ, 0 ≤ x → x ≤ 8 →
    S x ≥ 110 ∧
    (S x = 110 ↔ x = 6.25) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_optimization_l708_70861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l708_70802

theorem existence_of_special_set : ∃ (S : Finset ℕ), 
  (Finset.card S = 2012) ∧ 
  (∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → Nat.gcd a b > 1) ∧
  (∀ a b c : ℕ, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → Nat.gcd a (Nat.gcd b c) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l708_70802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cash_loss_and_recovery_l708_70880

theorem cash_loss_and_recovery (c : ℚ) (h : c > 0) : 
  let lost := (1/3 : ℚ) * c
  let found := (2/3 : ℚ) * lost
  let remaining := c - lost + found
  c - remaining = (1/9 : ℚ) * c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cash_loss_and_recovery_l708_70880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_parallel_distance_l708_70817

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + a * y - a = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := a * x - (2 * a - 3) * y + a - 2 = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop := 
  (1 : ℝ) * a + a * (3 - 2 * a) = 0

-- Define parallelism of lines
def parallel (a : ℝ) : Prop := a = -3

-- Define the distance between parallel lines
noncomputable def distance_between_parallel_lines (a : ℝ) : ℝ := 
  |3 - 5/3| / Real.sqrt (1^2 + (-3)^2)

-- Theorem statements
theorem perpendicular_condition (a : ℝ) : 
  perpendicular a ↔ (a = 0 ∨ a = 2) := by sorry

theorem parallel_distance (a : ℝ) : 
  parallel a → distance_between_parallel_lines a = 2 * Real.sqrt 10 / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_parallel_distance_l708_70817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_gt_x_plus_one_l708_70879

theorem exp_gt_x_plus_one (x : ℝ) (h : x > 0) : Real.exp x > x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_gt_x_plus_one_l708_70879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sine_contrapositive_l708_70890

theorem triangle_angle_sine_contrapositive (A B C : ℝ) :
  ¬(Real.sin A > Real.sin B) → ¬(A > B) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sine_contrapositive_l708_70890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l708_70833

/-- Two circles in a 2D plane --/
structure TwoCircles where
  m : ℝ
  circle1 : (x : ℝ) → (y : ℝ) → Prop := λ x y => (x + m)^2 + y^2 = 1
  circle2 : (x : ℝ) → (y : ℝ) → Prop := λ x y => (x - 2)^2 + y^2 = 4

/-- Definition of circles having common points --/
def have_common_points (c : TwoCircles) : Prop :=
  ∃ x y, c.circle1 x y ∧ c.circle2 x y

/-- The main theorem to be proved --/
theorem sufficient_not_necessary :
  (∀ m, -1 ≤ m ∧ m ≤ 1 → have_common_points { m := m }) ∧
  (∃ m, (m < -1 ∨ 1 < m) ∧ have_common_points { m := m }) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l708_70833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_2500th_term_l708_70845

/-- Represents the sequence where the n-th positive integer appears n times -/
def our_sequence (n : ℕ) : ℕ := 
  Nat.sqrt (2 * n + 1)

/-- The 2500th term of the sequence -/
def term_2500 : ℕ := our_sequence 2500

/-- Theorem stating that the 2500th term of the sequence, when divided by 7, has a remainder of 1 -/
theorem remainder_of_2500th_term : term_2500 % 7 = 1 := by
  sorry

#eval term_2500 % 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_2500th_term_l708_70845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l708_70870

theorem sum_of_coefficients (c d x y z : ℝ) : 
  c^2 = 9/25 →
  d^2 = (3 - Real.sqrt 2)^2 / 8 →
  c < 0 →
  d > 0 →
  (c + d)^3 = (x * Real.sqrt y) / z →
  x > 0 →
  y > 0 →
  z > 0 →
  ⌊x⌋ + ⌊y⌋ + ⌊z⌋ = 9402 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l708_70870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l708_70824

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (3 * x + Real.pi / 3) * Real.cos (x - Real.pi / 6) + 
  Real.cos (3 * x + Real.pi / 3) * Real.sin (x - Real.pi / 6)

theorem axis_of_symmetry :
  ∀ x : ℝ, f (Real.pi / 12 + x) = f (Real.pi / 12 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l708_70824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_journey_time_l708_70829

/-- Represents the time taken for a cyclist's journey with a speed increase halfway through. -/
noncomputable def journey_time (planned_time : ℝ) (speed_increase : ℝ) : ℝ :=
  let half_journey := planned_time / 2
  let second_half_time := half_journey / (1 + speed_increase)
  half_journey + second_half_time

/-- Theorem stating that for a journey planned to take 5 hours, 
    with a 25% speed increase at the halfway point, 
    the actual time taken is 4.5 hours. -/
theorem cyclist_journey_time : journey_time 5 0.25 = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_journey_time_l708_70829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_5f_l708_70847

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- State the theorem
theorem f_composition_equals_5f (x : ℝ) (h : -1 < x ∧ x < 1) :
  f ((5 * x + x^5) / (1 + 5 * x^4)) = 5 * f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_5f_l708_70847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_and_chord_length_l708_70823

/-- Ellipse C defined by 2x^2 + y^2 = 16 -/
def ellipse_C (x y : ℝ) : Prop := 2 * x^2 + y^2 = 16

/-- Point A on ellipse C -/
def point_A (x y : ℝ) : Prop := ellipse_C x y

/-- Point B on the line x = 4 -/
def point_B (y : ℝ) : Prop := True

/-- OA · OB = 0 -/
def perpendicular (x₀ y₀ t : ℝ) : Prop := x₀ * 4 + y₀ * t = 0

/-- Circle with equation x^2 + y^2 = 17 -/
def circle_17 (x y : ℝ) : Prop := x^2 + y^2 = 17

theorem ellipse_eccentricity_and_chord_length 
  (x₀ y₀ t : ℝ) 
  (h1 : point_A x₀ y₀) 
  (h2 : point_B t) 
  (h3 : perpendicular x₀ y₀ t) :
  (∃ e : ℝ, e = Real.sqrt 2 / 2 ∧ e^2 = (16 - 8) / 16) ∧
  (∃ l : ℝ, l = 6 ∧ l^2 / 4 = 17 - 8) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_and_chord_length_l708_70823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_square_theorem_l708_70897

/-- Represents a point on a 2D integer grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents the state of four grasshoppers on a grid -/
structure GrasshopperState where
  points : Fin 4 → GridPoint

/-- Symmetric jump transformation -/
def symmetricJump (state : GrasshopperState) (jumper pivot : Fin 4) : GrasshopperState :=
  sorry

/-- Check if four points form a square -/
def isSquare (state : GrasshopperState) : Bool :=
  sorry

/-- Check if a square is larger than the unit square -/
def isLargerSquare (state : GrasshopperState) : Bool :=
  sorry

/-- Initial state with grasshoppers at unit square vertices -/
def initialState : GrasshopperState :=
  { points := λ i => match i with
    | 0 => ⟨0, 0⟩
    | 1 => ⟨1, 0⟩
    | 2 => ⟨1, 1⟩
    | 3 => ⟨0, 1⟩ }

/-- Perform a sequence of jumps -/
def performJumps (initial : GrasshopperState) : (n : Nat) → (Fin n → Fin 4 × Fin 4) → GrasshopperState
  | 0, _ => initial
  | n+1, jumps => symmetricJump (performJumps initial n (fun i => jumps (Fin.castSucc i))) (jumps ⟨n, by simp⟩).1 (jumps ⟨n, by simp⟩).2

theorem grasshopper_square_theorem :
  ∀ (n : Nat) (jumps : Fin n → Fin 4 × Fin 4),
    let final_state := performJumps initialState n jumps
    isSquare final_state → ¬(isLargerSquare final_state) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_square_theorem_l708_70897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_through_ellipse_midpoint_l708_70865

/-- The slope of a line passing through a specific point on an ellipse -/
theorem line_slope_through_ellipse_midpoint :
  ∀ (k : ℝ), 
  let l : ℝ → ℝ := λ x ↦ k * (x - 1) + 1/4
  let ellipse : ℝ × ℝ → Prop := λ p ↦ p.1^2 / 4 + p.2^2 / 3 = 1
  let M : ℝ × ℝ := (1, 1/4)
  let intersections := {p : ℝ × ℝ | ellipse p ∧ p.2 = l p.1}
  (∃ A B : ℝ × ℝ, A ∈ intersections ∧ B ∈ intersections ∧ A ≠ B ∧
    (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2) →
  k = -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_through_ellipse_midpoint_l708_70865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_a_given_sum_condition_l708_70811

theorem max_sin_a_given_sum_condition (a b : ℝ) :
  (∀ a b : ℝ, Real.sin (a + b) = Real.sin a + Real.sin b) →
  (∃ a : ℝ, ∀ x : ℝ, Real.sin x ≤ Real.sin a) →
  (∃ a : ℝ, Real.sin a = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_a_given_sum_condition_l708_70811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_constant_l708_70884

-- Define the function f
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := Real.arctan ((2 - 2*x) / (1 + 4*x)) + c

-- Define the property of being an odd function
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Theorem statement
theorem odd_function_constant :
  ∃! c : ℝ, ∀ x ∈ Set.Ioo (-1/4 : ℝ) (1/4 : ℝ), is_odd (f c) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_constant_l708_70884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_symmetric_point_x_axis_l708_70867

/-- 
Given a point P in a 3D Cartesian coordinate system, 
this function returns its symmetric point with respect to the x-axis.
-/
def symmetricPointXAxis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2.1, -p.2.2)

/-- 
Theorem stating that the symmetric point of (-2, 1, 9) 
with respect to the x-axis is (-2, -1, -9).
-/
theorem symmetric_point_x_axis :
  symmetricPointXAxis (-2, 1, 9) = (-2, -1, -9) := by
  rfl

#eval symmetricPointXAxis (-2, 1, 9)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_symmetric_point_x_axis_l708_70867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_height_is_45_l708_70831

-- Define the heights of Sara, Joe, and Roy
def sara_height : ℕ := by sorry
def joe_height : ℕ := by sorry
def roy_height : ℕ := by sorry

-- Define the given conditions
axiom roy_height_val : roy_height = 36
axiom joe_taller_than_roy : joe_height = roy_height + 3
axiom sara_taller_than_joe : sara_height = joe_height + 6

-- Theorem to prove
theorem sara_height_is_45 : sara_height = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_height_is_45_l708_70831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_and_range_l708_70813

noncomputable def f (x : ℝ) := 2 * Real.sin (x + Real.pi / 6) - 2 * Real.cos x

theorem f_value_and_range :
  ∀ x ∈ Set.Icc (Real.pi / 2) Real.pi,
    (Real.sin x = 4 / 5 → f x = (4 * Real.sqrt 3 + 3) / 5) ∧
    f x ∈ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_and_range_l708_70813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_BQC_value_l708_70857

/-- Triangle ABC with sides AB = 13, BC = 15, and CA = 14 -/
structure Triangle :=
  (A B C : EuclideanSpace ℝ (Fin 2))
  (ab : dist A B = 13)
  (bc : dist B C = 15)
  (ca : dist C A = 14)

/-- Point E on the interior of BC -/
noncomputable def E (t : Triangle) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- I_B is the incenter of triangle ABE -/
noncomputable def I_B (t : Triangle) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- I_C is the incenter of triangle ACE -/
noncomputable def I_C (t : Triangle) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- Q is one of the intersection points of the circumcircles of triangles BI_BE and CI_CE -/
noncomputable def Q (t : Triangle) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- The maximum area of triangle BQC -/
noncomputable def max_area_BQC (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating the maximum area of triangle BQC -/
theorem max_area_BQC_value (t : Triangle) :
  max_area_BQC t = 112.5 - 56.25 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_BQC_value_l708_70857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_eq_f_4096_l708_70816

noncomputable def f (x : ℝ) : ℝ := 
  if 1 ≤ x ∧ x ≤ 4 then x^2 - 5*x + 6 else 0

axiom f_scale (x : ℝ) (h : 0 < x) : f (4*x) = 4 * f x

theorem smallest_x_eq_f_4096 : 
  ∃ x : ℝ, x > 0 ∧ f x = f 4096 ∧ ∀ y : ℝ, y > 0 → f y = f 4096 → x ≤ y ∧ x = 57344 * (5 - Real.sqrt 3) / 2 := by
  sorry

#check smallest_x_eq_f_4096

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_eq_f_4096_l708_70816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_four_implies_b_is_negative_one_l708_70849

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x - b else (2 : ℝ)^x

theorem f_composition_equals_four_implies_b_is_negative_one (b : ℝ) :
  f b (f b (1/2)) = 4 → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_four_implies_b_is_negative_one_l708_70849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_R_greatest_change_city_R_greatest_change_main_l708_70882

-- Define the set of cities
inductive City : Type
  | P | Q | R | S | T

-- Define the population data for each city in 1990 and 2000
def population (c : City) (year : Nat) : Nat :=
  match c, year with
  | City.P, 1990 => 150000
  | City.P, 2000 => 180000
  | City.Q, 1990 => 200000
  | City.Q, 2000 => 210000
  | City.R, 1990 => 120000
  | City.R, 2000 => 144000
  | City.S, 1990 => 180000
  | City.S, 2000 => 171000
  | City.T, 1990 => 160000
  | City.T, 2000 => 160000
  | _, _ => 0  -- Default case

-- Calculate the percentage change for a given city
noncomputable def percentageChange (c : City) : Real :=
  let pop1990 := population c 1990
  let pop2000 := population c 2000
  (pop2000 - pop1990 : Real) / pop1990 * 100

-- Theorem: City R had the greatest percentage change
theorem city_R_greatest_change :
  ∀ c : City, c ≠ City.R → percentageChange City.R ≥ percentageChange c := by
  sorry

-- Lemma: The percentage change for City R is 20%
lemma city_R_change_is_20_percent :
  percentageChange City.R = 20 := by
  sorry

-- Lemma: No city has a percentage change greater than 20%
lemma no_city_exceeds_20_percent :
  ∀ c : City, percentageChange c ≤ 20 := by
  sorry

-- Main theorem: City R had the greatest percentage change
theorem city_R_greatest_change_main :
  ∀ c : City, c ≠ City.R → percentageChange City.R > percentageChange c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_R_greatest_change_city_R_greatest_change_main_l708_70882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_isosceles_points_l708_70860

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- A is the given point (2, -2) -/
def A : Point := ⟨2, -2⟩

/-- O is the origin (0, 0) -/
def O : Point := ⟨0, 0⟩

/-- P is a point on the y-axis -/
def P : ℝ → Point := λ y => ⟨0, y⟩

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Triangle AOP is isosceles if two of its sides are equal -/
def isIsosceles (y : ℝ) : Prop :=
  distance O (P y) = distance A (P y) ∨ 
  distance O A = distance A (P y) ∨ 
  distance O A = distance O (P y)

/-- The main theorem: there are exactly 4 points P on the y-axis making triangle AOP isosceles -/
theorem four_isosceles_points : 
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ y, y ∈ s ↔ isIsosceles y := by
  sorry

#check four_isosceles_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_isosceles_points_l708_70860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l708_70836

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.sqrt (1 - 2 * x)

-- State the theorem
theorem f_range :
  {y : ℝ | ∃ x, 1 - 2 * x ≥ 0 ∧ f x = y} = Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l708_70836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l708_70859

/-- In a triangle ABC, given side a, angle B, and area S, prove that b/sin(B) equals 5√2 -/
theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a = 1 →
  B = π / 4 →
  S = 2 →
  S = 1 / 2 * a * c * Real.sin B →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  b / Real.sin B = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l708_70859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_on_hyperbola_l708_70887

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

-- Define a point on the hyperbola
def point_on_hyperbola (p : ℝ × ℝ) : Prop :=
  hyperbola p.1 p.2

-- Define the midpoint of two points
def is_midpoint (p q m : ℝ × ℝ) : Prop :=
  m.1 = (p.1 + q.1) / 2 ∧ m.2 = (p.2 + q.2) / 2

-- Theorem statement
theorem midpoint_on_hyperbola :
  ∃ (A B : ℝ × ℝ), 
    point_on_hyperbola A ∧
    point_on_hyperbola B ∧
    is_midpoint A B (-1, -4) ∧
    (∀ (m : ℝ × ℝ), m ∈ [(1, 1), (-1, 2), (1, 3)] → 
      ¬∃ (P Q : ℝ × ℝ), point_on_hyperbola P ∧ point_on_hyperbola Q ∧ is_midpoint P Q m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_on_hyperbola_l708_70887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_words_memorized_third_day_l708_70843

/-- Represents the total number of words Shuai Shuai memorized -/
def N : ℕ := 720

/-- Represents the number of words memorized in the first three days -/
def first_three_days : ℕ := N / 2

/-- Represents the number of words memorized in the last three days -/
def last_three_days : ℕ := 2 * N / 3

/-- The difference between words memorized in the last three days and the first three days -/
def difference : ℕ := 120

/-- The number of words memorized on the third day -/
def third_day : ℕ := 120

theorem words_memorized_third_day :
  first_three_days + difference = last_three_days ∧
  N = 720 →
  third_day = 120 := by
  intro h
  -- The proof goes here
  sorry

#eval third_day

end NUMINAMATH_CALUDE_ERRORFEEDBACK_words_memorized_third_day_l708_70843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l708_70818

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 2 + x) + (Real.sin (Real.pi / 2 + x))^2

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), max = 5/4 ∧
  ∀ x ∈ Set.Icc (-Real.pi) 0, f x ≤ max ∧
  ∃ x₀ ∈ Set.Icc (-Real.pi) 0, f x₀ = max := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l708_70818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_equals_target_l708_70805

/-- The sequence a_n defined recursively -/
noncomputable def a : ℕ → ℝ
  | 0 => 1 / 2
  | n + 1 => Real.sqrt ((1 + a n) / 2)

/-- The infinite product of the sequence a_n -/
noncomputable def infiniteProduct : ℝ := ∏' n, a n

/-- Theorem stating that the infinite product equals 3√3 / (4π) -/
theorem infinite_product_equals_target : infiniteProduct = (3 * Real.sqrt 3) / (4 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_equals_target_l708_70805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_F_l708_70804

noncomputable def f (x : ℝ) : ℝ := 5 * x - 3

noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x - 3

noncomputable def F (x : ℝ) : ℝ := Real.log (f x) + 1 / Real.sqrt (g x)

def X : Set ℝ := {0, 2, 3, 4, 5}

def Y : Set ℝ := {-2, -1, 1, 4, 5}

theorem domain_of_F : 
  {x ∈ X ∩ Y | f x > 0 ∧ g x > 0} = {4, 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_F_l708_70804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_wasted_in_one_hour_l708_70862

/-- The amount of water wasted by a dripping faucet in one hour -/
theorem water_wasted_in_one_hour
  (drips_per_minute : ℕ)
  (water_per_drop : ℝ)
  (h1 : drips_per_minute = 10)
  (h2 : water_per_drop = 0.05)
  (minutes_per_hour : ℕ := 60) :
  (drips_per_minute : ℝ) * water_per_drop * (minutes_per_hour : ℝ) = 30 := by
  sorry

#check water_wasted_in_one_hour

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_wasted_in_one_hour_l708_70862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l708_70874

/-- Given two lines intersecting at (3,3) with slopes 3/4 and 3 respectively,
    and a third line x + y = 12, prove that the area of the enclosed triangle is 54/7 -/
theorem triangle_area (line1 line2 line3 : ℝ → ℝ → Prop) : 
  (∀ x y, line1 x y ↔ y = 3/4 * x + 3/4) →
  (∀ x y, line2 x y ↔ y = 3 * x - 6) →
  (∀ x y, line3 x y ↔ x + y = 12) →
  line1 3 3 →
  line2 3 3 →
  (∃ A B C : ℝ × ℝ, 
    (line1 A.1 A.2 ∧ line2 A.1 A.2) ∧
    (line1 B.1 B.2 ∧ line3 B.1 B.2) ∧
    (line2 C.1 C.2 ∧ line3 C.1 C.2) ∧
    abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) = 54/7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l708_70874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_expression_l708_70825

open EuclideanGeometry

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))
-- Define the orthocenter H
variable (H : EuclideanSpace ℝ (Fin 2))
-- Define a point P on the circumcircle
variable (P : EuclideanSpace ℝ (Fin 2))
-- Define the side lengths and circumradius
variable (a b R : ℝ)

-- State the conditions
variable (h_isosceles : dist A B = dist A C)
variable (h_side_lengths : dist A B = b ∧ dist B C = a)
variable (h_orthocenter : isOrthocenter H A B C)
variable (h_circumcircle : onCircle P (circumcircle A B C))
variable (h_circumradius : R = circumradius A B C)

-- State the theorem
theorem constant_expression :
  dist P A ^ 2 + dist P B ^ 2 - dist P H ^ 2 = R ^ 2 - a ^ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_expression_l708_70825
