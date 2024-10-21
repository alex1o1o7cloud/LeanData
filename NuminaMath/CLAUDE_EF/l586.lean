import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_program_arrangements_l586_58611

theorem program_arrangements (n m : ℕ) (hn : n = 6) (hm : m = 3) : 
  (Nat.choose (n + 1) 1 * Nat.factorial m) + 
  (Nat.factorial (n + 1) / Nat.factorial (n + 1 - m)) + 
  (Nat.choose m 1 * Nat.choose (n + 1) 1 * Nat.choose n 1 * Nat.factorial (m - 1)) = 540 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_program_arrangements_l586_58611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_lines_l586_58647

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ 1}

-- Define the line l
def line_l : Set (ℝ × ℝ) := {p | p.1 + p.2 = 1}

-- Define the point P
def point_P : ℝ × ℝ := (2, 3)

-- State the theorem
theorem circle_and_tangent_lines :
  -- Given conditions
  (∃ (a b : ℝ × ℝ), a ∈ circle_C ∧ b ∈ circle_C ∧ a ∈ line_l ∧ b ∈ line_l ∧ 
   Real.sqrt 2 = Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)) →
  -- Conclusions
  ((∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - 1)^2 + (p.2 - 1)^2 = 1) ∧
   (∃ (t₁ t₂ : Set (ℝ × ℝ)), 
     (t₁ = {p : ℝ × ℝ | p.1 = 2}) ∧
     (t₂ = {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 + 6 = 0}) ∧
     (∀ p : ℝ × ℝ, p ∈ t₁ ∨ p ∈ t₂ ↔ 
       p ∉ circle_C ∧ ∃ q : ℝ × ℝ, q ∈ circle_C ∧ 
       ((p.1 - q.1) * (q.1 - 1) + (p.2 - q.2) * (q.2 - 1) = 0)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_lines_l586_58647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_time_period_round_time_period_l586_58600

/-- Compound interest calculation --/
theorem compound_interest_time_period 
  (P : ℝ) -- Principal amount
  (r : ℝ) -- Annual interest rate
  (n : ℝ) -- Number of times interest is compounded per year
  (CI : ℝ) -- Compound interest
  (h1 : P = 4000)
  (h2 : r = 0.15)
  (h3 : n = 1)
  (h4 : CI = 1554.5)
  : ∃ t : ℝ, 0 < t ∧ t < 3 ∧ P * (1 + r / n) ^ (n * t) = P + CI :=
by
  sorry

/-- Rounding the time period to the nearest whole number --/
theorem round_time_period 
  (t : ℝ)
  (h : 2 < t ∧ t < 3)
  : Int.floor t = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_time_period_round_time_period_l586_58600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equality_l586_58695

-- Define the functions
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)
noncomputable def f (x : ℝ) : ℝ := 7 - t x
def g (x : ℝ) : ℝ := x - 1

-- State the theorem
theorem composition_equality : t (g (f 9)) = Real.sqrt (32 - 5 * Real.sqrt 47) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equality_l586_58695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_7_l586_58637

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  decreasing : d < 0
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum_7 (seq : ArithmeticSequence) 
  (h1 : seq.a 3 = -1)
  (h2 : ∃ r, seq.a 4 = seq.a 1 * r ∧ -seq.a 6 = seq.a 4 * r) :
  sum_n seq 7 = -14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_7_l586_58637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_A_is_pi_over_three_triangle_perimeter_range_l586_58636

theorem triangle_angle_A_is_pi_over_three (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  (c = Real.sqrt 3 * a * Real.sin C - c * Real.cos A ∨
   Real.sin A ^ 2 - Real.sin B ^ 2 = Real.sin C ^ 2 - Real.sin B * Real.sin C ∨
   Real.tan B + Real.tan C - Real.sqrt 3 * Real.tan B * Real.tan C = -Real.sqrt 3) →
  A = π / 3 := by
sorry

theorem triangle_perimeter_range (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < π / 2) ∧ (0 < B) ∧ (B < π / 2) ∧ (0 < C) ∧ (C < π / 2) →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  a = Real.sqrt 3 →
  3 + Real.sqrt 3 < a + b + c ∧ a + b + c ≤ 3 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_A_is_pi_over_three_triangle_perimeter_range_l586_58636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_b_when_a_is_9_l586_58669

/-- The least possible value of b when a = 9, given the conditions -/
theorem least_possible_b_when_a_is_9 :
  ∀ b : ℕ+,
  (Nat.divisors 3).card = 3 →
  (Nat.divisors 9).card = 3 →
  (Nat.divisors b.val).card = 9 →
  (9 : ℕ+) ∣ b →
  2304 ≤ b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_b_when_a_is_9_l586_58669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_50_value_sequence_a_shift_l586_58618

def sequence_a : ℕ → ℕ
  | 0 => 3  -- We define a₀ = 3 to match the problem's a₁ = 3
  | n + 1 => sequence_a n + 2 * n + 4

theorem a_50_value : sequence_a 49 = 2649 := by
  sorry

-- Helper theorem to show the relationship between our definition and the problem statement
theorem sequence_a_shift : ∀ n : ℕ, sequence_a n = sequence_a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_50_value_sequence_a_shift_l586_58618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l586_58616

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4)

theorem f_properties :
  ∃ (axis_of_symmetry : ℤ → ℝ)
    (monotonic_increasing_intervals : ℤ → Set ℝ)
    (max_value min_value : ℝ),
  (∀ k : ℤ, axis_of_symmetry k = k * Real.pi / 2 + Real.pi / 8) ∧
  (∀ k : ℤ, monotonic_increasing_intervals k = Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8)) ∧
  (max_value = 1 ∧ min_value = -Real.sqrt 2) ∧
  (∀ x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4), f x ≤ max_value ∧ f x ≥ min_value) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l586_58616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_comp_domain_l586_58642

/-- The function f(x) = x / (1 - x) -/
noncomputable def f (x : ℝ) : ℝ := x / (1 - x)

/-- The composite function f^(n)(x) -/
noncomputable def f_comp (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => f (f_comp n x)

/-- The set of real numbers where f^(n)(x) is defined for all positive integers n -/
def S : Set ℝ := {x | ∀ n : ℕ+, IsRegular (f_comp n x)}

theorem f_comp_domain : S = (Set.univ : Set ℝ) \ {x | ∃ n : ℕ+, x = 1 / (n : ℝ)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_comp_domain_l586_58642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_interval_l586_58661

/-- The function f(x) = (x^3 - 125) / (x + 5) -/
noncomputable def f (x : ℝ) : ℝ := (x^3 - 125) / (x + 5)

/-- The domain of f is all real numbers except -5 -/
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -5} := by
  sorry

/-- The domain of f is (-∞, -5) ∪ (-5, +∞) -/
theorem domain_interval :
  {x : ℝ | ∃ y, f x = y} = Set.Iio (-5) ∪ Set.Ioi (-5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_interval_l586_58661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extrema_l586_58627

theorem sum_of_extrema (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10)
  (sum_squares_condition : p^2 + q^2 + r^2 + s^2 = 20) :
  let f := λ (a b c d : ℝ) => 3 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4)
  ∃ (min max : ℝ), 
    (∀ x y z w, x + y + z + w = 10 → x^2 + y^2 + z^2 + w^2 = 20 → f x y z w ≥ min) ∧
    (∀ x y z w, x + y + z + w = 10 → x^2 + y^2 + z^2 + w^2 = 20 → f x y z w ≤ max) ∧
    f p q r s ≥ min ∧
    f p q r s ≤ max ∧
    min = 40 ∧
    max = 52 ∧
    min + max = 92 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extrema_l586_58627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_figure_symmetry_l586_58680

/-- A structure representing a grid with figures -/
structure GridWithFigures where
  m : ℕ  -- Size of the grid (m × m)
  n : ℕ  -- Number of k-figures
  k : ℕ  -- Size of each k-figure
  h1 : m ^ 2 = n * k  -- The grid is fully covered by k-figures

/-- 
If a grid can be divided into n identical, non-overlapping k-figures, 
then it can also be divided into k identical, non-overlapping n-figures 
-/
theorem grid_figure_symmetry (g : GridWithFigures) : 
  ∃ (division : Fin g.m → Fin g.m → Fin g.k × Fin g.n), 
    (∀ i j, division i j = division j i → i = j) ∧ 
    (∀ x, (Finset.univ.filter (λ i => (division i i).1 = x)).card = g.n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_figure_symmetry_l586_58680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_theorem_l586_58602

noncomputable def vector_sum_magnitude (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2)

theorem vector_sum_magnitude_theorem (a b : ℝ × ℝ) 
  (h1 : Real.sqrt (a.1^2 + a.2^2) = 1)
  (h2 : Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 2)
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  vector_sum_magnitude a b = Real.sqrt 3 := by
  sorry

#check vector_sum_magnitude_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_theorem_l586_58602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_alpha_gamma_sum_min_alpha_gamma_sum_achievable_l586_58665

/-- Complex function f(z) = (3 + 2i)z^2 + αz + γ -/
def f (α γ : ℂ) (z : ℂ) : ℂ := (3 + 2*Complex.I)*z^2 + α*z + γ

/-- Theorem stating the minimum value of |α| + |γ| given the conditions -/
theorem min_alpha_gamma_sum (α γ : ℂ) : 
  (f α γ 1).im = 0 → (f α γ Complex.I).im = 0 → Complex.abs α + Complex.abs γ ≥ 2 := by
  sorry

/-- Theorem stating that the minimum value of 2 is achievable -/
theorem min_alpha_gamma_sum_achievable : 
  ∃ α γ : ℂ, (f α γ 1).im = 0 ∧ (f α γ Complex.I).im = 0 ∧ Complex.abs α + Complex.abs γ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_alpha_gamma_sum_min_alpha_gamma_sum_achievable_l586_58665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_valid_grid_l586_58672

/-- Represents a 3x3 grid filled with numbers from 2 to 10 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if a given number is even -/
def isEven (n : Fin 9) : Bool := (n.val + 2) % 2 == 0

/-- Checks if the sum of numbers in a row is even -/
def isRowSumEven (g : Grid) (row : Fin 3) : Bool :=
  isEven (g row 0 + g row 1 + g row 2)

/-- Checks if the sum of numbers in a column is even -/
def isColumnSumEven (g : Grid) (col : Fin 3) : Bool :=
  isEven (g 0 col + g 1 col + g 2 col)

/-- Checks if all rows and columns have even sums -/
def isValidGrid (g : Grid) : Bool :=
  (∀ row, isRowSumEven g row) ∧ (∀ col, isColumnSumEven g col)

/-- The total number of possible 3x3 grids using numbers 2 to 10 -/
def totalGrids : ℕ := 362880 -- 9!

/-- The number of valid grids where all rows and columns have even sums -/
def validGrids : ℕ := totalGrids / 14

theorem probability_of_valid_grid :
  (validGrids : ℚ) / totalGrids = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_valid_grid_l586_58672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_to_twelve_l586_58694

theorem cube_root_eight_to_twelve : (8 : ℝ) ^ (1/3 : ℝ) ^ 12 = 4096 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_to_twelve_l586_58694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_calculation_l586_58676

/-- The volume of a right circular cone -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The volume of a hemisphere -/
noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

/-- The total volume of water in the cone and hemisphere -/
noncomputable def total_water_volume (cone_radius cone_height : ℝ) : ℝ :=
  cone_volume cone_radius cone_height + hemisphere_volume cone_radius

theorem water_volume_calculation :
  let cone_radius : ℝ := 3
  let cone_height : ℝ := 12
  total_water_volume cone_radius cone_height = 54 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_calculation_l586_58676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_isosceles_triangle_l586_58629

-- Define the grid
def Grid := Fin 5 × Fin 5

-- Define points A and B
def A : Grid := ⟨2, 2⟩
def B : Grid := ⟨5, 2⟩

-- Define the distance function
def distance (p q : Grid) : ℕ :=
  Int.natAbs (p.1.val - q.1.val) + Int.natAbs (p.2.val - q.2.val)

-- Define what it means for a triangle to be isosceles
def isIsosceles (a b c : Grid) : Prop :=
  distance a c = distance b c ∨ distance a b = distance a c ∨ distance a b = distance b c

-- Theorem statement
theorem no_isosceles_triangle :
  ∀ c : Grid, c ≠ A → c ≠ B → ¬ isIsosceles A B c := by
  sorry

#check no_isosceles_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_isosceles_triangle_l586_58629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_distinct_roots_l586_58653

/-- The quadratic equation kx^2 - 3x - 9/4 = 0 -/
def quadratic_equation (k : ℤ) (x : ℝ) : Prop :=
  k * x^2 - 3 * x - 9/4 = 0

/-- The discriminant of the quadratic equation -/
noncomputable def discriminant (k : ℤ) : ℝ :=
  (-3)^2 - 4 * k * (-9/4)

/-- The condition for two distinct real roots -/
def has_two_distinct_roots (k : ℤ) : Prop :=
  discriminant k > 0 ∧ k ≠ 0

/-- The smallest integer k for which the equation has two distinct real roots is 1 -/
theorem smallest_k_for_distinct_roots :
  ∀ k : ℤ, has_two_distinct_roots k → k ≥ 1 :=
by
  intro k h
  sorry

#check smallest_k_for_distinct_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_distinct_roots_l586_58653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l586_58624

/-- Tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- The specific tetrahedron from the problem -/
noncomputable def specificTetrahedron : Tetrahedron where
  PQ := 6
  PR := 4
  PS := 5
  QR := 5
  QS := 4
  RS := 7/5 * Real.sqrt 11

theorem tetrahedron_volume : 
  volume specificTetrahedron = 7/2 * Real.sqrt 22 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l586_58624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l586_58660

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)

-- Define the point of tangency
def point : ℝ × ℝ := (0, 1)

-- State the theorem
theorem tangent_line_at_point :
  ∃ (m b : ℝ), 
    (∀ x : ℝ, m * x + b = f point.1 + (deriv f point.1) * (x - point.1)) ∧
    m = 2 ∧ 
    b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l586_58660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_shortest_chord_l586_58649

/-- Circle equation: (x)²+(y)²-6x-8y+21=0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 21 = 0

/-- Line equation: kx-y-4k+3=0 -/
def line_eq (k x y : ℝ) : Prop := k*x - y - 4*k + 3 = 0

theorem circle_line_intersection (k : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ 
  circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
  line_eq k x₁ y₁ ∧ line_eq k x₂ y₂ := by sorry

theorem shortest_chord :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
  line_eq 1 x₁ y₁ ∧ line_eq 1 x₂ y₂ ∧
  (∀ (k x₃ y₃ x₄ y₄ : ℝ), 
    circle_eq x₃ y₃ ∧ circle_eq x₄ y₄ ∧ line_eq k x₃ y₃ ∧ line_eq k x₄ y₄ →
    (x₁ - x₂)^2 + (y₁ - y₂)^2 ≤ (x₃ - x₄)^2 + (y₃ - y₄)^2) ∧
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_shortest_chord_l586_58649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l586_58678

-- Define the complex number z
noncomputable def z : ℂ := (5 + Complex.I) / (1 + Complex.I)

-- Theorem statement
theorem z_properties :
  (z.im = -2) ∧ (z.re > 0 ∧ z.im < 0) :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l586_58678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_g_l586_58634

/-- The function g defined as g(x) = x^3 + 3x^(1/2) -/
noncomputable def g (x : ℝ) : ℝ := x^3 + 3 * Real.sqrt x

/-- Theorem stating that 3g(3) - 2g(9) = -1395 + 9√3 -/
theorem evaluate_g : 3 * g 3 - 2 * g 9 = -1395 + 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_g_l586_58634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_difference_inequality_l586_58650

open Set
open Function
open Interval

theorem function_difference_inequality 
  {a b : ℝ} 
  (f g : ℝ → ℝ) 
  (h_diff_f : DifferentiableOn ℝ f (Icc a b))
  (h_diff_g : DifferentiableOn ℝ g (Icc a b))
  (h_deriv : ∀ x ∈ Icc a b, deriv f x > deriv g x)
  (x : ℝ) 
  (h_x : x ∈ Ioo a b) :
  f x + g a > g x + f a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_difference_inequality_l586_58650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_values_l586_58673

-- Define the ellipse equation
noncomputable def ellipse_equation (x y k : ℝ) : Prop := x^2/4 + y^2/k = 1

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 2 / 2

-- Theorem statement
theorem ellipse_k_values (k : ℝ) :
  (∀ x y, ellipse_equation x y k) ∧ 
  (∃ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    ((a^2 = 4 ∧ b^2 = k) ∨ (a^2 = k ∧ b^2 = 4)) ∧
    c^2 = a^2 - b^2 ∧
    c/a = eccentricity) →
  k = 2 ∨ k = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_values_l586_58673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kinetic_energy_ranking_l586_58604

/-- Represents a rotating object with mass and radius -/
structure RotatingObject where
  mass : ℝ
  radius : ℝ

/-- Calculates the moment of inertia of a disk -/
noncomputable def momentOfInertiaDisk (obj : RotatingObject) : ℝ :=
  (1 / 2) * obj.mass * obj.radius^2

/-- Calculates the moment of inertia of a hoop -/
noncomputable def momentOfInertiaHoop (obj : RotatingObject) : ℝ :=
  obj.mass * obj.radius^2

/-- Calculates the moment of inertia of a sphere -/
noncomputable def momentOfInertiaSphere (obj : RotatingObject) : ℝ :=
  (2 / 5) * obj.mass * obj.radius^2

/-- Calculates the kinetic energy of a rotating object -/
noncomputable def kineticEnergy (obj : RotatingObject) (force time : ℝ) (momentOfInertia : ℝ) : ℝ :=
  (1 / 2) * momentOfInertia * ((force * obj.radius * time) / momentOfInertia)^2

theorem kinetic_energy_ranking (M R F t : ℝ) (hM : M > 0) (hR : R > 0) (hF : F > 0) (ht : t > 0) :
  let obj := { mass := M, radius := R : RotatingObject }
  let ke_hoop := kineticEnergy obj F t (momentOfInertiaHoop obj)
  let ke_disk := kineticEnergy obj F t (momentOfInertiaDisk obj)
  let ke_sphere := kineticEnergy obj F t (momentOfInertiaSphere obj)
  ke_hoop < ke_disk ∧ ke_disk < ke_sphere := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kinetic_energy_ranking_l586_58604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_eleven_l586_58619

def a : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 3
  | (n + 3) => (n + 4) * a (n + 2) - (n + 3) * a (n + 1)

theorem divisible_by_eleven (n : ℕ) :
  11 ∣ a n ↔ n = 4 ∨ n = 8 ∨ n = 10 ∨ n ≥ 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_eleven_l586_58619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_wins_by_12_meters_l586_58698

/-- Represents a runner in the race -/
structure Runner where
  speed : ℚ
  start_position : ℚ

/-- Calculates the time taken to cover a given distance at a given speed -/
def time_to_cover (distance : ℚ) (speed : ℚ) : ℚ :=
  distance / speed

/-- Represents the race scenario -/
structure RaceScenario where
  sunny : Runner
  windy : Runner
  race_distance : ℚ

def initial_race : RaceScenario :=
  { sunny := { speed := 5, start_position := 0 },
    windy := { speed := 4, start_position := 0 },
    race_distance := 300 }

def rematch : RaceScenario :=
  { sunny := { speed := initial_race.sunny.speed, start_position := -60 },
    windy := { speed := initial_race.windy.speed, start_position := 10 },
    race_distance := 300 }

/-- Calculates the distance covered by a runner in a given time -/
def distance_covered (runner : Runner) (time : ℚ) : ℚ :=
  runner.speed * time

theorem sunny_wins_by_12_meters :
  let initial_sunny_finish_time := time_to_cover initial_race.race_distance initial_race.sunny.speed
  let initial_windy_distance := distance_covered initial_race.windy initial_sunny_finish_time
  let rematch_sunny_finish_time := time_to_cover (rematch.race_distance - rematch.sunny.start_position) rematch.sunny.speed
  let rematch_windy_distance := distance_covered rematch.windy rematch_sunny_finish_time + rematch.windy.start_position
  rematch.race_distance - rematch_windy_distance = 12 := by
  sorry

#check sunny_wins_by_12_meters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_wins_by_12_meters_l586_58698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_product_l586_58697

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis 4 and semi-minor axis √3 -/
def Ellipse := {p : Point | p.x^2 / 16 + p.y^2 / 3 = 1}

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The maximum value of |PF₁| * |PF₂| is 16 for any point P on the ellipse -/
theorem ellipse_max_product (F1 F2 : Point) (h : F1 ∈ Ellipse ∧ F2 ∈ Ellipse) :
  ∃ (c : ℝ), c = 16 ∧ ∀ (P : Point), P ∈ Ellipse → (distance P F1) * (distance P F2) ≤ c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_product_l586_58697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_profit_l586_58640

noncomputable section

/-- Daily cost function -/
def C (x : ℝ) : ℝ := 3 + x

/-- Daily sales revenue function -/
noncomputable def S (x k : ℝ) : ℝ :=
  if 0 < x ∧ x < 6 then 3*x + k/(x-8) + 5 else 14

/-- Daily profit function -/
noncomputable def L (x k : ℝ) : ℝ := S x k - C x

/-- The value of k satisfying the given condition -/
def k : ℝ := 18

/-- Theorem stating the maximum daily profit -/
theorem max_daily_profit :
  (L 2 k = 3) →
  ∃ (max_profit : ℝ), max_profit = 6 ∧
    ∀ x, L x k ≤ max_profit ∧
    ∃ x₀, x₀ = 5 ∧ L x₀ k = max_profit :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_profit_l586_58640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_li_payment_l586_58652

/-- Calculates the discounted price based on the given rules -/
noncomputable def discountedPrice (price : ℝ) : ℝ :=
  if price ≤ 100 then price
  else if price ≤ 300 then price * 0.9
  else 300 * 0.9 + (price - 300) * 0.8

/-- Xiao Mei's first purchase amount -/
def purchase1 : ℝ := 94.5

/-- Xiao Mei's second purchase amount after discount -/
def purchase2 : ℝ := 282.8

/-- Theorem stating the possible amounts Xiao Li should pay -/
theorem xiao_li_payment :
  ∃ (original_price1 original_price2 : ℝ),
    discountedPrice original_price1 = purchase1 ∧
    discountedPrice original_price2 = purchase2 ∧
    (discountedPrice (original_price1 + original_price2) = 358.4 ∨
     discountedPrice (original_price1 + original_price2) = 366.8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_li_payment_l586_58652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l586_58685

-- Define the ellipse Γ
noncomputable def Γ (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, -1)

-- Define lines ℓ₁ and ℓ₂
def ℓ₁ (x : ℝ) : Prop := x = -2
def ℓ₂ (y : ℝ) : Prop := y = -1

-- Define point P
noncomputable def P (x₀ y₀ : ℝ) : Prop := Γ x₀ y₀ ∧ x₀ > 0 ∧ y₀ > 0

-- Define tangent line ℓ₃
def ℓ₃ (x₀ y₀ x y : ℝ) : Prop := x₀ * x / 4 + y₀ * y = 1

-- Define intersection points C, D, and E
def C : ℝ × ℝ := (-2, -1)
noncomputable def D (x₀ y₀ : ℝ) : ℝ × ℝ := (4 * (y₀ + 1) / x₀, -1)
noncomputable def E (x₀ y₀ : ℝ) : ℝ × ℝ := (-2, (x₀ + 2) / (2 * y₀))

-- Theorem statement
theorem lines_concurrent (x₀ y₀ : ℝ) (h₁ : P x₀ y₀) : 
  ∃ (x y : ℝ), (x - A.1) * (D x₀ y₀).2 = (y - A.2) * (D x₀ y₀).1 ∧ 
                (x - B.1) * (E x₀ y₀).2 = (y - B.2) * (E x₀ y₀).1 ∧ 
                (x - C.1) * y₀ = (y - C.2) * x₀ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l586_58685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f₁_derivative_f₂_l586_58632

open Real

-- Function 1
noncomputable def f₁ (x : ℝ) : ℝ := sin x / (1 + sin x)

-- Function 2
noncomputable def f₂ (x : ℝ) : ℝ := x * tan x

-- Theorem for the derivative of f₁
theorem derivative_f₁ : 
  ∀ x : ℝ, deriv f₁ x = cos x / (1 + sin x)^2 := by sorry

-- Theorem for the derivative of f₂
theorem derivative_f₂ : 
  ∀ x : ℝ, deriv f₂ x = (sin x * cos x + x) / (cos x)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f₁_derivative_f₂_l586_58632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l586_58690

/-- The distance between two parallel lines in 2D space -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₂ - c₁) / Real.sqrt (a^2 + b^2)

theorem distance_between_specific_lines :
  let line1 : ℝ → ℝ → ℝ := λ x y => x + y - 1
  let line2 : ℝ → ℝ → ℝ := λ x y => x + y + 1
  distance_between_parallel_lines 1 1 (-1) 1 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l586_58690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_e_value_l586_58675

-- Define the circle and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define the diameter and its length
noncomputable def diameter (c : Circle) : ℝ := 2 * c.radius

noncomputable def PQ (c : Circle) : ℝ := 4/5 * diameter c

-- Define the points on the circle
noncomputable def P (c : Circle) : Point := ⟨c.center.1 - c.radius, c.center.2⟩
noncomputable def Q (c : Circle) : Point := ⟨c.center.1 + c.radius, c.center.2⟩
noncomputable def X (c : Circle) : Point := sorry
noncomputable def Y (c : Circle) : Point := sorry
noncomputable def Z (c : Circle) : Point := sorry

-- Define the property that X is the midpoint of the semicircle
def X_is_midpoint (c : Circle) : Prop := 
  X c = ⟨c.center.1, c.center.2 + c.radius⟩

-- Define the line segment e
noncomputable def e (c : Circle) : ℝ := sorry

-- State the theorem
theorem max_e_value (c : Circle) : 
  ∃ (e_max : ℝ), e_max = 8 - 6 * Real.sqrt 2 ∧ ∀ (e_val : ℝ), e c ≤ e_max :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_e_value_l586_58675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_a_nonpositive_monotonicity_a_positive_range_of_a_l586_58622

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x

-- State the theorems
theorem monotonicity_a_nonpositive (a : ℝ) (h : a ≤ 0) :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f a x₁ < f a x₂ := by sorry

theorem monotonicity_a_positive (a : ℝ) (h : a > 0) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → x₂ < a → f a x₁ > f a x₂) ∧
  (∀ x₁ x₂ : ℝ, a < x₁ → a < x₂ → x₁ < x₂ → f a x₁ < f a x₂) := by sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 0 < x → x ≤ Real.exp 1 → f a x ≥ 1) :
  a ≥ 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_a_nonpositive_monotonicity_a_positive_range_of_a_l586_58622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_asymptotes_l586_58668

/-- The value of M for which the hyperbolas (x²/9) - (y²/16) = 1 and (y²/25) - (x²/M) = 1 have the same asymptotes -/
def M : ℚ := 225 / 16

/-- The first hyperbola equation -/
def hyperbola1 (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

/-- The second hyperbola equation -/
def hyperbola2 (x y : ℝ) : Prop := y^2 / 25 - x^2 / M = 1

/-- The slope of the asymptotes for the first hyperbola -/
noncomputable def slope1 : ℝ := 4 / 3

/-- The slope of the asymptotes for the second hyperbola -/
noncomputable def slope2 : ℝ := 5 / Real.sqrt M

/-- Theorem stating that M is the correct value for which the hyperbolas have the same asymptotes -/
theorem hyperbolas_same_asymptotes : slope1 = slope2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_asymptotes_l586_58668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_circle_intersection_l586_58683

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^2 - 6*x + 5

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = 13

-- Define the line that intersects with circle C
def intersecting_line (x y : ℝ) (a : ℝ) : Prop := x - y + a = 0

-- Theorem statement
theorem curve_circle_intersection (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ x₃ y₃, 
    curve x₁ y₁ ∧ (x₁ = 0 ∨ y₁ = 0) ∧
    curve x₂ y₂ ∧ (x₂ = 0 ∨ y₂ = 0) ∧
    curve x₃ y₃ ∧ (x₃ = 0 ∨ y₃ = 0) ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ circle_C x₃ y₃) →
  (∃ x_A y_A x_B y_B,
    circle_C x_A y_A ∧ circle_C x_B y_B ∧
    intersecting_line x_A y_A a ∧ intersecting_line x_B y_B a ∧
    ((x_A - 3) * (x_B - 3) + (y_A - 3) * (y_B - 3) = 0)) →
  a = Real.sqrt 13 ∨ a = -Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_circle_intersection_l586_58683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camerons_list_count_l586_58693

theorem camerons_list_count : 
  let start := 900
  let stop := 27000
  let step := 30
  (List.range ((stop - start) / step + 1)).length = 871 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camerons_list_count_l586_58693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_functions_sum_l586_58670

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (9^x - a) / 3^x

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := Real.log (10^x + 1) + b * x

theorem symmetric_functions_sum (a b : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (∀ x, g b x = g b (-x)) →
  a + b = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_functions_sum_l586_58670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_parallelogram_l586_58658

/-- A quadrilateral with sides a, b, c, d is a parallelogram if a^2 + b^2 + c^2 + d^2 - 2ac - 2bd = 0 -/
theorem quadrilateral_is_parallelogram (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 - 2*a*c - 2*b*d = 0) : 
  a = c ∧ b = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_parallelogram_l586_58658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_concurrency_l586_58612

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- The intersection point of two line segments -/
def intersectionPoint (A B C D : Point) : Point := sorry

/-- The length of a line segment between two points -/
def segmentLength (A B : Point) : ℝ := sorry

/-- The angle bisector of an angle formed by three points -/
def angleBisector (A B C : Point) : Set Point := sorry

/-- Checks if three sets of points have a common intersection -/
def areConcurrent (S₁ S₂ S₃ : Set Point) : Prop := sorry

theorem angle_bisector_concurrency 
  (q : Quadrilateral) 
  (h₁ : isConvex q) 
  (P : Point) 
  (h₂ : P = intersectionPoint q.A q.C q.B q.D) 
  (h₃ : segmentLength q.A q.C + segmentLength q.A q.D = 
        segmentLength q.B q.C + segmentLength q.B q.D) : 
  areConcurrent 
    (angleBisector q.A q.C q.B) 
    (angleBisector q.A q.D q.B) 
    (angleBisector q.A P q.B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_concurrency_l586_58612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sorting_problem_l586_58614

/-- The time it takes one person to complete the task -/
noncomputable def one_person_time : ℝ := 60

/-- The initial work time -/
noncomputable def initial_work_time : ℝ := 1

/-- The additional work time after more people joined -/
noncomputable def additional_work_time : ℝ := 2

/-- The number of additional people who joined -/
def additional_people : ℕ := 15

/-- The rate at which one person completes the task per hour -/
noncomputable def work_rate : ℝ := 1 / one_person_time

theorem book_sorting_problem (x : ℕ) :
  (x : ℝ) * work_rate * initial_work_time + 
  ((x : ℝ) + additional_people) * work_rate * additional_work_time = 1 →
  x = 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sorting_problem_l586_58614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l586_58689

theorem trigonometric_identity (θ : ℝ) (h1 : Real.sin θ ≠ 0) (h2 : Real.cos θ ≠ 0) :
  (Real.sin θ + (1 / Real.sin θ) + 1)^2 + (Real.cos θ + (1 / Real.cos θ) + 1)^2 =
  9 + 2 * (Real.sin θ + Real.cos θ) * (1 + 1 / (Real.sin θ * Real.cos θ)) + (Real.sin θ / Real.cos θ)^2 + (Real.cos θ / Real.sin θ)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l586_58689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_1050_l586_58681

theorem prime_divisors_of_1050 :
  (Finset.filter (fun p => Nat.Prime p ∧ p ∣ 1050) (Finset.range 1051)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_1050_l586_58681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalence_l586_58621

theorem proposition_equivalence (a : ℝ) : 
  ((∀ x : ℝ, x^2 + x + a > 0) ∧ ¬(∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0)) ∨
  ((∃ x : ℝ, x^2 + x + a ≤ 0) ∧ (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0)) ↔ 
  (a ≤ -1 ∨ (1/4 < a ∧ a < 1)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_equivalence_l586_58621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_squared_lt_9_necessary_not_sufficient_l586_58699

theorem a_squared_lt_9_necessary_not_sufficient :
  (∀ a : ℝ, |a - 1| < 2 → a^2 < 9) ∧
  (∃ a : ℝ, a^2 < 9 ∧ |a - 1| ≥ 2) :=
by
  constructor
  · intro a h
    sorry
  · use (-2 : ℝ)
    constructor
    · norm_num
    · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_squared_lt_9_necessary_not_sufficient_l586_58699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_investment_is_400_l586_58630

-- Define the constants
noncomputable def mary_investment : ℝ := 600
noncomputable def total_profit : ℝ := 7500
noncomputable def mary_extra : ℝ := 1000

-- Define Mike's investment as a variable
variable (mike_investment : ℝ)

-- Define the profit shares
noncomputable def effort_share : ℝ := total_profit / 3 / 2
noncomputable def investment_share : ℝ := 2 * total_profit / 3

-- Define Mary's and Mike's total shares
noncomputable def mary_share (mike_investment : ℝ) : ℝ := 
  effort_share + (mary_investment / (mary_investment + mike_investment)) * investment_share

noncomputable def mike_share (mike_investment : ℝ) : ℝ := 
  effort_share + (mike_investment / (mary_investment + mike_investment)) * investment_share

-- State the theorem
theorem mike_investment_is_400 :
  mary_share mike_investment = mike_share mike_investment + mary_extra → mike_investment = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_investment_is_400_l586_58630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_phi_for_even_g_l586_58662

noncomputable section

/-- The original function f(x) = cos(2x + π/6) -/
def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

/-- The translated function g(x) = f(x + φ) -/
def g (φ : ℝ) (x : ℝ) : ℝ := f (x + φ)

/-- A function is even if f(-x) = f(x) for all x -/
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

/-- The smallest positive φ that makes g an even function is 5π/12 -/
theorem smallest_phi_for_even_g :
  (∃ φ > 0, is_even (g φ)) ∧ 
  (∀ ψ > 0, is_even (g ψ) → ψ ≥ 5 * Real.pi / 12) ∧
  is_even (g (5 * Real.pi / 12)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_phi_for_even_g_l586_58662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mean_weight_B_and_C_l586_58654

/-- Represents a pile of rocks -/
structure RockPile where
  totalWeight : ℝ
  count : ℝ

/-- Calculates the mean weight of a rock pile -/
noncomputable def meanWeight (pile : RockPile) : ℝ := pile.totalWeight / pile.count

/-- Combines two rock piles -/
def combinePiles (pile1 pile2 : RockPile) : RockPile :=
  { totalWeight := pile1.totalWeight + pile2.totalWeight
  , count := pile1.count + pile2.count }

/-- The main theorem statement -/
theorem max_mean_weight_B_and_C (A B C : RockPile) 
  (hA : meanWeight A = 40)
  (hB : meanWeight B = 50)
  (hAB : meanWeight (combinePiles A B) = 43)
  (hAC : meanWeight (combinePiles A C) = 44) :
  ∃ (n : ℕ), n = 59 ∧ 
    (∀ (m : ℕ), meanWeight (combinePiles B C) ≤ m → m ≤ n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mean_weight_B_and_C_l586_58654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_cube_relationship_l586_58651

theorem ln_cube_relationship (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, Real.log x > Real.log y → x^3 > y^3) ∧
  (∃ x y, x^3 > y^3 ∧ ¬(Real.log x > Real.log y)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_cube_relationship_l586_58651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_amount_l586_58692

/-- Given a total amount of 8000 among three people (P, Q, and R), where R has two-thirds of the
    total amount with P and Q, prove that R's amount is 3200. -/
theorem r_amount (total : ℕ) (r_fraction : ℚ) (r_amount : ℕ) : 
  total = 8000 →
  r_fraction = 2/3 →
  r_amount = total - Int.floor (total * (1 - r_fraction)) →
  r_amount = 3200 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_amount_l586_58692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_polynomials_are_correct_l586_58674

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  b : ℤ
  c : ℤ

/-- The set of roots of a quadratic polynomial -/
def roots (p : QuadraticPolynomial) : Set ℤ :=
  {x : ℤ | x^2 + p.b * x + p.c = 0}

/-- The condition that the sum of coefficients is 10 -/
def sum_of_coefficients_is_10 (p : QuadraticPolynomial) : Prop :=
  1 + p.b + p.c = 10

/-- The set of all valid quadratic polynomials according to the problem conditions -/
def valid_polynomials : Set QuadraticPolynomial :=
  {p : QuadraticPolynomial | (∃ x, x ∈ roots p) ∧ sum_of_coefficients_is_10 p}

/-- The set of correct quadratic polynomials from the solution -/
def correct_polynomials : Set QuadraticPolynomial :=
  { ⟨-13, 22⟩, ⟨-9, 18⟩, ⟨9, 0⟩, ⟨5, 4⟩ }

theorem valid_polynomials_are_correct :
  valid_polynomials = correct_polynomials := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_polynomials_are_correct_l586_58674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l586_58633

theorem matrix_scalar_multiplication (u : Fin 3 → ℝ) :
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![3, 0, 0; 0, 3, 0; 0, 0, 3]
  N.mulVec u = (3 : ℝ) • u :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l586_58633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_six_l586_58609

/-- The speed of a car moving at a constant rate -/
noncomputable def car_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: The speed of a car moving at a constant rate is 6 m/s, given it covers 48 m in 8 s -/
theorem car_speed_is_six :
  car_speed 48 8 = 6 := by
  -- Unfold the definition of car_speed
  unfold car_speed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_six_l586_58609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_log_4_9_l586_58615

-- Define an odd function f
noncomputable def f (x : ℝ) : ℝ := sorry

-- State the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_neg : ∀ x, x < 0 → f x = Real.exp (x * Real.log 2)

-- Theorem to prove
theorem f_log_4_9 : f (Real.log 9 / Real.log 4) = -1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_log_4_9_l586_58615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_second_quadrant_l586_58625

theorem cos_minus_sin_second_quadrant (α : ℝ) (h1 : Real.cos α * Real.sin α = -1/8) 
  (h2 : π/2 < α ∧ α < π) : Real.cos α - Real.sin α = -Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_second_quadrant_l586_58625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_for_euler_totient_l586_58603

theorem infinitely_many_n_for_euler_totient (a b : ℕ) (ha : a > 1) (hb : b > 1) :
  Set.Infinite {n : ℕ | ∀ m t : ℕ, Nat.totient (a^n - 1) ≠ b^m - b^t} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_for_euler_totient_l586_58603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l586_58687

noncomputable section

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 -/
def distancePointToLine (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- A circle with equation x² + y² = 4 -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- A line with equation y = x + b -/
def lineEq (x y b : ℝ) : Prop := y = x + b

theorem circle_line_intersection (b : ℝ) :
  (∃! (p₁ p₂ p₃ : ℝ × ℝ), 
    circleEq p₁.1 p₁.2 ∧ circleEq p₂.1 p₂.2 ∧ circleEq p₃.1 p₃.2 ∧
    distancePointToLine p₁.1 p₁.2 (-1) 1 (-b) = 1 ∧
    distancePointToLine p₂.1 p₂.2 (-1) 1 (-b) = 1 ∧
    distancePointToLine p₃.1 p₃.2 (-1) 1 (-b) = 1) →
  b = Real.sqrt 2 ∨ b = -Real.sqrt 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l586_58687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_partition_l586_58631

/-- A cell in the grid --/
structure Cell where
  row : Fin 8
  col : Fin 8

/-- A part of the grid --/
structure GridPart where
  cells : List Cell
  connected : Bool
  size_le_16 : cells.length ≤ 16

/-- A partition of the grid --/
structure Partition where
  parts : List GridPart
  all_cells_covered : ∀ c : Cell, ∃ p ∈ parts, c ∈ p.cells
  no_overlap : ∀ c : Cell, ∀ p1 p2 : GridPart, p1 ∈ parts → p2 ∈ parts → c ∈ p1.cells → c ∈ p2.cells → p1 = p2
  grid_connected : Bool

/-- Theorem stating the existence of a valid partition --/
theorem exists_valid_partition :
  ∃ p : Partition, p.parts.length ≥ 27 ∧ 
  (∀ part : GridPart, part ∈ p.parts → part.connected) ∧ 
  p.grid_connected := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_partition_l586_58631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l586_58628

/-- Calculates the speed of a train in km/h given its length and time to pass a point -/
noncomputable def trainSpeed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Proves that a 250-meter long train passing a point in 12 seconds has a speed of 75 km/h -/
theorem train_speed_proof :
  let train_length : ℝ := 250
  let passing_time : ℝ := 12
  trainSpeed train_length passing_time = 75 := by
  unfold trainSpeed
  simp
  norm_num
  -- The proof is completed automatically by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l586_58628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_l586_58638

theorem sector_angle (circumference area : ℝ) (h1 : circumference = 20) (h2 : area = 9) :
  ∃ R l α : ℝ, R > 0 ∧ l > 0 ∧ α > 0 ∧
  2 * R + l = circumference ∧
  1 / 2 * l * R = area ∧
  α = l / R ∧
  α = 2 / 9 := by
  sorry

#check sector_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_l586_58638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_a_positive_l586_58679

noncomputable def f (x a : ℝ) := 2^(x - 1) - a

theorem root_implies_a_positive (a : ℝ) :
  (∃ x, f x a = 0) → a ∈ Set.Ioi 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_a_positive_l586_58679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partitions_into_two_parts_l586_58639

theorem partitions_into_two_parts (N : ℕ) : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 + p.2 = N ∧ p.1 ≤ p.2) (Finset.product (Finset.range (N + 1)) (Finset.range (N + 1)))).card = 
  (N / 2 : ℚ).floor + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partitions_into_two_parts_l586_58639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l586_58635

def a : ℕ → ℚ
  | 0 => 0  -- Add a case for 0 to cover all natural numbers
  | 1 => 3/2
  | 2 => 1
  | 3 => 7/10
  | 4 => 9/17
  | n + 1 => (2 * (n + 1) + 1) / ((n + 1)^2 + 1)

theorem sequence_formula (n : ℕ) : n > 0 → a n = (2 * n + 1) / (n^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l586_58635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_santa_prob_101_l586_58666

/-- The number of people participating in the Secret Santa gift exchange. -/
def n : ℕ := 101

/-- The derangement function, which gives the number of permutations of a set
    where no element appears in its original position. -/
noncomputable def D (k : ℕ) : ℝ := k.factorial / Real.exp 1

/-- The probability that the first person neither gives gifts to nor receives gifts
    from the second or third person in a Secret Santa gift exchange with n people. -/
noncomputable def santa_prob (n : ℕ) : ℝ :=
  1 - (4 * D (n - 1) - 2 * D (n - 3)) / D n

/-- The theorem stating the probability for the specific case of 101 people. -/
theorem santa_prob_101 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ |santa_prob n - 0.96039| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_santa_prob_101_l586_58666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sequence_theorem_l586_58645

def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 5*x = 0

def point_on_circle (x y : ℝ) : Prop := circle_equation x y ∧ x = 5/2 ∧ y = 3/2

noncomputable def chord_length (x y : ℝ) : ℝ := sorry

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d

def possible_k (k : ℕ) : Prop := k = 3 ∨ k = 4 ∨ k = 5 ∨ k = 6

theorem chord_sequence_theorem (k : ℕ) (d : ℝ) :
  (∃ x y : ℝ, point_on_circle x y) →
  k ≥ 3 →
  (∃ a : ℕ → ℝ, arithmetic_sequence a d ∧
    (∀ n : ℕ, n < k → ∃ x y : ℝ, point_on_circle x y ∧ chord_length x y = a n)) →
  1/6 < d ∧ d ≤ 1/3 →
  possible_k k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sequence_theorem_l586_58645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_points_on_circle_at_distance_from_line_l586_58613

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the line
def lineEq (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the distance from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - y + 1| / Real.sqrt 2

-- Theorem statement
theorem exactly_three_points_on_circle_at_distance_from_line :
  ∃! (points : Finset (ℝ × ℝ)), 
    points.card = 3 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ points → circleEq x y ∧ distance_to_line x y = Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_points_on_circle_at_distance_from_line_l586_58613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l586_58677

theorem trigonometric_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α = 4 / 5) :
  (Real.tan α = 4 / 3) ∧ (Real.cos (2 * α) + Real.sin (α + π / 2) = 8 / 25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l586_58677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_statement_l586_58607

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, Real.sin x > Real.sqrt 3 / 2) ↔ (∃ x : ℝ, Real.sin x ≤ Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_statement_l586_58607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l586_58601

/-- The time it takes for A and B to complete the work together -/
noncomputable def time_together : ℝ := 10

/-- The time it takes for A to complete the work alone -/
noncomputable def time_a_alone : ℝ := 14

/-- The rate at which A and B work together -/
noncomputable def rate_together : ℝ := 1 / time_together

/-- The rate at which A works alone -/
noncomputable def rate_a_alone : ℝ := 1 / time_a_alone

/-- The rate at which B works alone -/
noncomputable def rate_b_alone : ℝ := rate_together - rate_a_alone

theorem work_completion_time :
  rate_together = rate_a_alone + rate_b_alone := by
  -- Expand definitions
  unfold rate_together rate_a_alone rate_b_alone
  -- Simplify the equation
  simp [time_together, time_a_alone]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l586_58601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_need_change_probability_l586_58610

/-- Represents the cost of a toy in quarters -/
def ToyCost := Fin 10 → Nat

/-- The number of toys in the machine -/
def numToys : Nat := 10

/-- Lisa's initial number of quarters -/
def initialQuarters : Nat := 16

/-- Cost of Lisa's favorite toy in quarters -/
def favoriteToyQuarters : Nat := 11

/-- Represents a valid toy cost distribution -/
def validToyCosts (c : ToyCost) : Prop :=
  ∀ i : Fin 10, 2 ≤ c i ∧ c i ≤ 12 ∧
  ∀ j : Fin 10, i.val < j.val → c i = c j + 1

/-- The probability of needing change -/
def probabilityNeedChange (c : ToyCost) : ℚ :=
  1 - (9 * Nat.factorial 8 + 9 * Nat.factorial 8) / Nat.factorial 10

theorem need_change_probability (c : ToyCost) 
  (h_valid : validToyCosts c) 
  (h_favorite : c 1 = favoriteToyQuarters) : 
  probabilityNeedChange c = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_need_change_probability_l586_58610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eddys_climbing_rate_l586_58606

/-- Proves that Eddy's climbing rate is 500 ft/hr given the conditions of the problem -/
theorem eddys_climbing_rate : 
  ∀ (eddy_rate : ℝ),
  (let summit_distance : ℝ := 4700
   let hillary_climb_rate : ℝ := 800
   let hillary_stop_distance : ℝ := 700
   let hillary_descent_rate : ℝ := 1000
   let total_time : ℝ := 6  -- 06:00 to 12:00 is 6 hours

   -- Hillary's climb distance
   let hillary_climb_distance : ℝ := summit_distance - hillary_stop_distance

   -- Time Hillary spends climbing
   let hillary_climb_time : ℝ := hillary_climb_distance / hillary_climb_rate

   -- Distance Eddy climbs (equal to Hillary's descent distance)
   let eddy_climb_distance : ℝ := hillary_descent_rate * (total_time - hillary_climb_time)

   eddy_rate = eddy_climb_distance / total_time) → eddy_rate = 500 := by
  intro eddy_rate
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eddys_climbing_rate_l586_58606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l586_58664

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4*a*x else (2*a+3)*x - 4*a + 5

theorem f_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (1/2 ≤ a ∧ a ≤ 3/2) := by
  sorry

#check f_increasing_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l586_58664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_even_l586_58682

/-- A function f(x) = ax^3 - 1/x + b, where a > 0 and b is an integer -/
noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - 1/x + b

theorem f_sum_even (a b : ℝ) (ha : a > 0) (hb : ∃ n : ℤ, b = ↑n) :
  ∃ k : ℤ, f a b (Real.log a) + f a b (Real.log (1/a)) = 2 * ↑k := by
  sorry

#check f_sum_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_even_l586_58682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_reducible_l586_58605

/-- The operation that replaces two adjacent digits with their sum according to the rule -/
def digitOperation (a b : ℕ) : ℕ := a + 2 * b

/-- Predicate that checks if a number can be reduced to 1 using the digit operation -/
def canReduceToOne (n : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (sequence : Fin (k + 1) → ℕ),
    sequence ⟨0, Nat.zero_lt_succ k⟩ = n ∧
    sequence ⟨k, Nat.lt_succ_self k⟩ = 1 ∧
    ∀ (i : Fin k), 
      ∃ (a b : ℕ),
        sequence ⟨i.val, Nat.lt_succ_of_lt i.isLt⟩ = a * 10 + b ∧
        sequence ⟨i.val + 1, Nat.succ_lt_succ i.isLt⟩ = digitOperation a b

theorem all_numbers_reducible :
  ∀ n : ℕ, n > 0 → canReduceToOne n := by
  sorry

#check all_numbers_reducible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_reducible_l586_58605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_y_coordinate_difference_l586_58655

/-- The maximum difference between the y-coordinates of the intersection points of
    y = 3 - 2x^2 + x^3 and y = 1 + x^2 + x^3 is 4√6/9 -/
theorem intersection_y_coordinate_difference : 
  let f (x : ℝ) := 3 - 2*x^2 + x^3
  let g (x : ℝ) := 1 + x^2 + x^3
  let intersection_points := {x : ℝ | f x = g x}
  let y_coordinates := {y : ℝ | ∃ x ∈ intersection_points, y = f x}
  ∃ (y₁ y₂ : ℝ), y₁ ∈ y_coordinates ∧ y₂ ∈ y_coordinates ∧ 
    |y₁ - y₂| = (4 * Real.sqrt 6) / 9 ∧ 
    ∀ (y₃ y₄ : ℝ), y₃ ∈ y_coordinates → y₄ ∈ y_coordinates → 
      |y₃ - y₄| ≤ (4 * Real.sqrt 6) / 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_y_coordinate_difference_l586_58655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_inequality_l586_58617

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  sum_angles : angleA + angleB + angleC = Real.pi
  obtuse : Real.pi / 2 < max angleA (max angleB angleC)

-- State the theorem
theorem obtuse_triangle_inequality (t : Triangle) :
  t.a^3 * Real.cos t.angleA + t.b^3 * Real.cos t.angleB + t.c^3 * Real.cos t.angleC < t.a * t.b * t.c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_inequality_l586_58617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_specific_truncated_cone_l586_58648

/-- The volume of a truncated right circular cone -/
noncomputable def truncated_cone_volume (r₁ r₂ h : ℝ) : ℝ :=
  (1/3) * Real.pi * h * (r₁^2 + r₂^2 + r₁*r₂)

/-- Theorem: Volume of the specific truncated cone -/
theorem volume_specific_truncated_cone :
  truncated_cone_volume 10 3 9 = (8757/21) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_specific_truncated_cone_l586_58648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_investment_share_l586_58671

/-- Partnership investment problem -/
theorem partnership_investment_share (initial_investment : ℝ) (annual_gain : ℝ) :
  annual_gain = 18900 →
  (initial_investment * 12) / (initial_investment * 12 + 2 * initial_investment * 6 + 3 * initial_investment * 4) * annual_gain = 6300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_investment_share_l586_58671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l586_58663

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 10 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + y + 6 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x + y + 6| / Real.sqrt 2

-- Theorem statement
theorem max_distance_circle_to_line :
  ∃ (x y : ℝ), circle_eq x y ∧ 
  (∀ (x' y' : ℝ), circle_eq x' y' → 
  distance_to_line x y ≥ distance_to_line x' y') ∧
  distance_to_line x y = 8 * Real.sqrt 2 := by
  sorry

#check max_distance_circle_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l586_58663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_irrational_l586_58696

theorem sqrt_5_irrational :
  Irrational (Real.sqrt 5) ∧
  ¬ Irrational (7 / 4) ∧
  ¬ Irrational (0.3) ∧
  ¬ Irrational ((8 : ℝ) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_irrational_l586_58696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_l586_58684

/-- Four points lie on a single straight line if and only if e = 1/4 -/
theorem collinear_points (a b c e : ℝ) : 
  (∃ (t : ℝ → ℝ × ℝ × ℝ), 
    t 0 = (1, 0, a) ∧ 
    t 1 = (b, 1, 0) ∧ 
    ∃ u v, t u = (0, c, 1) ∧ t v = (8*e, 8*e, -e)) ↔ 
  e = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_l586_58684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erik_pie_amount_l586_58659

theorem erik_pie_amount (frank_pie erik_extra : Real) : 
  frank_pie = 0.33 →
  erik_extra = 0.34 →
  frank_pie + erik_extra = 0.67 := by
    intro h1 h2
    rw [h1, h2]
    norm_num

#check erik_pie_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erik_pie_amount_l586_58659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_g_symmetry_at_pi_8_l586_58643

open Real

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := cos ((2/3) * x + Real.pi/2)
noncomputable def g (x : ℝ) : ℝ := sin (2 * x + 5*Real.pi/4)

-- State the theorems
theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

theorem g_symmetry_at_pi_8 : ∀ x, g (Real.pi/4 - x) = g (Real.pi/4 + x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_g_symmetry_at_pi_8_l586_58643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l586_58688

noncomputable def b_sequence (k m l n : ℤ) : ℤ := 
  k * Int.floor (Real.sqrt (n + m : ℝ)) + l

theorem sequence_sum (k m l : ℤ) :
  (∀ n : ℤ, n > 0 → ∃ t : ℤ, b_sequence k m l n = 2 * t) →
  b_sequence k m l 1 = 2 →
  k + m + l = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l586_58688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_shortest_path_l586_58686

/-- The length of the shortest closed path on a cone's surface -/
noncomputable def shortest_path_length (r : ℝ) (l : ℝ) : ℝ :=
  2 * l * Real.sin (Real.pi / 3)

/-- Theorem: For a cone with base radius 2/3 and slant height 2,
    the shortest closed path length is 2√3 -/
theorem cone_shortest_path :
  let r : ℝ := 2/3
  let l : ℝ := 2
  shortest_path_length r l = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_shortest_path_l586_58686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_and_triangle_side_l586_58657

noncomputable def m (x : ℝ) : ℝ × ℝ := (1/2 * Real.sin x, Real.sqrt 3/2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x^2 - 1/2)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem axis_of_symmetry_and_triangle_side (k : ℤ) (A B : ℝ) (a b : ℝ) :
  f A = 0 → Real.sin B = 4/5 → a = Real.sqrt 3 →
  (∃ k : ℤ, x = k * Real.pi / 2 + Real.pi / 12 ↔ ∀ y, f (x + y) = f (x - y)) ∧
  b = 8/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_and_triangle_side_l586_58657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_vectors_l586_58691

theorem min_sum_of_vectors (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (1 * (m + n) + 2 * n * m = 1) → 
  (m + n ≥ Real.sqrt 3 - 1 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ m₀ + n₀ = Real.sqrt 3 - 1) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_vectors_l586_58691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_half_l586_58656

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * Real.arcsin x)

theorem f_one_half : f (1/2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_half_l586_58656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l586_58644

-- Define the line l
def line_l (x y : ℝ) : Prop := x + Real.sqrt 3 * y = 5 * Real.sqrt 3

-- Define the circle C in polar coordinates
def circle_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Define the ray OP
def ray_OP (ρ θ : ℝ) : Prop := θ = Real.pi / 6 ∧ ρ ≥ 0

-- Define point A as the intersection of circle C and ray OP
def point_A (ρ_A θ_A : ℝ) : Prop := 
  circle_C ρ_A θ_A ∧ ray_OP ρ_A θ_A

-- Define point B as the intersection of line l and ray OP
def point_B (ρ_B θ_B : ℝ) : Prop := 
  line_l (ρ_B * Real.cos θ_B) (ρ_B * Real.sin θ_B) ∧ ray_OP ρ_B θ_B

theorem length_AB : 
  ∀ (ρ_A θ_A ρ_B θ_B : ℝ), 
    point_A ρ_A θ_A → point_B ρ_B θ_B → 
    |ρ_A - ρ_B| = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l586_58644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_equals_nine_l586_58641

theorem power_product_equals_nine (a b : ℝ) (h : a + 3 * b - 2 = 0) :
  (3 : ℝ)^a * (27 : ℝ)^b = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_equals_nine_l586_58641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_property_l586_58608

/-- An arithmetic progression -/
structure ArithmeticProgression where
  a : ℕ → ℚ
  d : ℚ
  progression : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  n / 2 * (2 * ap.a 1 + (n - 1) * ap.d)

/-- Main theorem -/
theorem arithmetic_progression_property (ap : ArithmeticProgression) :
  sum_n ap 6 < sum_n ap 7 ∧ sum_n ap 7 > sum_n ap 8 →
  ∀ n ≥ 8, ap.a n < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_property_l586_58608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l586_58623

noncomputable def f (x : ℝ) := x^2
noncomputable def g (x m : ℝ) := (1/2)^x - m

theorem min_m_value (m : ℝ) :
  (∀ x₁ ∈ Set.Icc (-1) 3, ∃ x₂ ∈ Set.Icc 0 2, f x₁ ≥ g x₂ m) ↔ m ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l586_58623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_sequence_form_arithmetic_sequence_general_term_b_geometric_sequence_l586_58626

/-- An arithmetic sequence with a_1 = 1 and S_3 = 0 -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = 1
  sum_three : (a 1) + (a 2) + (a 3) = 0

/-- The B_n sequence derived from the arithmetic sequence -/
noncomputable def B (seq : ArithmeticSequence) : ℕ → ℚ :=
  λ n => 2 * (1 - (-2)^n) / 3

/-- Theorem stating the form of B_n -/
theorem b_sequence_form (seq : ArithmeticSequence) :
  ∀ n : ℕ, B seq n = 2 * (1 - (-2)^n) / 3 := by
  intro n
  rfl

/-- Lemma: The common difference of the arithmetic sequence is -1 -/
lemma common_difference (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a (n + 1) - seq.a n = -1 := by
  sorry

/-- Theorem: The general term of the arithmetic sequence -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 - n := by
  sorry

/-- Lemma: b₁ = 2a₁ = 2 and b₂ = a₆ = -4 -/
lemma b_terms (seq : ArithmeticSequence) :
  B seq 1 = 2 ∧ B seq 2 = -4 := by
  sorry

/-- Theorem: The B_n sequence forms a geometric sequence with common ratio -2 -/
theorem b_geometric_sequence (seq : ArithmeticSequence) :
  ∀ n : ℕ, B seq (n + 1) = -2 * B seq n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_sequence_form_arithmetic_sequence_general_term_b_geometric_sequence_l586_58626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l586_58620

-- Define the function f
noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + φ)

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) (φ : ℝ) : ℝ := -Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + φ)

-- Define the function g
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := f x φ + f_derivative x φ

-- State the theorem
theorem phi_value (φ : ℝ) 
  (h1 : φ > -π ∧ φ < 0) 
  (h2 : ∀ x, g x φ = g (-x) φ) : 
  φ = -π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l586_58620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_is_integer_l586_58646

noncomputable def A (a b n : ℕ) (θ : ℝ) : ℝ := (a^2 + b^2)^n * Real.sin (n * θ)

theorem A_is_integer (a b : ℕ) (h1 : a > b) (h2 : b > 0) 
  (θ : ℝ) (h3 : 0 < θ ∧ θ < Real.pi / 2) 
  (h4 : Real.sin θ = (2 * a * b : ℝ) / ((a^2 + b^2) : ℝ)) :
  ∀ n : ℕ, ∃ k : ℤ, A a b n θ = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_is_integer_l586_58646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_sum_quarter_half_l586_58667

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infiniteGeometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: The sum of the infinite geometric series with first term 1/4 and common ratio 1/2 is 1/2 -/
theorem infinite_geometric_sum_quarter_half :
  infiniteGeometricSum (1/4 : ℝ) (1/2 : ℝ) = 1/2 := by
  -- Unfold the definition of infiniteGeometricSum
  unfold infiniteGeometricSum
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_sum_quarter_half_l586_58667
