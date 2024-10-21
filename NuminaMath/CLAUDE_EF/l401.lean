import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_edge_length_l401_40105

/-- A right pyramid with a hexagonal base -/
structure HexagonalPyramid where
  /-- Length of each side of the hexagonal base -/
  base_side : ℝ
  /-- Height of the pyramid (distance from peak to base center) -/
  height : ℝ

/-- Calculate the total length of edges for a hexagonal pyramid -/
noncomputable def total_edge_length (p : HexagonalPyramid) : ℝ :=
  let perimeter := 6 * p.base_side
  let slant_height := (p.base_side^2 + p.height^2).sqrt
  perimeter + 6 * slant_height

/-- Theorem: The total edge length of a specific hexagonal pyramid -/
theorem specific_pyramid_edge_length :
  ∃ (p : HexagonalPyramid), 
    p.base_side = 6 ∧ 
    p.height = 10 ∧ 
    total_edge_length p = 36 + 6 * (136 : ℝ).sqrt := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_edge_length_l401_40105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_distance_l401_40189

-- Define the points as 2D vectors
def A : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (3, 4)
def D : ℝ × ℝ := (3, 9)
def E : ℝ × ℝ := (5, 9)
def F : ℝ × ℝ := (5, 0)

-- Define distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem largest_distance :
  let AE := distance A E
  let CD := distance C D
  let CF := distance C F
  let AC := distance A C
  let FD := distance F D
  let CE := distance C E
  AC + CE = max AE (max (CD + CF) (max (AC + CF) (max FD (AC + CE)))) := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_distance_l401_40189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_determines_set_l401_40147

open Set

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

-- Define the complement of A with respect to U
def CUA : Set ℝ := {x | x > 4 ∨ x < 3}

-- Theorem statement
theorem complement_determines_set :
  ∀ a b : ℝ, ((A a b)ᶜ = CUA) → (a = 3 ∧ b = 4) :=
by
  intros a b h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_determines_set_l401_40147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_increasing_function_inequality_l401_40175

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is monotonically increasing on a set s if
    for all x, y in s, x < y implies f(x) < f(y) -/
def MonoIncOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

theorem odd_increasing_function_inequality
    (f : ℝ → ℝ)
    (h_odd : IsOdd f)
    (h_mono : MonoIncOn f (Set.Ici 0)) :
    {x : ℝ | f (2*x - 1) > f 1} = Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_increasing_function_inequality_l401_40175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_existence_l401_40114

/-- Given two distinct positive real numbers a and b, 
    we define a function f and prove the existence of a unique positive real number α 
    for which f(α) equals the cube root of (a³ + b³)/2, for all s in (0,1). -/
theorem unique_root_existence (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  let f : ℝ → ℝ := λ x ↦ -x + Real.sqrt ((x + a) * (x + b))
  ∀ s : ℝ, 0 < s ∧ s < 1 →
    ∃! α : ℝ, α > 0 ∧ f α = (((a^3 + b^3) / 2) ^ (1/3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_existence_l401_40114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_distances_area_range_l401_40137

-- Define the circle
def circleA (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 15 = 0

-- Define point B
def B : ℝ × ℝ := (1, 0)

-- Define the trajectory of point E
def trajectory (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1 ∧ y ≠ 0

-- Define the center of the circle
def A : ℝ × ℝ := (-1, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem constant_sum_distances :
  ∀ (x y : ℝ), trajectory x y →
    distance (x, y) A + distance (x, y) B = 4 := by sorry

-- Define the area of quadrilateral MPNQ
noncomputable def area_MPNQ (k : ℝ) : ℝ :=
  12 * Real.sqrt (1 + 1 / (4 * k^2 + 3))

-- Theorem for the range of the area
theorem area_range :
  ∀ (k : ℝ), k ≠ 0 → 12 ≤ area_MPNQ k ∧ area_MPNQ k < 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_distances_area_range_l401_40137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_inequality_l401_40164

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2)) ≥ 
  a * b * c + (((a^3 + a * b * c) * (b^3 + a * b * c) * (c^3 + a * b * c)) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_inequality_l401_40164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_inclination_line_through_points_with_45_degree_inclination_l401_40133

-- Define a line passing through two points
noncomputable def line_through_points (x1 y1 x2 y2 : ℝ) : ℝ → ℝ := 
  λ x ↦ y1 + (y2 - y1) / (x2 - x1) * (x - x1)

-- Define the angle of inclination of a line
noncomputable def angle_of_inclination (f : ℝ → ℝ) : ℝ :=
  Real.arctan ((f 1 - f 0) / 1)

-- Statement for B
theorem vertical_line_inclination :
  angle_of_inclination (line_through_points 1 (-3) 1 3) = π / 2 := by sorry

-- Statement for C
theorem line_through_points_with_45_degree_inclination :
  let f := λ x ↦ x + 1
  angle_of_inclination f = π / 4 ∧ f 3 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_inclination_line_through_points_with_45_degree_inclination_l401_40133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_greater_than_four_l401_40157

-- Define a standard die
def standard_die : Finset Nat := Finset.range 6

-- Define the favorable outcomes (numbers greater than 4)
def favorable_outcomes : Finset Nat := {5, 6}

-- Theorem statement
theorem probability_greater_than_four :
  (favorable_outcomes.card : ℚ) / standard_die.card = 1 / 3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_greater_than_four_l401_40157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_correct_l401_40128

/-- The coefficient of x^3 in the expansion of (2x + √x)^4 is 24 -/
def coefficient_x_cubed_in_expansion : ℕ := 24

/-- Helper function to represent the general term of the expansion -/
def general_term (r : ℕ) : ℚ :=
  (Nat.choose 4 r) * (2^(4 - r))

/-- The expansion of (2x + √x)^4 -/
noncomputable def expansion : Polynomial ℚ :=
  Finset.sum (Finset.range 5) (λ r => (general_term r : ℚ) • Polynomial.X^((8 - r) / 2))

/-- The coefficient of x^3 in the expansion is equal to the result of coefficient_x_cubed_in_expansion -/
theorem coefficient_x_cubed_correct :
  (expansion.coeff 3) = coefficient_x_cubed_in_expansion := by
  sorry

#eval coefficient_x_cubed_in_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_correct_l401_40128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l401_40126

-- Define the variables
variable (a w c d x : ℝ)

-- Define the equation
def equation (a w c d : ℝ) : Prop := ∀ x : ℝ, (a * x + w) * (c * x + d) = 6 * x^2 + x - 12

-- Theorem statement
theorem coefficient_of_x_squared (h : equation a w c d) : a * c = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l401_40126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_optimal_game_l401_40192

-- Define the set of numbers
def NumberSet : Set ℕ := Finset.range 82

-- Define the number of players
def NumPlayers : ℕ := 2

-- Define a strategy as a function that selects a number from the remaining set
def Strategy : Type := Finset ℕ → ℕ

-- Define the game result as the sums of numbers chosen by each player
structure GameResult :=
  (sum1 : ℕ)
  (sum2 : ℕ)

-- Define the optimal strategy for the first player
noncomputable def optimalStrategy1 : Strategy := sorry

-- Define the optimal strategy for the second player
noncomputable def optimalStrategy2 : Strategy := sorry

-- The main theorem
theorem gcd_of_optimal_game : 
  ∀ (result : GameResult), 
  (result.sum1 + result.sum2 = (81 * 82) / 2) →
  (∀ n ∈ NumberSet, n = result.sum1 ∨ n = result.sum2) →
  Nat.gcd result.sum1 result.sum2 = 41 := by
  sorry

#check gcd_of_optimal_game

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_optimal_game_l401_40192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_right_triangle_l401_40130

/-- Given an equilateral triangle XYZ with side length 6 and an inscribed right triangle DEF where DE is the hypotenuse, XD = 4, and DY = 3, prove that EZ = 3. -/
theorem inscribed_right_triangle (X Y Z D E F : ℝ × ℝ) : 
  -- Equilateral triangle XYZ with side length 6
  dist X Y = 6 ∧ dist Y Z = 6 ∧ dist Z X = 6 →
  -- Right triangle DEF inscribed in XYZ
  (D.1 - E.1) * (F.1 - E.1) + (D.2 - E.2) * (F.2 - E.2) = 0 →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • X + t • Y ∨ D = (1 - t) • Y + t • Z ∨ D = (1 - t) • Z + t • X) ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • X + t • Y ∨ E = (1 - t) • Y + t • Z ∨ E = (1 - t) • Z + t • X) ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (1 - t) • X + t • Y ∨ F = (1 - t) • Y + t • Z ∨ F = (1 - t) • Z + t • X) →
  -- DE is the hypotenuse
  dist D E ≥ dist D F ∧ dist D E ≥ dist E F →
  -- Given conditions
  dist X D = 4 →
  dist D Y = 3 →
  -- Prove that EZ = 3
  dist E Z = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_right_triangle_l401_40130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l401_40180

-- Define propositions p and q as axioms
axiom p : Prop
axiom q : Prop

-- Axioms for the truth values of p and q
axiom hp : ¬p
axiom hq : ¬q

-- Theorem to prove
theorem propositions_truth : 
  (¬p ∧ ¬q) ∧ ¬(p ∧ q) ∧ ¬(p ∨ q) ∧ (¬p ∧ ¬q) := by
  constructor
  · exact And.intro hp hq
  constructor
  · intro h
    exact hp h.left
  constructor
  · intro h
    cases h with
    | inl hp' => exact hp hp'
    | inr hq' => exact hq hq'
  · exact And.intro hp hq

#check propositions_truth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l401_40180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_preserving_l401_40138

-- Define the domain of the functions
def Domain : Type := {x : ℝ // x ≠ 0}

-- Define a geometric sequence
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 2) = (a (n + 1))^2

-- Define the functions
def f (x : Domain) : ℝ := x.val^2

noncomputable def g (x : Domain) : ℝ := Real.sqrt (abs x.val)

-- State the theorem
theorem geometric_sequence_preserving 
  (a : ℕ → Domain) 
  (h : IsGeometricSequence (fun n ↦ (a n).val)) :
  IsGeometricSequence (fun n ↦ f (a n)) ∧
  IsGeometricSequence (fun n ↦ g (a n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_preserving_l401_40138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_five_term_half_series_sum_l401_40194

theorem geometric_series_sum (a₀ r : ℚ) (n : ℕ) (h : r ≠ 1) :
  (Finset.range n).sum (λ i => a₀ * r ^ i) = a₀ * (1 - r ^ n) / (1 - r) :=
sorry

theorem five_term_half_series_sum :
  let a₀ : ℚ := 1 / 2
  let r : ℚ := 1 / 2
  let n : ℕ := 5
  (Finset.range n).sum (λ i => a₀ * r ^ i) = 31 / 32 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_five_term_half_series_sum_l401_40194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_5_l401_40161

-- Define the circle equation
def circle_equation (x y c : ℝ) : Prop :=
  x^2 + 10*x + y^2 + 8*y + c = 0

-- Define the center of the circle
def circle_center (x y : ℝ) : Prop :=
  x = -5 ∧ y = -4

-- Define the radius of the circle
noncomputable def circle_radius (x y c : ℝ) : ℝ :=
  Real.sqrt (41 - c)

-- Theorem statement
theorem circle_radius_5 (c : ℝ) :
  (∀ x y, circle_equation x y c → 
    ∃ h k, circle_center h k ∧ 
    ((x - h)^2 + (y - k)^2 = 25)) ↔ 
  c = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_5_l401_40161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_l401_40129

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x + x

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ≥ 1, Monotone (f a)) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_l401_40129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_proof_l401_40162

noncomputable def triangle_theorem (a b c : ℝ) (A B C : ℝ) : Prop :=
  let S := 2 * Real.sqrt 3
  c * Real.cos B + b * Real.cos C = 2 * a * Real.cos C ∧
  c = 2 * Real.sqrt 3 ∧
  1/2 * a * b * Real.sin C = S ∧
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi →
  C = Real.pi/3 ∧ ((a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2))

theorem triangle_theorem_proof : ∀ a b c A B C, triangle_theorem a b c A B C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_proof_l401_40162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l401_40185

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * Real.pi * x)^2

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 8 * Real.pi^2 * x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l401_40185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l401_40190

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def arithmetic_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 2 + a 8 = 4/3) :
  arithmetic_sum a 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l401_40190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l401_40166

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (5, 6)
def C : ℝ × ℝ := (8, 6)

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem longest_side_length :
  let sides := [distance A B, distance B C, distance A C]
  (sides.maximum? = some (Real.sqrt 52)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l401_40166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_expansion_l401_40182

theorem coefficient_expansion (a : ℝ) : 
  (∃ c : ℝ, c = 80 ∧ c = (Finset.range 6).sum (λ k ↦ (-1)^(5-k) * (Nat.choose 5 k) * a^k * (if k = 3 then 1 else 0))) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_expansion_l401_40182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_magnitude_l401_40109

theorem order_of_magnitude (π : ℝ) (h1 : π > 3) (h2 : π > 1) : 
  (0.3 : ℝ)^π < Real.sin (20*π/3) ∧ Real.sin (20*π/3) < π^(0.3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_magnitude_l401_40109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l401_40123

theorem cube_root_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) :
  (((1 / a + 6 * b) : ℝ) ^ (1/3 : ℝ)) + 
  (((1 / b + 6 * c) : ℝ) ^ (1/3 : ℝ)) + 
  (((1 / c + 6 * a) : ℝ) ^ (1/3 : ℝ)) ≤ 1 / (a * b * c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l401_40123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_line_AB_max_distance_l401_40167

/-- Curve C parameterized by θ -/
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

/-- Point A in Cartesian coordinates -/
def point_A : ℝ × ℝ := (-2, 0)

/-- Point B in Cartesian coordinates -/
noncomputable def point_B : ℝ × ℝ := (-1, -Real.sqrt 3)

/-- Line AB equation: √3x + y + 2√3 = 0 -/
noncomputable def line_AB (x y : ℝ) : Prop := Real.sqrt 3 * x + y + 2 * Real.sqrt 3 = 0

/-- Distance from a point to line AB -/
noncomputable def distance_to_line_AB (x y : ℝ) : ℝ :=
  (|2 * Real.sqrt 3 * x + y + 2 * Real.sqrt 3|) / 2

theorem curve_C_line_AB_max_distance :
  (∀ θ : ℝ, line_AB (curve_C θ).1 (curve_C θ).2 ↔ 
    Real.sqrt 3 * (2 * Real.cos θ) + Real.sin θ + 2 * Real.sqrt 3 = 0) ∧
  (∃ d : ℝ, d = (Real.sqrt 13 + 2 * Real.sqrt 3) / 2 ∧
    ∀ θ : ℝ, distance_to_line_AB (curve_C θ).1 (curve_C θ).2 ≤ d) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_line_AB_max_distance_l401_40167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l401_40149

theorem min_value_expression (θ : ℝ) (h1 : 0 < Real.sin θ) (h2 : Real.sin θ ≤ 1/4)
  (h3 : ∃ (x₁ x₂ x₃ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧
    x₁^3 * Real.sin θ - (Real.sin θ + 2) * x₁^2 + 6 * x₁ - 4 = 0 ∧
    x₂^3 * Real.sin θ - (Real.sin θ + 2) * x₂^2 + 6 * x₂ - 4 = 0 ∧
    x₃^3 * Real.sin θ - (Real.sin θ + 2) * x₃^2 + 6 * x₃ - 4 = 0) :
  (∀ u : ℝ, u ≥ (9 * (Real.sin θ)^2 - 4 * Real.sin θ + 3) /
    ((1 - Real.cos θ) * (2 * Real.cos θ - 6 * Real.sin θ - 3 * Real.sin (2 * θ) + 2)) →
    u ≥ 621/8) ∧
  (∃ u : ℝ, u = (9 * (Real.sin θ)^2 - 4 * Real.sin θ + 3) /
    ((1 - Real.cos θ) * (2 * Real.cos θ - 6 * Real.sin θ - 3 * Real.sin (2 * θ) + 2)) ∧
    u = 621/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l401_40149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_inequality_l401_40148

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define altitudes of the triangle
noncomputable def altitudes (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Define a point P inside the triangle
noncomputable def inside_point (t : Triangle) : ℝ × ℝ := sorry

-- Define distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_altitude_inequality (t : Triangle) :
  let (h_a, h_b, h_c) := altitudes t
  let P := inside_point t
  (distance P t.A / (h_b + h_c)) + 
  (distance P t.B / (h_a + h_c)) + 
  (distance P t.C / (h_a + h_b)) ≥ 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_inequality_l401_40148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l401_40187

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) + Real.log (4 - x)

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}

-- State the theorem
theorem f_properties :
  ∀ x y, x ∈ domain → y ∈ domain →
    ((-2 < x ∧ x < y ∧ y < 1) → f x < f y) ∧
    ((1 < x ∧ x < y ∧ y < 4) → f x > f y) ∧
    (∀ h, 1 - h ∈ domain → 1 + h ∈ domain → f (1 - h) = f (1 + h)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l401_40187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_paper_distance_l401_40159

-- Define the square sheet
def square_area : ℝ := 18

-- Define the folding condition
def black_area_double_white (x : ℝ) : Prop := (1/2) * x^2 = 2 * (square_area - x^2)

-- Define the distance function
noncomputable def distance_to_original (x : ℝ) : ℝ := Real.sqrt (2 * x^2)

-- Theorem statement
theorem folded_paper_distance :
  ∃ x : ℝ, black_area_double_white x ∧ 
  distance_to_original x = (6 * Real.sqrt 10) / 5 := by
  sorry

#check folded_paper_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_paper_distance_l401_40159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_random_events_l401_40127

/-- Represents the type of an event --/
inductive EventType
  | Certain
  | Impossible
  | Random
  deriving BEq, Repr

/-- Defines a list of four events --/
def events : List EventType := [EventType.Certain, EventType.Impossible, EventType.Random, EventType.Random]

/-- Counts the number of random events in a list --/
def countRandomEvents (list : List EventType) : Nat :=
  list.filter (λ e => e == EventType.Random) |>.length

/-- Theorem stating that there are exactly two random events in the given list --/
theorem two_random_events : countRandomEvents events = 2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_random_events_l401_40127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_degrees_decomposition_l401_40122

theorem tan_22_5_degrees_decomposition :
  ∃ (a b c d : ℕ), 
    (Real.tan (22.5 * π / 180) = (Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) - Real.sqrt (c : ℝ) + (d : ℝ)) ∧
     a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) →
    (a + b + c + d = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_degrees_decomposition_l401_40122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_visitor_ratio_l401_40170

theorem zoo_visitor_ratio (friday_visitors saturday_visitors : ℕ) :
  friday_visitors = 1250 ∧ saturday_visitors = 3750 →
  (saturday_visitors : ℚ) / (friday_visitors : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_visitor_ratio_l401_40170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_less_than_three_l401_40165

/-- A convex quadrilateral in 2D space -/
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

/-- The area of a quadrilateral -/
noncomputable def area (q : ConvexQuadrilateral) : ℝ := sorry

/-- The quadrilateral formed by reflecting each vertex with respect to the opposite diagonal -/
noncomputable def reflectedQuadrilateral (q : ConvexQuadrilateral) : ConvexQuadrilateral := sorry

/-- The theorem stating that the area ratio is less than 3 -/
theorem area_ratio_less_than_three (q : ConvexQuadrilateral) :
  area (reflectedQuadrilateral q) / area q < 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_less_than_three_l401_40165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_davey_has_18_guitars_l401_40100

/-- The total number of guitars owned by all people --/
def total_guitars : ℕ := 46

/-- The number of guitars Steve has --/
def steve : ℕ := 3  -- We set this based on our calculation

/-- The number of guitars Barbeck has --/
def barbeck : ℕ := 2 * steve

/-- The number of guitars Davey has --/
def davey : ℕ := 3 * barbeck

/-- The number of guitars Jane has --/
def jane : ℕ := (barbeck + davey) / 2 - 1

/-- Theorem stating that Davey has 18 guitars --/
theorem davey_has_18_guitars :
  steve + barbeck + davey + jane = total_guitars →
  davey = 18 := by
  intro h
  -- The proof goes here
  sorry

#eval davey  -- This will evaluate to 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_davey_has_18_guitars_l401_40100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_is_zero_matrix_l401_40118

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 6; -2, -3]

theorem matrix_inverse_is_zero_matrix :
  A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_is_zero_matrix_l401_40118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_move_l401_40113

/-- Represents a digit in matchstick notation -/
inductive MatchstickDigit
| zero
| one
| two
| three
| seven
| eleven

/-- Represents an equation in matchstick notation -/
structure MatchstickEquation where
  lhs : MatchstickDigit
  rhs : List MatchstickDigit
  operations : List (ℤ → ℤ → ℤ)

/-- Evaluates a matchstick digit to its integer value -/
def evalDigit : MatchstickDigit → ℤ
| MatchstickDigit.zero => 0
| MatchstickDigit.one => 1
| MatchstickDigit.two => 2
| MatchstickDigit.three => 3
| MatchstickDigit.seven => 7
| MatchstickDigit.eleven => 11

/-- Evaluates a matchstick equation -/
def evalEquation (eq : MatchstickEquation) : Bool :=
  let rhsValue := eq.rhs.zip eq.operations
    |>.foldl (fun acc (digit, op) => op acc (evalDigit digit)) 0
  evalDigit eq.lhs = rhsValue

/-- Represents a move of one matchstick -/
structure MatchstickMove where
  fromDigit : MatchstickDigit
  toDigit : MatchstickDigit

/-- Applies a move to an equation (placeholder implementation) -/
def applyMove (eq : MatchstickEquation) (move : MatchstickMove) : MatchstickEquation :=
  eq  -- Placeholder implementation, replace with actual logic

/-- Theorem: There exists a valid move that keeps the equation true -/
theorem exists_valid_move (eq : MatchstickEquation) : 
  ∃ (move : MatchstickMove), 
    evalEquation eq = true → 
    evalEquation (applyMove eq move) = true := by
  sorry

#check exists_valid_move

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_move_l401_40113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_moves_vertex_A_to_3_l401_40119

/-- Represents a face of the cube -/
inductive Face
| Green
| FarWhite
| BottomRightWhite
| TopLeftWhite

/-- Represents a vertex of the cube -/
def Vertex := Fin 8

/-- Represents the rotation of the cube -/
def Rotation := Face → Face

/-- Returns the vertex at the intersection of three faces -/
def intersectionVertex (f1 f2 f3 : Face) : Vertex := sorry

/-- The main theorem stating that after rotation, vertex A moves to vertex 3 -/
theorem rotation_moves_vertex_A_to_3 (r : Rotation) 
  (h1 : r Face.Green ≠ Face.Green)
  (h2 : r Face.FarWhite ≠ Face.FarWhite)
  (h3 : r Face.BottomRightWhite = Face.TopLeftWhite) :
  intersectionVertex (r Face.Green) (r Face.FarWhite) (r Face.BottomRightWhite) = ⟨3, by norm_num⟩ := by
  sorry

#check rotation_moves_vertex_A_to_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_moves_vertex_A_to_3_l401_40119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_solution_l401_40172

/-- Given a positive integer n satisfying (n+1)! + (n+2)! = n! * 675, prove that n = 24 -/
theorem unique_n_solution (n : ℕ) (hn : n > 0) (h : (n + 1).factorial + (n + 2).factorial = n.factorial * 675) : n = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_solution_l401_40172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l401_40115

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 / Real.sqrt (1 - 2*x) + Real.log (1 + 2*x)

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, (∃ y, f x = y) ↔ -1/2 < x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l401_40115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersects_circle_l401_40178

-- Define the curve
noncomputable def curve (a x : ℝ) : ℝ := Real.exp x * (x^2 + a*x + 1 - 2*a)

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 16

-- Define the tangent line
def tangent_line (a x y : ℝ) : Prop := 
  y - (1 - 2*a) = (1 - a) * x

-- Theorem statement
theorem tangent_intersects_circle (a : ℝ) : 
  ∃ x y : ℝ, tangent_line a x y ∧ circle_eq x y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersects_circle_l401_40178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l401_40120

noncomputable def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (x + φ)

theorem function_properties (A : ℝ) (φ : ℝ) 
  (h1 : A > 0) (h2 : 0 < φ) (h3 : φ < Real.pi)
  (h4 : ∀ x, f A φ x ≤ 1) 
  (h5 : f A φ (Real.pi / 3) = 1) :
  (∀ x, f 1 (Real.pi / 6) x = f A φ x) ∧
  (∀ k : ℤ, StrictMonoOn (f 1 (Real.pi / 6)) 
    (Set.Icc (2 * k * Real.pi - 2 * Real.pi / 3) (2 * k * Real.pi + Real.pi / 3))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l401_40120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_specific_quadratic_l401_40199

/-- A quadratic equation with coefficients a, b, and c in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4 * eq.a * eq.c

/-- A quadratic equation has exactly one real root if and only if its discriminant is zero -/
def has_exactly_one_root (eq : QuadraticEquation) : Prop :=
  discriminant eq = 0

/-- The root of a quadratic equation with exactly one real root -/
noncomputable def root_of_one_root_equation (eq : QuadraticEquation) : ℝ :=
  -eq.b / (2 * eq.a)

theorem root_of_specific_quadratic :
  ∀ C : ℝ,
  let eq : QuadraticEquation := { a := 2, b := 20, c := C }
  has_exactly_one_root eq →
  root_of_one_root_equation eq = -5 := by
  sorry

#check root_of_specific_quadratic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_specific_quadratic_l401_40199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_special_number_l401_40107

/-- A five-digit number represented as a list of its digits -/
def FiveDigitNumber := { d : List Nat // d.length = 5 }

/-- The value of a five-digit number -/
def value (n : FiveDigitNumber) : Nat :=
  n.val[0]! * 10000 + n.val[1]! * 1000 + n.val[2]! * 100 + n.val[3]! * 10 + n.val[4]!

/-- The sum of digits of a five-digit number -/
def digitSum (n : FiveDigitNumber) : Nat :=
  n.val[0]! + n.val[1]! + n.val[2]! + n.val[3]! + n.val[4]!

theorem digit_sum_of_special_number :
  ∀ (n : FiveDigitNumber), value n * 3 = 111111 → digitSum n = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_special_number_l401_40107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_ant_path_l401_40179

/-- A square with vertices A, B, C, D -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The path of an ant starting and ending at point P, touching three sides of the square -/
structure AntPath (S : Square) where
  P : ℝ × ℝ
  path : List (ℝ × ℝ)
  touches_three_sides : path.length ≥ 3
  starts_ends_at_P : path.head? = some P ∧ path.getLast? = some P

/-- The length of a path -/
noncomputable def pathLength (path : List (ℝ × ℝ)) : ℝ := sorry

/-- The theorem to be proved -/
theorem shortest_ant_path (S : Square) 
  (h1 : S.A = (0, 0))
  (h2 : S.B = (1, 1))
  (P : ℝ × ℝ)
  (h3 : P = (2/7, 1/4)) :
  ∃ (path : AntPath S), 
    (∀ (other_path : AntPath S), pathLength path.path ≤ pathLength other_path.path) ∧
    pathLength path.path = Real.sqrt 17 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_ant_path_l401_40179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_ADB_in_terms_of_x_y_l401_40111

-- Define a tetrahedron structure
structure Tetrahedron where
  A : EuclideanSpace ℝ (Fin 3)
  B : EuclideanSpace ℝ (Fin 3)
  C : EuclideanSpace ℝ (Fin 3)
  D : EuclideanSpace ℝ (Fin 3)

-- Define angle function
def angle (p q r : EuclideanSpace ℝ (Fin 3)) : ℝ := sorry

-- Define the theorem
theorem sin_angle_ADB_in_terms_of_x_y (ABCD : Tetrahedron)
  (right_angles : angle ABCD.A ABCD.B ABCD.C = π/2 ∧ 
                  angle ABCD.A ABCD.C ABCD.B = π/2 ∧ 
                  angle ABCD.B ABCD.A ABCD.C = π/2)
  (x : ℝ) (hx : x = Real.cos (angle ABCD.C ABCD.A ABCD.D))
  (y : ℝ) (hy : y = Real.cos (angle ABCD.C ABCD.B ABCD.D)) :
  Real.sin (angle ABCD.A ABCD.D ABCD.B) = (x * y) / Real.sqrt (x^2 + y^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_ADB_in_terms_of_x_y_l401_40111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_time_taken_l401_40195

/-- Actual time taken by a car to cover a distance, given it's 15 minutes late when running at 4/5 of its actual speed -/
theorem actual_time_taken (S : ℝ) : 
  S > 0 → -- Actual speed is positive
  (let T := 1 -- Actual time taken (to be proven)
   let T_late := T + 1/4 -- Time taken when late (15 minutes = 1/4 hour)
   T_late / T = 5/4) → -- Speed and time are inversely proportional
  1 = 1 := by
  intro hS hProp
  -- The proof goes here
  sorry

#check actual_time_taken

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_time_taken_l401_40195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l401_40112

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.log x + 1

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 1)

-- Theorem stating that g is the inverse of f for x > 1
theorem f_inverse_is_g :
  ∀ x : ℝ, x > 1 → 
    (f (g x) = x ∧ g (f x) = x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l401_40112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABCDE_prime_l401_40198

-- Define the perimeters and similarity ratio
noncomputable def perimeter_ABCDE : ℝ := 6
noncomputable def similarity_ratio : ℝ := 3 / 4

-- Define the theorem
theorem perimeter_ABCDE_prime (perimeter_ABCDE_prime : ℝ) 
  (h : perimeter_ABCDE_prime / perimeter_ABCDE = 1 / similarity_ratio) : 
  perimeter_ABCDE_prime = 8 := by
  -- Replace the entire proof with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABCDE_prime_l401_40198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l401_40168

/-- The distance between two points in a plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The set of points (x, y) at a distance of √3 from (-1, 0) -/
def points_set (x y : ℝ) : Prop :=
  distance x y (-1) 0 = Real.sqrt 3

theorem circle_equation (x y : ℝ) :
  points_set x y ↔ (x + 1)^2 + y^2 = 3 := by
  sorry

#check circle_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l401_40168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l401_40169

theorem triangle_side_length (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 - 6*a - 10*b + 34 = 0 → 
  a < c ∧ b < c →
  c = 6 ∨ c = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l401_40169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_g_l401_40143

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (4 * x + Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := f ((x - Real.pi / 6) / 2)

theorem axis_of_symmetry_g :
  ∀ x : ℝ, g (Real.pi / 3 + x) = g (Real.pi / 3 - x) :=
by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_g_l401_40143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_triangle_area_l401_40136

noncomputable section

-- Define the areas of the squares
def square1_area : ℝ := 25
def square2_area : ℝ := 144
def square3_area : ℝ := 169

-- Define the triangle area calculation function
def triangle_area (a b : ℝ) : ℝ := (1/2) * a * b

-- Theorem statement
theorem interior_triangle_area : 
  ∃ (s1 s2 s3 : ℝ), 
    s1^2 = square1_area ∧ 
    s2^2 = square2_area ∧ 
    s3^2 = square3_area ∧ 
    s1^2 + s2^2 = s3^2 ∧
    triangle_area s1 s2 = 30 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_triangle_area_l401_40136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l401_40108

theorem range_of_a (x y : ℝ) (hx : x > 4) (hy : y ≥ 4) (hxy : x + 4*y - x*y = 0) :
  (∀ a : ℝ, x - y + 6 ≤ a ∧ a ≤ x + y - 1) → 
  x - y + 6 = 22/3 ∧ x + y - 1 = 25/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l401_40108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_tetrahedrons_l401_40134

noncomputable def largeTetrahedronVertices : List (Fin 3 → ℝ) := [
  ![0, 0, 0],
  ![1, 0, 0],
  ![0, 1, 0],
  ![0, 0, 1]
]

noncomputable def faceCenter (v1 v2 v3 : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => (v1 i + v2 i + v3 i) / 3

noncomputable def smallTetrahedronVertices (vertices : List (Fin 3 → ℝ)) : List (Fin 3 → ℝ) :=
  [
    faceCenter vertices[1]! vertices[2]! vertices[3]!,
    faceCenter vertices[0]! vertices[2]! vertices[3]!,
    faceCenter vertices[0]! vertices[1]! vertices[3]!,
    faceCenter vertices[0]! vertices[1]! vertices[2]!
  ]

noncomputable def tetrahedronVolume (vertices : List (Fin 3 → ℝ)) : ℝ :=
  sorry

theorem volume_ratio_of_tetrahedrons :
  let largeVertices := largeTetrahedronVertices
  let smallVertices := smallTetrahedronVertices largeVertices
  (tetrahedronVolume smallVertices) / (tetrahedronVolume largeVertices) = 2 * Real.sqrt 2 / 27 := by
  sorry

#check volume_ratio_of_tetrahedrons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_tetrahedrons_l401_40134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_l401_40132

/-- The perimeter of a regular hexagon with side length 10 is 60. -/
theorem hexagon_perimeter (side_length : ℝ) : side_length = 10 → 6 * side_length = 60 := by
  intro h
  rw [h]
  norm_num

#check hexagon_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_l401_40132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_l401_40183

/-- The equation of the curve -/
def curve_equation (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 - 9*y^2 + 6*x = 0

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ × ℝ → Prop) : Prop :=
  ∃ a b h k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧
  ∀ x y, f (x, y) ↔ ((x - h) / a)^2 - ((y - k) / b)^2 = 1

/-- Theorem stating that the curve equation represents a hyperbola -/
theorem curve_is_hyperbola : is_hyperbola curve_equation := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_l401_40183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l401_40121

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the theorem
theorem solution_set_of_inequality 
  (h1 : ∀ x, f' x - 2 * f x > 4)
  (h2 : f 0 = -1)
  : Set.Ioi 0 = {x | f x + 2 > Real.exp (2 * x)} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l401_40121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exscribed_circles_radii_product_l401_40124

/-- Given a triangle with side lengths a, b, c and exscribed circles with radii r_a, r_b, r_c,
    the product of the radii is at most (√3/8) times the product of the side lengths,
    with equality if and only if the triangle is equilateral. -/
theorem exscribed_circles_radii_product (a b c r_a r_b r_c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_exscribed : r_a > 0 ∧ r_b > 0 ∧ r_c > 0) 
  (h_relation : ∃ (p S : ℝ), 
    p = (a + b + c) / 2 ∧ 
    S = Real.sqrt (p * (p - a) * (p - b) * (p - c)) ∧ 
    r_a * (p - a) = S ∧ r_b * (p - b) = S ∧ r_c * (p - c) = S) :
  r_a * r_b * r_c ≤ (Real.sqrt 3 / 8) * a * b * c ∧ 
  (r_a * r_b * r_c = (Real.sqrt 3 / 8) * a * b * c ↔ a = b ∧ b = c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exscribed_circles_radii_product_l401_40124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_l401_40153

-- Define the angle α
variable (α : Real)

-- Define the point P
variable (y : Real)

-- State the conditions
axiom terminal_side : ∃ (P : Real × Real), P = (3, y) ∧ P.2 < 0
axiom cos_value : Real.cos α = 3/5

-- State the theorem
theorem tan_value : Real.tan α = -4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_l401_40153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_camp_children_l401_40188

theorem summer_camp_children (total : ℚ) : 
  (0.9 * total = total - total * 0.1) →
  (total * 0.1 = 0.05 * (total + 100)) →
  total = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_camp_children_l401_40188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l401_40102

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a + t.c = 4 ∧
  (2 - Real.cos t.A) * Real.tan (t.B / 2) = Real.sin t.A ∧
  t.a = 2 * Real.sin t.A ∧
  t.b = 2 * Real.sin t.B ∧
  t.c = 2 * Real.sin t.C

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ :=
  Real.sqrt (3 * (3 - t.a) * (3 - t.b) * (3 - t.c))

-- State the theorem
theorem max_area_triangle (t : Triangle) (h : satisfiesConditions t) :
  area t ≤ Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l401_40102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_iterations_l401_40186

-- Define the function f(x) = -ln x
noncomputable def f (x : ℝ) : ℝ := -Real.log x

-- State the theorem
theorem bisection_method_iterations :
  ∃ (x₀ : ℝ), x₀ ∈ Set.Ioo 1 2 ∧ f x₀ = 0 →
  ∃ (n : ℕ), n = 4 ∧
    ∀ (m : ℕ), (2 - 1) / (2^m : ℝ) ≤ 0.1 → m ≥ n :=
by
  sorry

#check bisection_method_iterations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_iterations_l401_40186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_salmon_oxygen_consumption_l401_40125

/-- The swimming speed of a salmon in m/s -/
noncomputable def swimming_speed (x : ℝ) : ℝ := (1 / 2) * Real.log ((x / 100) * Real.pi) / Real.log 3

/-- Theorem: The unit of oxygen consumption for a stationary salmon is 100/π -/
theorem stationary_salmon_oxygen_consumption :
  ∃ x : ℝ, swimming_speed x = 0 ∧ x = 100 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_salmon_oxygen_consumption_l401_40125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dense_position_solutions_l401_40140

/-- Dense position system representation of an angle --/
structure DensePosition where
  value : ℕ
  inv_value : value < 6000

/-- Convert dense position to radians --/
noncomputable def dense_to_radians (d : DensePosition) : ℝ :=
  (d.value : ℝ) / 6000 * (2 * Real.pi)

/-- Check if a dense position is a solution to the equation --/
noncomputable def is_solution (d : DensePosition) : Prop :=
  let α := dense_to_radians d
  (Real.sin α - Real.cos α)^2 = 2 * Real.sin α * Real.cos α

/-- The theorem to be proved --/
theorem dense_position_solutions : 
  ∀ d : DensePosition, is_solution d ↔ 
    d.value = 1250 ∨ d.value = 250 ∨ d.value = 3250 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dense_position_solutions_l401_40140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_3_and_4_are_correct_l401_40176

-- Definition of an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Definition of acute angle
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

theorem propositions_3_and_4_are_correct :
  (is_even (λ x => Real.sin (2/3 * x + Real.pi/2))) ∧
  (∀ A B C : ℝ, is_acute A → is_acute B → is_acute C → A + B + C = Real.pi → Real.sin A > Real.cos B) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_3_and_4_are_correct_l401_40176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_pi_l401_40174

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 3 = 0

-- Define the area of a circle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Theorem statement
theorem circle_area_is_pi : 
  ∃ (r : ℝ), r > 0 ∧ (∀ x y : ℝ, circle_equation x y ↔ (x + 2)^2 + y^2 = r^2) ∧ circle_area r = Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_pi_l401_40174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l401_40154

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 - 1 / 2

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) ∧
  (∀ x, f (x + π) = f x) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l401_40154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_markup_rate_l401_40116

/-- Given a toy with a selling price, profit percentage, and expense percentage,
    calculate the rate of markup on the cost of the toy. -/
noncomputable def rate_of_markup (selling_price : ℝ) (profit_percent : ℝ) (expense_percent : ℝ) : ℝ :=
  let cost := selling_price * (1 - profit_percent - expense_percent)
  ((selling_price - cost) / cost) * 100

/-- The rate of markup on the cost of a toy with given conditions is approximately 47.06%. -/
theorem toy_markup_rate :
  let selling_price : ℝ := 8
  let profit_percent : ℝ := 0.12
  let expense_percent : ℝ := 0.20
  abs (rate_of_markup selling_price profit_percent expense_percent - 47.06) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_markup_rate_l401_40116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlas_initial_marbles_l401_40160

-- Define the variables
def initial_marbles : ℝ := sorry
def bought_marbles : ℝ := 489.0
def total_marbles : ℝ := 2778.0

-- State the theorem
theorem carlas_initial_marbles :
  initial_marbles = total_marbles - bought_marbles :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlas_initial_marbles_l401_40160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_is_one_l401_40106

/-- Curve C₁ in parametric form -/
noncomputable def curve_C1 (t : ℝ) : ℝ × ℝ := (2 * t^2, 2 * t)

/-- Curve C₂ in polar form -/
noncomputable def curve_C2 (a : ℝ) (θ : ℝ) : ℝ := 2 / (Real.sin θ + a * Real.cos θ)

/-- Intersection points of C₁ and C₂ -/
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t θ, curve_C1 t = p ∧ 
       curve_C2 a θ * Real.cos θ = p.1 ∧ 
       curve_C2 a θ * Real.sin θ = p.2}

/-- Slope of the line connecting the origin to a point -/
noncomputable def slope_to_origin (p : ℝ × ℝ) : ℝ := p.2 / p.1

theorem sum_of_slopes_is_one (a : ℝ) (ha : a ≠ 0) :
  ∀ p₁ p₂, p₁ ∈ intersection_points a → p₂ ∈ intersection_points a → p₁ ≠ p₂ → 
  slope_to_origin p₁ + slope_to_origin p₂ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_is_one_l401_40106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_pi_l401_40110

-- Define a right triangle with hypotenuse h and circumscribed circle of radius R
structure RightTriangle where
  h : ℝ  -- Length of hypotenuse
  R : ℝ  -- Radius of circumscribed circle
  h_pos : 0 < h  -- Hypotenuse length is positive
  R_pos : 0 < R  -- Radius is positive
  euler_theorem : R = h / 2  -- Euler's theorem for right triangles

-- Define the ratio of areas
noncomputable def area_ratio (t : RightTriangle) : ℝ :=
  (Real.pi * t.R^2) / ((t.h^2) / 4)

-- Theorem statement
theorem area_ratio_is_pi (t : RightTriangle) : area_ratio t = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_pi_l401_40110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_identity_l401_40197

theorem sin_product_identity (θ : Real) :
  Real.sin θ * Real.sin (π/3 - θ) * Real.sin (π/3 + θ) = (1/4) * Real.sin (3*θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_identity_l401_40197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l401_40163

theorem solve_exponential_equation (x : ℝ) :
  (3 : ℝ)^(x - 4) = 27^(3/2) → x = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l401_40163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_flow_speed_l401_40141

/-- The speed of water flow given the rowing conditions -/
theorem water_flow_speed (total_distance : ℝ) (usual_speed : ℝ) (water_speed : ℝ)
  (h1 : total_distance > 0)
  (h2 : usual_speed > water_speed)
  (h3 : total_distance / (usual_speed - water_speed) - total_distance / (usual_speed + water_speed) = 5)
  (h4 : total_distance / (2 * usual_speed - water_speed) - total_distance / (2 * usual_speed + water_speed) = 1) :
  water_speed = 2 := by
  sorry

#check water_flow_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_flow_speed_l401_40141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l401_40142

noncomputable def C (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - Real.arcsin a) * (p.1 - Real.arccos a) + (p.2 - Real.arcsin a) * (p.2 + Real.arccos a) = 0}

noncomputable def chord_length (a : ℝ) : ℝ :=
  let p₁ := (Real.pi/4, Real.sqrt ((Real.pi/4 - Real.arcsin a) * (Real.pi/4 - Real.arccos a)))
  let p₂ := (Real.pi/4, -Real.sqrt ((Real.pi/4 - Real.arcsin a) * (Real.pi/4 - Real.arccos a)))
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem min_chord_length :
  ∀ a : ℝ, chord_length a ≥ Real.pi/2 ∧ ∃ a₀ : ℝ, chord_length a₀ = Real.pi/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l401_40142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_sector_l401_40103

theorem cone_height_from_circular_sector (r : ℝ) (h : r = 8) : 
  let circumference := 2 * Real.pi * r;
  let sector_arc_length := circumference / 4;
  let base_radius := sector_arc_length / (2 * Real.pi);
  let slant_height := r;
  let height := Real.sqrt (slant_height^2 - base_radius^2)
  height = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_sector_l401_40103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_hiker_problem_l401_40139

/-- Calculates the total waiting time for a cyclist to be caught by a hiker -/
noncomputable def cyclist_wait_time (hiker_speed : ℝ) (cyclist_speed : ℝ) (initial_wait_time : ℝ) : ℝ :=
  let distance_traveled := cyclist_speed * (initial_wait_time / 60)
  let relative_speed := cyclist_speed - hiker_speed
  let catch_up_time := distance_traveled / (relative_speed / 60)
  initial_wait_time + catch_up_time

theorem cyclist_hiker_problem :
  let hiker_speed := (4 : ℝ)
  let cyclist_speed := (30 : ℝ)
  let initial_wait_time := (5 : ℝ)
  abs (cyclist_wait_time hiker_speed cyclist_speed initial_wait_time - 10.77) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_hiker_problem_l401_40139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_volume_l401_40184

/-- A right prism with a parallelogram base -/
structure RightPrism where
  φ : ℝ  -- Acute angle between base diagonals
  α : ℝ  -- Angle of intersection of diagonals in one lateral face
  β : ℝ  -- Angle of intersection of diagonals in the other lateral face
  h : ℝ  -- Height of the prism
  α_gt_β : α > β
  φ_acute : 0 < φ ∧ φ < π / 2

/-- The volume of the right prism -/
noncomputable def volume (p : RightPrism) : ℝ :=
  (p.h ^ 3 / 2) * Real.tan p.φ * (Real.sin ((p.α - p.β) / 2) * Real.sin ((p.α + p.β) / 2)) /
  (Real.cos (p.α / 2) ^ 2 * Real.cos (p.β / 2) ^ 2)

theorem right_prism_volume (p : RightPrism) :
  volume p = (p.h ^ 3 / 2) * Real.tan p.φ * (Real.sin ((p.α - p.β) / 2) * Real.sin ((p.α + p.β) / 2)) /
             (Real.cos (p.α / 2) ^ 2 * Real.cos (p.β / 2) ^ 2) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_prism_volume_l401_40184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sqrt2_lt_1_iff_a_range_l401_40193

theorem log_sqrt2_lt_1_iff_a_range (a : ℝ) :
  (0 < a ∧ (Real.log (Real.sqrt 2) / Real.log a < 1)) ↔ (0 < a ∧ a < 1) ∨ (a > Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sqrt2_lt_1_iff_a_range_l401_40193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_power_l401_40177

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property that (n+i)^6 is an integer
def is_integer_power (n : ℤ) : Prop :=
  ∃ m : ℤ, (Complex.ofReal (↑n) + i) ^ 6 = Complex.ofReal (↑m)

-- The main theorem
theorem unique_integer_power :
  (i ^ 2 = -1) →
  ∃! n : ℤ, is_integer_power n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_power_l401_40177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_tangent_l401_40158

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

/-- The hyperbola equation -/
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+1)^2 = 1

/-- Two curves are tangent if they intersect at exactly one point -/
def tangent (f g : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, f p.fst p.snd ∧ g p.fst p.snd

theorem ellipse_hyperbola_tangent :
  tangent (λ x y ↦ ellipse x y) (λ x y ↦ hyperbola x y 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_tangent_l401_40158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l401_40145

/-- Calculate compound interest given principal, rate, and time -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Calculate simple interest given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * (time : ℝ) / 100

theorem interest_calculation (P : ℝ) (h : simpleInterest P 5 2 = 50) :
  compoundInterest P 5 2 = 51.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l401_40145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_c_l401_40101

theorem triangle_cosine_c (A B C : Real) : 
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  Real.sin A = 3/5 →
  Real.cos B = 5/13 →
  Real.cos C = 16/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_c_l401_40101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_implies_a_range_l401_40152

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1/3) * Real.sin (2*x) + a * Real.sin x

theorem monotonic_increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (-1/3) (1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_implies_a_range_l401_40152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_solutions_iff_valid_c_l401_40171

open Set

def is_valid_c (c : ℕ) : Prop :=
  c ∈ ({1363, 1377, 1391, 1396, 1401, 1405, 1410, 1415} : Set ℕ) ∪
  Icc 1419 1420 ∪ ({1424, 1429} : Set ℕ) ∪ Icc 1433 1434 ∪
  Icc 1438 1439 ∪ ({1443} : Set ℕ) ∪ Icc 1447 1448 ∪
  Icc 1452 1453 ∪ Icc 1457 1458 ∪ Icc 1461 1462 ∪
  Icc 1471 1472 ∪ Icc 1475 1477 ∪ Icc 1480 1481 ∪
  Icc 1485 1486 ∪ Icc 1489 1491 ∪ Icc 1494 1496 ∪
  Icc 1499 1500 ∪ Icc 1503 1505 ∪ Icc 1508 1511 ∪
  Icc 1513 1515 ∪ Icc 1517 1519 ∪ Icc 1522 1524 ∪
  Icc 1527 1529 ∪ Icc 1531 1534 ∪ Icc 1536 1538 ∪
  Icc 1541 1543 ∪ Icc 1545 1548 ∪ Icc 1550 1553 ∪
  Icc 1555 1557 ∪ Icc 1559 1562 ∪ Icc 1564 1567 ∪
  Icc 1569 1576 ∪ Icc 1578 1581 ∪ Icc 1583 1595 ∪
  Icc 1597 1609 ∪ Icc 1611 1628 ∪ Icc 1630 1642 ∪
  Icc 1644 1647 ∪ Icc 1649 1656 ∪ Icc 1658 1661 ∪
  Icc 1663 1666 ∪ Icc 1668 1670 ∪ Icc 1672 1675 ∪
  Icc 1677 1680 ∪ Icc 1682 1684 ∪ Icc 1687 1689 ∪
  Icc 1691 1694 ∪ Icc 1696 1698 ∪ Icc 1701 1703 ∪
  Icc 1706 1708 ∪ Icc 1710 1712 ∪ Icc 1715 1717 ∪
  Icc 1720 1722 ∪ Icc 1725 1726 ∪ Icc 1729 1731 ∪
  Icc 1734 1736 ∪ Icc 1739 1740 ∪ Icc 1744 1745 ∪
  Icc 1748 1750 ∪ Icc 1753 1754 ∪ Icc 1758 1759 ∪
  Icc 1763 1764 ∪ Icc 1767 1768 ∪ Icc 1772 1773 ∪
  Icc 1777 1778 ∪ Icc 1782 1783 ∪ Icc 1786 1787 ∪
  Icc 1791 1792 ∪ ({1796, 1801, 1805, 1810, 1815, 1820, 1829, 1834, 1848, 1862} : Set ℕ)

def has_six_natural_solutions (c : ℕ) : Prop :=
  (Finset.filter (fun p : ℕ × ℕ => 19 * p.1 + 14 * p.2 = c) (Finset.product (Finset.range 1000) (Finset.range 1000))).card = 6

theorem six_solutions_iff_valid_c (c : ℕ) :
  has_six_natural_solutions c ↔ is_valid_c c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_solutions_iff_valid_c_l401_40171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l401_40104

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

-- Define the interval
def interval : Set ℝ := { x | -Real.pi/6 ≤ x ∧ x ≤ Real.pi/4 }

-- Theorem statement
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (max : ℝ), max = Real.sqrt 3 ∧ ∀ (x : ℝ), x ∈ interval → f x ≤ max) ∧
  (∃ (min : ℝ), min = -2 ∧ ∀ (x : ℝ), x ∈ interval → min ≤ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l401_40104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_equal_and_max_value_l401_40144

-- Define the solution set of |x-2| > 1
def solution_set_1 : Set ℝ := {x | |x - 2| > 1}

-- Define the solution set of x^2 - 4x + 3 > 0
def solution_set_2 : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 4 * (x - 3).sqrt + 3 * (5 - x).sqrt

theorem solution_sets_equal_and_max_value :
  (solution_set_1 = solution_set_2) ∧
  (∃ (x_max : ℝ), x_max ∈ Set.Icc 3 5 ∧
    (∀ (x : ℝ), x ∈ Set.Icc 3 5 → f x ≤ f x_max) ∧
    f x_max = 5 * Real.sqrt 2 ∧
    x_max = 107 / 25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_equal_and_max_value_l401_40144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_acute_l401_40173

/-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively,
    if 2/a = 1/b + 1/c, then angle A is acute. -/
theorem triangle_angle_acute (a b c : ℝ) (h : 2/a = 1/b + 1/c) :
  ∃ (A B C : ℝ), 
    0 < A ∧ A < Real.pi/2 ∧
    0 < B ∧ B < Real.pi ∧
    0 < C ∧ C < Real.pi ∧
    A + B + C = Real.pi ∧
    a = b * Real.sin C / Real.sin A ∧
    b = c * Real.sin A / Real.sin B ∧
    c = a * Real.sin B / Real.sin C :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_acute_l401_40173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l401_40131

-- Define the hyperbola and its properties
noncomputable def hyperbola (m : ℝ) := λ (x y : ℝ) => x^2 / 9 - y^2 / m = 1

-- Define eccentricity
noncomputable def eccentricity (m : ℝ) := Real.sqrt (1 + m / 9)

-- Theorem statement
theorem hyperbola_m_value (m : ℝ) 
  (h1 : m > 0)
  (h2 : eccentricity m = 2) :
  m = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l401_40131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_for_perpendicular_chords_l401_40196

/-- An ellipse with semi-major axis a, semi-minor axis b, and right focus at (c, 0) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_equation : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ Set.range (λ t : ℝ ↦ (a * Real.cos t, b * Real.sin t))

/-- A line passing through the focus F(c, 0) of the ellipse -/
def line_through_focus (E : Ellipse) (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = k * p.2 + E.c}

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ := E.c / E.a

/-- The theorem stating the range of eccentricity for an ellipse with perpendicular intersecting chords -/
theorem eccentricity_range_for_perpendicular_chords (E : Ellipse) :
  (∃ k : ℝ, ∃ A B : ℝ × ℝ,
    A ∈ line_through_focus E k ∧
    B ∈ line_through_focus E k ∧
    A ∈ Set.range (λ t : ℝ ↦ (E.a * Real.cos t, E.b * Real.sin t)) ∧
    B ∈ Set.range (λ t : ℝ ↦ (E.a * Real.cos t, E.b * Real.sin t)) ∧
    A.1 * B.1 + A.2 * B.2 = 0) →
  (Real.sqrt 5 - 1) / 2 ≤ eccentricity E ∧ eccentricity E < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_for_perpendicular_chords_l401_40196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_buckets_after_reduction_l401_40155

theorem total_buckets_after_reduction :
  ∃ (tank1 tank2 tank3 : ℕ) (reduction1 reduction2 reduction3 : ℚ),
  tank1 = 25 ∧
  tank2 = 35 ∧
  tank3 = 45 ∧
  reduction1 = 2/5 ∧
  reduction2 = 3/5 ∧
  reduction3 = 4/5 ∧
  ∀ (new_tank1 new_tank2 new_tank3 : ℕ),
    (new_tank1 = ⌈(tank1 : ℚ) / reduction1⌉) →
    (new_tank2 = ⌈(tank2 : ℚ) / reduction2⌉) →
    (new_tank3 = ⌈(tank3 : ℚ) / reduction3⌉) →
    new_tank1 + new_tank2 + new_tank3 = 179
  := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_buckets_after_reduction_l401_40155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_contains_large_hexagon_l401_40135

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  is_convex : Convex ℝ vertices

/-- A convex hexagon in a 2D plane -/
structure ConvexHexagon where
  vertices : Fin 6 → (ℝ × ℝ)
  is_convex : Convex ℝ (Set.range vertices)

/-- The area of a polygon -/
noncomputable def area (P : Set (ℝ × ℝ)) : ℝ := sorry

/-- A hexagon H is contained in a polygon P -/
def contained_in (H : ConvexHexagon) (P : ConvexPolygon) : Prop :=
  ∀ v, v ∈ Set.range H.vertices → v ∈ P.vertices

/-- Main theorem: For any convex polygon, there exists a convex hexagon within it
    with an area at least 3/4 of the polygon's area -/
theorem convex_polygon_contains_large_hexagon (P : ConvexPolygon) :
  ∃ H : ConvexHexagon, contained_in H P ∧ area (Set.range H.vertices) ≥ 3/4 * area P.vertices := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_contains_large_hexagon_l401_40135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_from_cosine_condition_l401_40156

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of being isosceles
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

-- State the theorem
theorem triangle_isosceles_from_cosine_condition (t : Triangle) 
  (h : t.a * Real.cos t.B = t.b * Real.cos t.A) : 
  t.isIsosceles := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_from_cosine_condition_l401_40156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_m_values_l401_40181

def collatz_sequence (m : ℕ) : ℕ → ℕ
  | 0 => m
  | n + 1 => 
    let a := collatz_sequence m n
    if a % 2 = 0 then a / 2 else 3 * a + 1

theorem possible_m_values : 
  ∀ m : ℕ, m > 0 → (collatz_sequence m 5 = 1 ↔ m ∈ ({4, 5, 32} : Set ℕ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_m_values_l401_40181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_part_length_approx_l401_40150

/-- Represents the length of a scale in inches -/
noncomputable def scale_length : ℝ := 12 * 12 + 9

/-- Number of equal parts the scale is divided into -/
def num_parts : ℕ := 7

/-- Number of parts further divided into halves -/
def num_halved_parts : ℕ := 3

/-- Calculates the length of one half of a part -/
noncomputable def half_part_length : ℝ := scale_length / (2 * num_parts)

/-- Theorem stating that the length of one half is approximately 10.93 inches -/
theorem half_part_length_approx :
  ∃ ε > 0, abs (half_part_length - 10.93) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_part_length_approx_l401_40150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_roots_l401_40146

-- Define the function f(x) = x - ln x - 2
noncomputable def f (x : ℝ) := x - Real.log x - 2

-- State the theorem
theorem f_has_roots :
  (∃ x₁ ∈ Set.Ioo 0 1, f x₁ = 0) ∧
  (∃ x₂ ∈ Set.Ioo 3 4, f x₂ = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_roots_l401_40146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_m_bound_l401_40191

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := x^2 + Real.log x - 2*m*x

-- State the theorem
theorem increasing_f_implies_m_bound (m : ℝ) : 
  (∀ x > 0, Monotone (fun x => f x m)) → m ≤ Real.sqrt 2 :=
by
  sorry

-- Additional lemma to help with the proof
lemma derivative_f (x m : ℝ) (h : x > 0) : 
  deriv (fun x => f x m) x = 2*x + 1/x - 2*m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_m_bound_l401_40191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_separation_l401_40151

theorem integral_separation (f g : ℝ → ℝ) (a b : ℝ) :
  ∫ x in a..b, (5 * f x - 2 * g x) = 5 * ∫ x in a..b, f x - 2 * ∫ x in a..b, g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_separation_l401_40151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_with_leak_theorem_l401_40117

/-- The time it takes for Pipe A to fill a tank with a leak present -/
noncomputable def fill_time_with_leak (fill_time_A : ℝ) (empty_time_leak : ℝ) : ℝ :=
  1 / (1 / fill_time_A - 1 / empty_time_leak)

/-- Theorem: Given Pipe A can fill a tank in 4 hours and a leak can empty the full tank in 8 hours,
    it takes 8 hours for Pipe A to fill the tank with the leak present -/
theorem fill_time_with_leak_theorem :
  fill_time_with_leak 4 8 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_with_leak_theorem_l401_40117
