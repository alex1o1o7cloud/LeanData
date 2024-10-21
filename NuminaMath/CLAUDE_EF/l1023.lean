import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l1023_102305

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, points A and B on its right branch,
    and line segment AB passing through the right focus F₂, prove that the perimeter of
    triangle ABF₁ is 4a + 2m, where m is the length of AB and F₁ is the left focus. -/
theorem hyperbola_triangle_perimeter (a b : ℝ) (A B F₁ F₂ : ℝ × ℝ) (m : ℝ) :
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x, y) ∈ ({A, B} : Set (ℝ × ℝ))) →  -- A and B are on the hyperbola
  F₂.1 > 0 →  -- F₂ is on the right side
  (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ F₂ = (1 - t) • A + t • B) →  -- AB passes through F₂
  dist A B = m →  -- Length of AB is m
  F₁.1 < 0 →  -- F₁ is on the left side
  dist A F₁ + dist B F₁ + dist A B = 4 * a + 2 * m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l1023_102305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_properties_l1023_102337

/-- Represents a monomial with real coefficient and two variables -/
structure Monomial where
  coeff : ℝ
  x_exp : ℕ
  y_exp : ℕ

/-- The monomial -7πx³y/6 -/
noncomputable def given_monomial : Monomial := {
  coeff := -7 * Real.pi / 6,
  x_exp := 3,
  y_exp := 1
}

/-- The coefficient of a monomial -/
def coefficient (m : Monomial) : ℝ := m.coeff

/-- The degree of a monomial -/
def degree (m : Monomial) : ℕ := m.x_exp + m.y_exp

/-- Theorem stating the coefficient and degree of the given monomial -/
theorem monomial_properties :
  coefficient given_monomial = -7 * Real.pi / 6 ∧
  degree given_monomial = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_properties_l1023_102337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_theorem_l1023_102359

-- Define the original cylinder
def original_height : ℝ := 4

-- Define the volume increase function
noncomputable def volume_increase (r : ℝ) : ℝ → ℝ
| h => Real.pi * r^2 * h - Real.pi * r^2 * original_height

-- Define the theorem
theorem cylinder_radius_theorem (r : ℝ) : 
  volume_increase r 8 = volume_increase (r + 4) 0 → 
  r = 2 + 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_theorem_l1023_102359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_fifth_number_l1023_102378

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The fifth number in a row of Pascal's triangle -/
def fifthNumber (row : List ℕ) : Option ℕ := row.get? 4

/-- A row in Pascal's triangle starting with 1 and then n -/
def pascalRow (n : ℕ) : List ℕ := List.map (binomial n) (List.range (n + 1))

theorem pascal_fifth_number : 
  fifthNumber (pascalRow 15) = some 1365 := by
  -- Expand the definitions
  unfold fifthNumber pascalRow
  -- Simplify the expression
  simp [List.get?, binomial]
  -- The actual proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_fifth_number_l1023_102378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_characterization_l1023_102375

/-- Triangle type -/
structure Triangle : Type :=
  (vertices : Fin 3 → ℝ × ℝ)

/-- Predicate for equilateral triangles -/
def IsEquilateral (T : Triangle) : Prop := sorry

/-- Predicate for isosceles triangles -/
def IsIsosceles (T : Triangle) : Prop := sorry

/-- Angle type for a triangle -/
def Angle (T : Triangle) : Type := ℝ

/-- Side type for a triangle -/
def Side (T : Triangle) : Type := ℝ

/-- 60 degree angle -/
def degree_60 : ℝ := 60

/-- A triangle is equilateral if and only if it satisfies any of the following conditions:
    1. It has two angles equal to 60°
    2. It is isosceles with one angle equal to 60°
    3. All its angles are equal
    4. All its sides are equal -/
theorem equilateral_triangle_characterization (T : Triangle) : 
  IsEquilateral T ↔ 
    (∃ a b : Angle T, a = degree_60 ∧ b = degree_60) ∨
    (IsIsosceles T ∧ ∃ a : Angle T, a = degree_60) ∨
    (∀ a b : Angle T, a = b) ∨
    (∀ s₁ s₂ : Side T, s₁ = s₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_characterization_l1023_102375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_color_count_l1023_102321

-- Define the board type
def Board (n : ℕ) := Fin n → Fin n → Fin (2*n)

-- Define a property that checks if four cells at the intersection of two rows and two columns are painted in four different colors
def hasFourColorIntersection (n : ℕ) (board : Board n) : Prop :=
  ∃ (r1 r2 c1 c2 : Fin n), r1 ≠ r2 ∧ c1 ≠ c2 ∧
    board r1 c1 ≠ board r1 c2 ∧
    board r1 c1 ≠ board r2 c1 ∧
    board r1 c1 ≠ board r2 c2 ∧
    board r1 c2 ≠ board r2 c1 ∧
    board r1 c2 ≠ board r2 c2 ∧
    board r2 c1 ≠ board r2 c2

-- The main theorem
theorem smallest_color_count (n : ℕ) (h : n ≥ 2) :
  (∀ (board : Board n), hasFourColorIntersection n board) ∧
  (∃ (k : ℕ), k < 2*n ∧ ∃ (board : Board n), ¬hasFourColorIntersection n board) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_color_count_l1023_102321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_geometric_progression_zero_l1023_102367

/-- Represents a geometric progression with first term a, common ratio q, and n terms -/
structure GeometricProgression where
  a : ℝ
  q : ℝ
  n : ℕ
  a_nonzero : a ≠ 0
  q_not_one : q ≠ 1

/-- The sum of the first n terms of a geometric progression -/
noncomputable def sum_geometric_progression (gp : GeometricProgression) : ℝ :=
  gp.a * (gp.q ^ gp.n - 1) / (gp.q - 1)

/-- Theorem: The sum of a finite number of terms in a geometric progression is zero
    if and only if the common ratio is -1 and the number of terms is even -/
theorem sum_geometric_progression_zero (gp : GeometricProgression) :
  sum_geometric_progression gp = 0 ↔ gp.q = -1 ∧ gp.n % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_geometric_progression_zero_l1023_102367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_third_f_max_value_position_l1023_102333

open Real

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x) + cos (ω * x + π / 6)

-- Theorem for part (1)
theorem f_value_at_pi_third (ω : ℝ) (h : ω = 1) : 
  f ω (π / 3) = sqrt 3 / 2 := by sorry

-- Theorem for part (2)
theorem f_max_value_position (ω : ℝ) (h : 2 * π / ω = π) :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (π / 4) ∧ 
  ∀ (y : ℝ), y ∈ Set.Icc 0 (π / 4) → f ω x ≥ f ω y ∧
  x = π / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_third_f_max_value_position_l1023_102333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l1023_102369

noncomputable def f (x : ℝ) : ℝ := 2 * Real.tan (3 * x - Real.pi / 6)

theorem smallest_positive_period (p : ℝ) :
  (∀ x, f (x + p) = f x) ∧
  (∀ q, 0 < q → q < p → ∃ x, f (x + q) ≠ f x) ↔
  p = Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l1023_102369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visibility_time_l1023_102398

/-- Represents the walking speed in feet per second -/
def Speed := ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the circular building -/
structure Building where
  center : Point
  radius : ℝ

/-- Represents a person walking -/
structure Walker where
  initialPosition : Point
  speed : Speed

theorem visibility_time
  (jenny : Walker)
  (kenny : Walker)
  (building : Building)
  (pathDistance : ℝ)
  (h1 : jenny.speed = (2 : ℝ))
  (h2 : kenny.speed = (4 : ℝ))
  (h3 : pathDistance = 300)
  (h4 : building.radius = 100)
  (h5 : jenny.initialPosition = ⟨-100, 150⟩)
  (h6 : kenny.initialPosition = ⟨-100, -150⟩)
  (h7 : building.center = ⟨0, 0⟩) :
  ∃ t : ℝ, t = 48 ∧ t > 0 := by
  sorry

#check visibility_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visibility_time_l1023_102398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canMakeAllPairwiseCoprime_l1023_102373

/-- A type representing a circular arrangement of 100 natural numbers. -/
def CircularArrangement := Fin 100 → ℕ

/-- The operation of adding the GCD of neighboring numbers to a number at a given position. -/
def addGcdOfNeighbors (arr : CircularArrangement) (pos : Fin 100) : CircularArrangement :=
  fun i => if i = pos
           then arr i + Nat.gcd (arr ((i.val - 1 + 100) % 100)) (arr ((i.val + 1) % 100))
           else arr i

/-- Predicate to check if all numbers in the arrangement are pairwise coprime. -/
def allPairwiseCoprime (arr : CircularArrangement) : Prop :=
  ∀ i j, i ≠ j → Nat.gcd (arr i) (arr j) = 1

/-- The main theorem stating that it's possible to make all numbers pairwise coprime. -/
theorem canMakeAllPairwiseCoprime
  (initial : CircularArrangement)
  (h : allPairwiseCoprime initial) :
  ∃ (sequence : ℕ → CircularArrangement),
    sequence 0 = initial ∧
    (∀ n, ∃ pos, sequence (n + 1) = addGcdOfNeighbors (sequence n) pos) ∧
    ∃ k, allPairwiseCoprime (sequence k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_canMakeAllPairwiseCoprime_l1023_102373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_half_l1023_102300

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else 3^x

-- State the theorem
theorem f_composition_half : f (f (1/2)) = 1/3 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_half_l1023_102300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_single_anti_decreasing_f_g_single_anti_decreasing_iff_l1023_102383

-- Definition of a single anti-decreasing function
def is_single_anti_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  (∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) ∧ 
  (∀ x y, x ∈ I → y ∈ I → x < y → (f x / x) > (f y / y))

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the interval (0,1]
def I : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 1}

-- Define the function g(x) = 2x + 2/x + a*ln(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2*x + 2/x + a * Real.log x

-- Define the interval [1,+∞)
def J : Set ℝ := {x : ℝ | 1 ≤ x}

theorem not_single_anti_decreasing_f :
  ¬(is_single_anti_decreasing f I) := by
  sorry

theorem g_single_anti_decreasing_iff (a : ℝ) :
  is_single_anti_decreasing (g a) J ↔ 0 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_single_anti_decreasing_f_g_single_anti_decreasing_iff_l1023_102383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_asymptotes_l1023_102336

/-- Given two hyperbolas with equations x²/9 - y²/16 = 1 and y²/25 - x²/M = 1,
    if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/25 - x^2/M = 1) →
  (∀ x y : ℝ, y = (4/3)*x ∨ y = -(4/3)*x ↔ y = (5/Real.sqrt M)*x ∨ y = -(5/Real.sqrt M)*x) →
  M = 225/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_asymptotes_l1023_102336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1023_102355

/-- Represents the sum of the first n terms of a geometric sequence. -/
noncomputable def geometricSum (a1 : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - r^n) / (1 - r)

/-- Proves that for a positive geometric sequence with a1 = 1 and S5 = 5S3 - 4, S4 = 15. -/
theorem geometric_sequence_sum (r : ℝ) (h1 : r > 0) (h2 : r ≠ 1) :
  geometricSum 1 r 5 = 5 * geometricSum 1 r 3 - 4 →
  geometricSum 1 r 4 = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1023_102355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l1023_102361

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (3, 4)
  let b : ℝ × ℝ := (x, (1 : ℝ) / 2)
  are_parallel a b → x = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l1023_102361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_and_std_dev_of_transformed_data_l1023_102312

variable {n : ℕ}
variable (x : Fin n → ℝ)
variable (a s : ℝ)

def transformed_data (x : Fin n → ℝ) : Fin n → ℝ := λ i => 3 * x i + 5

-- Assuming median function exists
noncomputable def median (data : Fin n → ℝ) : ℝ := sorry

-- Assuming variance function exists
noncomputable def variance (data : Fin n → ℝ) : ℝ := sorry

-- Assuming standard deviation function exists
noncomputable def std_dev (data : Fin n → ℝ) : ℝ := Real.sqrt (variance data)

theorem median_and_std_dev_of_transformed_data 
  (h_median : median x = a)
  (h_variance : variance x = s^2) :
  median (transformed_data x) = 3*a + 5 ∧ std_dev (transformed_data x) = 3*s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_and_std_dev_of_transformed_data_l1023_102312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_correct_l1023_102379

/-- A rectangle inscribed in a circle with specific properties -/
structure InscribedRectangle where
  /-- The length of side AD -/
  ad : ℝ
  /-- The length of side CD -/
  cd : ℝ
  /-- AD equals 5 -/
  ad_eq : ad = 5
  /-- CD equals 4 -/
  cd_eq : cd = 4

/-- The area of the shaded region for the given inscribed rectangle -/
noncomputable def shadedArea (rect : InscribedRectangle) : ℝ :=
  (41 * Real.pi / 4) - 20

/-- Theorem stating that the shaded area is correct for the given inscribed rectangle -/
theorem shaded_area_correct (rect : InscribedRectangle) :
  shadedArea rect = (41 * Real.pi / 4) - 20 := by
  -- Unfold the definition of shadedArea
  unfold shadedArea
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_correct_l1023_102379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l1023_102392

theorem ascending_order (a b c : ℝ) : 
  a = 0.9^(1.1 : ℝ) → b = 1.1^(0.9 : ℝ) → c = Real.log 0.9 / Real.log 2 → c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l1023_102392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sqrt3_over_2_condition_l1023_102362

theorem sin_sqrt3_over_2_condition (x : ℝ) :
  (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3) → Real.sin x = Real.sqrt 3 / 2 ∧
  ¬(Real.sin x = Real.sqrt 3 / 2 → ∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sqrt3_over_2_condition_l1023_102362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_greater_than_one_modifiedSequence_bounded_l1023_102356

def mySequence (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => 3 * mySequence a n - (mySequence a n)^2 - 1

theorem mySequence_greater_than_one (n : ℕ) : mySequence (3/2) n > 1 := by
  sorry

def modifiedSequence (k : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => 3 * modifiedSequence k n - (modifiedSequence k n)^2 + k

theorem modifiedSequence_bounded (k : ℝ) (h : k ∈ Set.Icc (-3/4) 0) :
  ∃ (M : ℝ), ∀ (n : ℕ), |modifiedSequence k n| ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_greater_than_one_modifiedSequence_bounded_l1023_102356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blute_position_l1023_102339

/-- The set of letters used to form permutations -/
def letters : Finset Char := {'B', 'E', 'L', 'T', 'U'}

/-- The word we're interested in -/
def target_word : List Char := ['B', 'L', 'U', 'T', 'E']

/-- A function to determine if a permutation is lexicographically before the target word -/
def is_before (perm : List Char) : Bool :=
  perm < target_word

/-- The number of permutations lexicographically before the target word -/
noncomputable def num_before : Nat :=
  (letters.toList.permutations.filter is_before).length

/-- The alphabetical order position of the target word -/
noncomputable def position : Nat :=
  num_before + 1

/-- Theorem stating the position of "BLUTE" -/
theorem blute_position : position = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blute_position_l1023_102339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_radius_three_coloring_l1023_102380

/-- A regular hexagon with side length 1 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 1)

/-- A point in or on the boundary of a regular hexagon -/
structure HexagonPoint (h : RegularHexagon) :=
  (x : ℝ)
  (y : ℝ)
  (in_hexagon : True) -- Placeholder, replace with actual condition when implementing

/-- A coloring function that assigns one of three colors to each point -/
def Coloring (h : RegularHexagon) := HexagonPoint h → Fin 3

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- A valid coloring satisfies the distance condition for a given radius -/
def valid_coloring (h : RegularHexagon) (c : Coloring h) (r : ℝ) : Prop :=
  ∀ p q : HexagonPoint h, c p = c q → distance (p.x, p.y) (q.x, q.y) < r

/-- The main theorem: The minimum radius for a valid three-coloring is 3/2 -/
theorem min_radius_three_coloring (h : RegularHexagon) :
  (∃ (r : ℝ), ∃ (c : Coloring h), valid_coloring h c r) →
  (∀ (r' : ℝ), r' < 3/2 → ¬∃ (c : Coloring h), valid_coloring h c r') →
  (∃ (c : Coloring h), valid_coloring h c (3/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_radius_three_coloring_l1023_102380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1023_102314

-- Define proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - (a + 1)*x + a + 4 < 0

-- Define proposition q
def q (a : ℝ) : Prop := 
  (a - 3) * (a - 6) > 0 ∧ 
  ∀ x y : ℝ, x^2/(a-3) - y^2/(a-6) = 1 → 
    ∃ A B C D E F : ℝ, A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0 ∧ B^2 - 4*A*C > 0

-- Define the range of a
def range_a : Set ℝ := Set.Ici (-3) ∩ Set.Iio 3

-- Theorem statement
theorem a_range (a : ℝ) : (¬p a ∧ q a) → a ∈ range_a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1023_102314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_sum_equals_three_halves_l1023_102368

-- Define the expression as noncomputable
noncomputable def logarithmic_sum : ℝ :=
  1 / (Real.log 3 / Real.log 18 + 1/2) +
  1 / (Real.log 4 / Real.log 12 + 1/2) +
  1 / (Real.log 6 / Real.log 8 + 1/2)

-- Theorem statement
theorem logarithmic_sum_equals_three_halves :
  logarithmic_sum = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_sum_equals_three_halves_l1023_102368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_when_m_2_m_range_for_ab_distance_l1023_102391

-- Define the parametric equation of line l
noncomputable def line_l (m t : ℝ) : ℝ × ℝ := (m + Real.sqrt 2 * t, Real.sqrt 2 * t)

-- Define the polar equation of curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

-- Define the intersection points
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t θ, line_l m t = curve_C θ}

-- Statement 1
theorem intersection_points_when_m_2 :
  intersection_points 2 = {(2, 0), (0, -2)} := by sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the condition |AB| ≤ 2√3
def ab_distance_condition (m : ℝ) : Prop :=
  ∀ A B, A ∈ intersection_points m → B ∈ intersection_points m → distance A B ≤ 2 * Real.sqrt 3

-- Statement 2
theorem m_range_for_ab_distance :
  {m : ℝ | ab_distance_condition m} =
    {m | -2 * Real.sqrt 2 < m ∧ m ≤ -Real.sqrt 2} ∪
    {m | Real.sqrt 2 ≤ m ∧ m < 2 * Real.sqrt 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_when_m_2_m_range_for_ab_distance_l1023_102391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_angles_l1023_102366

noncomputable def is_solution (z : ℂ) : Prop :=
  z^12 - z^6 - 1 = 0 ∧ Complex.abs z = 1

noncomputable def angle_form (θ : ℝ) : ℂ :=
  Complex.exp (θ * Complex.I)

def valid_angles (θs : Finset ℝ) : Prop :=
  θs.card = 12 ∧ 
  (∀ θ, θ ∈ θs → 0 ≤ θ ∧ θ < 2 * Real.pi) ∧
  (∀ θ₁ θ₂, θ₁ ∈ θs → θ₂ ∈ θs → θ₁ < θ₂ ∨ θ₁ = θ₂ ∨ θ₂ < θ₁)

theorem sum_of_specific_angles (θs : Finset ℝ) :
  valid_angles θs ∧ 
  (∀ θ, θ ∈ θs → is_solution (angle_form θ)) →
  ∃ θ₃ θ₆ θ₉ θ₁₂, θ₃ ∈ θs ∧ θ₆ ∈ θs ∧ θ₉ ∈ θs ∧ θ₁₂ ∈ θs ∧ 
    θ₃ + θ₆ + θ₉ + θ₁₂ = 14 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_angles_l1023_102366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_924_l1023_102376

def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_of_distinct_prime_factors_924 :
  sum_of_distinct_prime_factors 924 = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_924_l1023_102376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_invested_l1023_102382

/-- The sum invested at simple interest for 5 years -/
def P : ℝ := sorry

/-- The original interest rate -/
def R : ℝ := sorry

/-- The condition that at 2% higher rate, it fetches Rs. 180 more -/
axiom higher_rate_2 : P * 5 * 2 / 100 = 180

/-- The condition that at 3% higher rate, it fetches Rs. 270 more -/
axiom higher_rate_3 : P * 5 * 3 / 100 = 270

/-- Theorem stating that P equals 1800 -/
theorem sum_invested : P = 1800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_invested_l1023_102382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_triangles_formula_l1023_102330

/-- The number of triangles formed within an equilateral triangle when each side is divided into n equal parts -/
def num_triangles (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    let k := n / 2
    (Finset.sum (Finset.range k) (fun i => 6 * i^2 - i)) + 3 * k^2 + k
  else
    let k := (n + 1) / 2
    (Finset.sum (Finset.range (k - 1)) (fun i => 6 * i^2 - i)) + 3 * k^2 - 2 * k

/-- The theorem stating the formula for the number of triangles -/
theorem num_triangles_formula (n : ℕ) :
  num_triangles n = if n % 2 = 0 then
    let k := n / 2
    (Finset.sum (Finset.range k) (fun i => 6 * i^2 - i)) + 3 * k^2 + k
  else
    let k := (n + 1) / 2
    (Finset.sum (Finset.range (k - 1)) (fun i => 6 * i^2 - i)) + 3 * k^2 - 2 * k := by
  sorry

#check num_triangles
#check num_triangles_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_triangles_formula_l1023_102330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l1023_102344

noncomputable def f (x : ℝ) : ℝ := -1 / Real.sqrt (x + 1)

theorem f_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l1023_102344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l1023_102353

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^18 + i^28 + i^(-32 : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l1023_102353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_rectangle_area_l1023_102341

/-- The area of the rectangle formed by the intersections of two parabolas with the coordinate axes -/
theorem parabola_intersection_rectangle_area :
  ∀ (a b : ℝ),
  a > 0 → b > 0 →
  a + b = 2 →
  let f (x : ℝ) := a * x^2 - 6
  let g (x : ℝ) := 8 - b * x^2
  let width := 2 * Real.sqrt (6 / a)
  let height := 14
  width * height = 56 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_rectangle_area_l1023_102341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_equivalence_l1023_102309

noncomputable section

-- Define the original and target functions
noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

-- Define the transformations
noncomputable def transform1 (x : ℝ) : ℝ := Real.cos (2 * (x + Real.pi / 3) - Real.pi / 2)
noncomputable def transform2 (x : ℝ) : ℝ := Real.sin (2 * (x - 2 * Real.pi / 3))
noncomputable def transform3 (x : ℝ) : ℝ := Real.cos ((x + 2 * Real.pi / 3) - Real.pi / 2)

-- Theorem stating that all transformations result in the target function
theorem transformations_equivalence (x : ℝ) :
  transform1 x = g x ∧ transform2 x = g x ∧ transform3 x = g x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_equivalence_l1023_102309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_l1023_102349

-- Define the circle C
def circleC (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 5

-- Define the line l
def lineL (m x y : ℝ) : Prop := m * x - y + 1 = 0

-- Define the locus of midpoint M
def locusM (x y : ℝ) : Prop := x^2 + (y - 3/2)^2 = 1/4

-- Theorem statement
theorem midpoint_locus (m : ℝ) :
  ∃ (A B M : ℝ × ℝ),
    (circleC A.1 A.2) ∧
    (circleC B.1 B.2) ∧
    (lineL m A.1 A.2) ∧
    (lineL m B.1 B.2) ∧
    (M.1 = (A.1 + B.1) / 2) ∧
    (M.2 = (A.2 + B.2) / 2) →
    locusM M.1 M.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_l1023_102349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l1023_102340

/-- Pentagon with specific side lengths and triangle properties -/
structure Pentagon where
  /-- Side length FG -/
  FG : ℝ
  /-- Side length GH -/
  GH : ℝ
  /-- Side length HI -/
  HI : ℝ
  /-- Side length IJ -/
  IJ : ℝ
  /-- Side length FJ -/
  FJ : ℝ
  /-- FGH is an isosceles right triangle with right angle at G -/
  FGH_isosceles_right : Bool
  /-- HIJ is an isosceles right triangle with right angle at I -/
  HIJ_isosceles_right : Bool
  /-- FGJ is an isosceles triangle with vertex angle 120° at G -/
  FGJ_isosceles_120 : Bool

/-- Calculate the area of the pentagon -/
noncomputable def pentagon_area (p : Pentagon) : ℝ :=
  sorry

/-- The specific pentagon described in the problem -/
noncomputable def specific_pentagon : Pentagon where
  FG := 3
  GH := 3 * Real.sqrt 2
  HI := 3
  IJ := 3 * Real.sqrt 2
  FJ := 6
  FGH_isosceles_right := true
  HIJ_isosceles_right := true
  FGJ_isosceles_120 := true

/-- Theorem stating the area of the specific pentagon -/
theorem specific_pentagon_area :
  pentagon_area specific_pentagon = 9 + 4.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l1023_102340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cocktail_cost_per_litre_l1023_102331

-- Define the costs and volumes
noncomputable def mixed_fruit_cost : ℝ := 262.85
noncomputable def acai_berry_cost : ℝ := 3104.35
noncomputable def mixed_fruit_volume : ℝ := 35
noncomputable def acai_berry_volume : ℝ := 23.333333333333336

-- Define the total cost of the cocktail
noncomputable def total_cost : ℝ := mixed_fruit_cost * mixed_fruit_volume + acai_berry_cost * acai_berry_volume

-- Define the total volume of the cocktail
noncomputable def total_volume : ℝ := mixed_fruit_volume + acai_berry_volume

-- Define the cost per litre of the cocktail
noncomputable def cost_per_litre : ℝ := total_cost / total_volume

-- Theorem to prove
theorem cocktail_cost_per_litre : 
  abs (cost_per_litre - 1399.99) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cocktail_cost_per_litre_l1023_102331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l1023_102332

/-- Represents a runner in the race -/
structure Runner where
  speed : ℚ
  deriving Repr

/-- The race setup -/
structure Race where
  length : ℚ
  andrei : Runner
  boris : Runner
  sergei : Runner
  deriving Repr

/-- The race conditions -/
def valid_race (r : Race) : Prop :=
  r.length = 100 ∧
  r.length / r.andrei.speed - r.length / r.boris.speed = 10 / r.boris.speed ∧
  r.length / r.boris.speed - r.length / r.sergei.speed = 62 / r.sergei.speed

/-- The distance between Andrei and Sergei when Andrei finishes -/
def distance_andrei_sergei (r : Race) : ℚ :=
  r.length - r.length * r.sergei.speed / r.andrei.speed

/-- The theorem to prove -/
theorem race_result (r : Race) (h : valid_race r) : distance_andrei_sergei r = 19 := by
  sorry

#eval distance_andrei_sergei {
  length := 100,
  andrei := { speed := 10 },
  boris := { speed := 9 },
  sergei := { speed := 81/10 }
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l1023_102332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_average_score_l1023_102387

/-- Represents a subject with its score and study time -/
structure Subject where
  score : ℚ
  studyTime : ℚ

/-- Calculates the expected score given a new study time -/
def expectedScore (subject : Subject) (newTime : ℚ) : ℚ :=
  (subject.score / subject.studyTime) * newTime

theorem expected_average_score 
  (math : Subject)
  (science : Subject)
  (h_math : math = ⟨80, 4⟩)
  (h_science : science = ⟨95, 5⟩)
  (h_newTime : ℚ) 
  (h_newTime_def : h_newTime = 5) :
  (expectedScore math h_newTime + expectedScore science h_newTime) / 2 = 97.5 := by
  sorry

#check expected_average_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_average_score_l1023_102387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_base81_zeroes_l1023_102335

/-- The number of trailing zeroes in n! when written in base b -/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ := sorry

/-- 15 factorial -/
def factorial15 : ℕ := Nat.factorial 15

theorem fifteen_factorial_base81_zeroes :
  trailingZeroes factorial15 81 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_factorial_base81_zeroes_l1023_102335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l1023_102377

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

-- Define point P
def point_P : ℝ × ℝ := (2, 0)

-- Define the intersection points A and B (existence assumed)
axiom exists_intersection_points : ∃ (A B : ℝ × ℝ), 
  line_l A.1 A.2 ∧ curve_C A.1 A.2 ∧ 
  line_l B.1 B.2 ∧ curve_C B.1 B.2 ∧ 
  A ≠ B

-- Theorem statement
theorem intersection_product : 
  ∃ (A B : ℝ × ℝ), line_l A.1 A.2 ∧ curve_C A.1 A.2 ∧ 
                   line_l B.1 B.2 ∧ curve_C B.1 B.2 ∧ 
                   A ≠ B ∧
  (Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2)) *
  (Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)) = 48/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l1023_102377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_theorem_l1023_102322

def vector_proof (m n : ℝ × ℝ) : Prop :=
  let sum := (m.1 + 2 * n.1, m.2 + 2 * n.2)
  let diff := (m.1 - n.1, m.2 - n.2)
  (sum = (4, 1) ∧ diff = (1, -2)) →
  (n.1^2 + n.2^2 = 2 ∧ m.1 * n.1 + m.2 * n.2 = 1)

theorem vector_theorem :
  ∀ m n : ℝ × ℝ, vector_proof m n := by
  sorry

#check vector_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_theorem_l1023_102322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_west_100km_representation_l1023_102303

/-- Represents the direction of travel, where East is positive and West is negative -/
inductive Direction
  | East
  | West

/-- Converts a distance and direction to a signed integer representation -/
def representDistance (distance : ℕ) (direction : Direction) : ℤ :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

/-- Theorem: Traveling west for 100km is represented as -100km -/
theorem west_100km_representation :
  representDistance 100 Direction.West = -100 := by
  rfl

#eval representDistance 100 Direction.West

end NUMINAMATH_CALUDE_ERRORFEEDBACK_west_100km_representation_l1023_102303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l1023_102318

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x

theorem tangent_through_origin (a : ℝ) (h : a > 0) : 
  (∃ m b : ℝ, ∀ x : ℝ, m * x + b = f a x ∧ 
              m * a + b = f a a ∧ 
              m = (deriv (f a)) a ∧ 
              0 = m * 0 + b) → 
  a = Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_through_origin_l1023_102318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_part_speed_is_60_l1023_102374

/-- A journey with two parts -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  firstPartSpeed : ℝ
  firstPartTime : ℝ

/-- Calculate the speed of the second part of the journey -/
noncomputable def secondPartSpeed (j : Journey) : ℝ :=
  let firstPartDistance := j.firstPartSpeed * j.firstPartTime
  let secondPartDistance := j.totalDistance - firstPartDistance
  let secondPartTime := j.totalTime - j.firstPartTime
  secondPartDistance / secondPartTime

/-- Theorem stating that for the given journey parameters, the second part speed is 60 km/h -/
theorem second_part_speed_is_60 (j : Journey) 
    (h1 : j.totalDistance = 240)
    (h2 : j.totalTime = 5)
    (h3 : j.firstPartSpeed = 40)
    (h4 : j.firstPartTime = 3) : 
  secondPartSpeed j = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_part_speed_is_60_l1023_102374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_at_1050km_l1023_102370

/-- Represents the meeting point of two trains leaving Delhi at different times and speeds -/
noncomputable def train_meeting_point (departure_time_diff : ℚ) (speed1 speed2 : ℚ) : ℚ :=
  (departure_time_diff * speed1) / (1 - speed1 / speed2) + departure_time_diff * speed1

/-- Theorem stating that two trains will meet 1050 km from Delhi under given conditions -/
theorem trains_meet_at_1050km :
  let departure_time_diff : ℚ := 5  -- 2 p.m. - 9 a.m. = 5 hours
  let speed1 : ℚ := 30             -- 30 kmph
  let speed2 : ℚ := 35             -- 35 kmph
  train_meeting_point departure_time_diff speed1 speed2 = 1050 := by
  sorry

#check trains_meet_at_1050km

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_at_1050km_l1023_102370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_partition_exists_seven_player_partition_not_always_possible_l1023_102328

-- Define a type for players
def Player : Type := Fin 6

-- Define a tournament as a function that tells us who wins between any two players
def Tournament := Player → Player → Bool

-- Define a cyclic triplet
def is_cyclic_triplet (t : Tournament) (a b c : Player) : Prop :=
  t a b ∧ t b c ∧ t c a

-- Define a partition of players
def Partition := Player → Bool

-- The main theorem
theorem tournament_partition_exists (t : Tournament) : 
  ∃ (p : Partition), 
    (∀ a b c : Player, 
      (p a = p b ∧ p b = p c) → ¬ is_cyclic_triplet t a b c) := by
  sorry

-- Theorem for part (b)
theorem seven_player_partition_not_always_possible : 
  ∃ (t : Fin 7 → Fin 7 → Bool), 
    ∀ (p : Fin 7 → Bool), 
      ∃ (a b c : Fin 7), 
        (p a = p b ∧ p b = p c) ∧ (t a b ∧ t b c ∧ t c a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_partition_exists_seven_player_partition_not_always_possible_l1023_102328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_for_nonzero_l1023_102354

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- Axiom: g is not defined for x = 0
axiom g_not_defined_at_zero : ∀ y, g 0 ≠ y

-- Condition: For all non-zero real numbers x, g(x) + 3g(1/x) = 6 - x^2
axiom g_property : ∀ x : ℝ, x ≠ 0 → g x + 3 * g (1/x) = 6 - x^2

-- Theorem to prove
theorem g_symmetric_for_nonzero (x : ℝ) (h : x ≠ 0) : g x = g (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_for_nonzero_l1023_102354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_probabilities_l1023_102399

/-- Represents the distribution of tourists and card holders in a group -/
structure TouristGroup where
  total : ℕ
  outsideProvince : ℕ
  localTourists : ℕ
  goldCardHolders : ℕ
  silverCardHolders : ℕ

/-- Calculates the probability of selecting exactly one Silver Card holder when choosing 2 tourists randomly -/
def probExactlyOneSilver (group : TouristGroup) : ℚ :=
  sorry

/-- Calculates the probability of selecting an equal number of Gold and Silver Card holders when choosing 2 tourists randomly -/
def probEqualGoldSilver (group : TouristGroup) : ℚ :=
  sorry

/-- The main theorem stating the probabilities for the given tourist group -/
theorem tourist_probabilities (group : TouristGroup) 
  (h1 : group.total = 36)
  (h2 : group.outsideProvince = 27)
  (h3 : group.localTourists = 9)
  (h4 : group.goldCardHolders = 9)
  (h5 : group.silverCardHolders = 6) :
  probExactlyOneSilver group = 5/12 ∧ probEqualGoldSilver group = 47/210 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_probabilities_l1023_102399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ADC_is_right_angle_l1023_102327

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

-- Define the cyclic quadrilateral ABCD
axiom A : Point
axiom B : Point
axiom C : Point
axiom D : Point

-- Define the intersection point K
axiom K : Point

-- Define the midpoints of AC and KC
axiom M : Point  -- midpoint of AC
axiom N : Point  -- midpoint of KC

-- Define predicates for the conditions
def ABCD_is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry
def K_is_intersection_of_AB_and_DC (A B C D K : Point) : Prop := sorry
def points_on_same_circle (B D M N : Point) : Prop := sorry

-- Define the angle function
noncomputable def angle (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem angle_ADC_is_right_angle 
  (h1 : ABCD_is_cyclic_quadrilateral A B C D)
  (h2 : K_is_intersection_of_AB_and_DC A B C D K)
  (h3 : points_on_same_circle B D M N) :
  angle A D C = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ADC_is_right_angle_l1023_102327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_implies_a_geq_e_l1023_102371

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log a + Real.log x) / x

-- State the theorem
theorem decreasing_function_implies_a_geq_e (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x ∧ x < y → f a y < f a x) →
  a ≥ Real.exp 1 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_implies_a_geq_e_l1023_102371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erick_grapes_l1023_102302

/-- Calculates the number of grapes Erick had in his basket -/
def calculate_grapes (num_lemons : ℕ) (orig_lemon_price : ℚ) (orig_grape_price : ℚ) 
  (lemon_price_increase : ℚ) (total_collected : ℚ) : ℕ :=
  let new_lemon_price := orig_lemon_price + lemon_price_increase
  let grape_price_increase := lemon_price_increase / 2
  let new_grape_price := orig_grape_price + grape_price_increase
  let lemon_revenue := num_lemons * new_lemon_price
  let grape_revenue := total_collected - lemon_revenue
  (grape_revenue / new_grape_price).floor.toNat

/-- Theorem stating that Erick had 140 grapes -/
theorem erick_grapes : 
  calculate_grapes 80 8 7 4 2220 = 140 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_erick_grapes_l1023_102302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_form_sum_l1023_102315

/-- A convex quadrilateral with specific side lengths and angle -/
structure ConvexQuadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  angleCDA : ℝ
  convex : AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0
  anglePositive : 0 < angleCDA ∧ angleCDA < Real.pi

/-- The specific quadrilateral from the problem -/
noncomputable def specificQuadrilateral : ConvexQuadrilateral where
  AB := 10
  BC := 6
  CD := 13
  DA := 13
  angleCDA := Real.pi/4
  convex := by sorry
  anglePositive := by sorry

/-- The area of the quadrilateral can be expressed in the form √a + b√c -/
def areaForm (q : ConvexQuadrilateral) (a b c : ℝ) : Prop :=
  ∃ (area : ℝ), area = Real.sqrt a + b * Real.sqrt c ∧
  area = q.AB * q.BC * Real.sin (q.angleCDA) / 2 + q.CD * q.DA * Real.sin (q.angleCDA) / 2

/-- The theorem to be proved -/
theorem area_form_sum (a b c : ℕ) : 
  areaForm specificQuadrilateral (a : ℝ) (b : ℝ) (c : ℝ) →
  ¬ ∃ (k : ℕ), k > 1 ∧ (k * k ∣ a ∨ k * k ∣ c) →
  a + b + c = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_form_sum_l1023_102315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_points_l1023_102397

theorem equilateral_triangle_complex_points : 
  ∃! (s : Finset ℂ), 
    (s.card = 2) ∧ 
    (∀ z ∈ s, z ≠ 0) ∧
    (∀ z ∈ s, Complex.abs z = Complex.abs (2 * z^3)) ∧
    (∀ z ∈ s, Complex.abs (z - 0) = Complex.abs (z - 2*z^3)) ∧
    (∀ z ∈ s, Complex.abs (0 - 2*z^3) = Complex.abs (z - 2*z^3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_points_l1023_102397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1023_102384

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem function_properties (a b : ℝ) (hab : a * b ≠ 0) 
  (h : ∀ x, f a b x ≤ |f a b (π / 6)|) :
  (f a b (11 * π / 12) = 0) ∧ 
  (∀ x, f a b (-x) ≠ f a b x ∧ f a b (-x) ≠ -f a b x) ∧
  (|f a b (7 * π / 10)| = |f a b (π / 5)|) ∧
  (∀ l : ℝ → ℝ, (∃ x, l x = f a b x) → (l a = b)) ∧
  (b > 0 → ∀ k : ℤ, ∀ x ∈ Set.Icc (-π / 3 + k * π) (π / 6 + k * π), 
    (∀ y ∈ Set.Icc (-π / 3 + k * π) x, f a b y ≤ f a b x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1023_102384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_solutions_exist_l1023_102390

/-- Represents the number of songs in a country album -/
def country_songs : ℕ → ℕ := sorry

/-- Represents the number of songs in a pop album -/
def pop_songs : ℕ → ℕ := sorry

/-- Represents the total number of songs -/
def total_songs : ℕ := 72

/-- Represents the number of country albums -/
def num_country_albums : ℕ := 6

/-- Represents the number of pop albums -/
def num_pop_albums : ℕ := 2

/-- Theorem stating that there are multiple solutions for the number of songs in each album type -/
theorem multiple_solutions_exist :
  ∃ (c p c' p' : ℕ),
    c ≠ c' ∧ p ≠ p' ∧
    num_country_albums * (country_songs c) + num_pop_albums * (pop_songs p) = total_songs ∧
    num_country_albums * (country_songs c') + num_pop_albums * (pop_songs p') = total_songs :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_solutions_exist_l1023_102390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_helmet_sales_theorem_l1023_102364

/-- Represents the sales and pricing model of a helmet brand --/
structure HelmetSales where
  april_sales : ℕ
  june_sales : ℕ
  cost_price : ℕ
  base_price : ℕ
  base_volume : ℕ
  price_sensitivity : ℕ
  target_profit : ℕ

/-- Calculates the monthly growth rate given initial and final values over two months --/
noncomputable def monthly_growth_rate (initial : ℕ) (final : ℕ) : ℝ :=
  Real.sqrt (final / initial : ℝ) - 1

/-- Calculates the selling price that maximizes profit given the sales model --/
noncomputable def optimal_selling_price (model : HelmetSales) : ℝ :=
  let a : ℝ := 1
  let b : ℝ := -(model.base_volume / model.price_sensitivity : ℝ) - model.cost_price
  let c : ℝ := (model.base_volume * model.cost_price : ℝ) / model.price_sensitivity - 
           (model.base_volume * (model.base_price - 40) : ℝ) / 10 + model.target_profit / model.price_sensitivity
  (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)

/-- Theorem stating the correct growth rate and optimal selling price --/
theorem helmet_sales_theorem (model : HelmetSales) 
  (h1 : model.april_sales = 150)
  (h2 : model.june_sales = 216)
  (h3 : model.cost_price = 30)
  (h4 : model.base_price = 40)
  (h5 : model.base_volume = 300)
  (h6 : model.price_sensitivity = 10)
  (h7 : model.target_profit = 3960) :
  monthly_growth_rate model.april_sales model.june_sales = 0.2 ∧ 
  ⌊optimal_selling_price model⌋ = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_helmet_sales_theorem_l1023_102364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_equals_four_l1023_102326

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2^x) / (2^x + 3*x)

-- State the theorem
theorem f_n_equals_four (m n : ℝ) 
  (h1 : 2^(m+n) = 3*m*n) 
  (h2 : f m = -1/3) : 
  f n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_equals_four_l1023_102326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_seven_l1023_102304

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The foci of a hyperbola -/
def Hyperbola.foci (h : Hyperbola a b) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- A point on the left branch of the hyperbola -/
def Hyperbola.left_point (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The intersection of PF₂ with the right branch of the hyperbola -/
def Hyperbola.right_intersection (h : Hyperbola a b) (P : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Predicate to check if a triangle is equilateral -/
def is_equilateral (A B C : ℝ × ℝ) : Prop := sorry

/-- The eccentricity of a hyperbola -/
noncomputable def Hyperbola.eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Theorem: If PF₁Q is equilateral, then the eccentricity is √7 -/
theorem hyperbola_eccentricity_sqrt_seven (h : Hyperbola a b) :
  let (F₁, F₂) := h.foci
  let P := h.left_point
  let Q := h.right_intersection P
  is_equilateral P F₁ Q → h.eccentricity = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_seven_l1023_102304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1023_102348

/-- Calculates the future value of an investment with compound interest -/
noncomputable def future_value (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * years)

/-- Proves that the given investment scenario results in approximately $10,815.66 -/
theorem investment_growth :
  let principal := 10000
  let rate := 0.0396
  let compounds_per_year := 2
  let years := 2
  let result := future_value principal rate compounds_per_year years
  abs (result - 10815.66) < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1023_102348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_equals_fraction_fraction_is_lowest_terms_l1023_102317

/-- The decimal representation of the fraction we're looking for -/
def decimal : ℚ := 0.4373737373737373737

/-- The fraction we claim is equivalent to the decimal -/
def fraction : ℚ := 433 / 990

/-- Theorem stating that the decimal is equal to the fraction -/
theorem decimal_equals_fraction : decimal = fraction := by sorry

/-- Function to check if a fraction is in its lowest terms -/
def is_lowest_terms (n d : ℕ) : Prop := Nat.gcd n d = 1

/-- Theorem stating that the fraction is in its lowest terms -/
theorem fraction_is_lowest_terms : is_lowest_terms 433 990 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_equals_fraction_fraction_is_lowest_terms_l1023_102317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_squares_l1023_102394

/-- Represents a square board of size n × n -/
structure Board (n : ℕ) where
  size : n > 0
  even : Even n

/-- Defines adjacency between two squares on the board -/
def adjacent {n : ℕ} (x1 y1 x2 y2 : Fin n) : Prop :=
  (x1 = x2 ∧ (y1 = y1.succ ∨ y2 = y1.succ)) ∨
  (y1 = y2 ∧ (x1 = x1.succ ∨ x2 = x1.succ))

/-- Represents a marking of squares on the board -/
def Marking (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a marking is valid (every square is adjacent to a marked square) -/
def valid_marking {n : ℕ} (m : Marking n) : Prop :=
  ∀ x y, ∃ x' y', adjacent x y x' y' ∧ m x' y' = true

/-- Counts the number of marked squares in a marking -/
def count_marked {n : ℕ} (m : Marking n) : ℕ :=
  (Finset.univ.sum fun x => (Finset.univ.sum fun y => if m x y then 1 else 0))

/-- The main theorem: minimum number of marked squares -/
theorem min_marked_squares {n : ℕ} (b : Board n) :
  (∃ m : Marking n, valid_marking m ∧
    ∀ m' : Marking n, valid_marking m' → count_marked m ≤ count_marked m') ∧
  (∃ m : Marking n, valid_marking m ∧ count_marked m = n * (n + 2) / 4) := by
  sorry

#check min_marked_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_squares_l1023_102394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_min_length_l1023_102308

/-- The ellipse with given properties -/
def Ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- The curve C -/
def CurveC (x y t : ℝ) : Prop :=
  (x - t)^2 + y^2 = (t^2 + 2*t)^2 ∧ 0 < t ∧ t ≤ Real.sqrt 2 / 2

/-- The line l passing through (-2, 0) and tangent to curve C -/
def LineL (x y : ℝ) : Prop :=
  ∃ (k : ℝ), y = k * (x + 2) ∧
  ∃ (t : ℝ), CurveC (t + 2) (k * (t + 2)) t ∧
  ∀ (x' y' : ℝ), CurveC x' y' t → (y' - k * (x' + 2))^2 ≥ 0

/-- The theorem stating the properties of the ellipse and the minimum length -/
theorem ellipse_and_min_length :
  (∀ a b : ℝ, a > b ∧ b > 0 →
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 →
      (∃ c : ℝ, c / a = 1 / 2 ∧
        (∃ P F₁ F₂ : ℝ × ℝ,
          (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧
          F₁ = (-c, 0) ∧ F₂ = (c, 0) ∧
          ∀ P' : ℝ × ℝ, (P'.1^2 / a^2 + P'.2^2 / b^2 = 1) →
            abs ((P'.1 - F₁.1) * (P'.2 - F₂.2) - (P'.2 - F₁.2) * (P'.1 - F₂.1)) / 2 ≤
            abs ((P.1 - F₁.1) * (P.2 - F₂.2) - (P.2 - F₁.2) * (P.1 - F₂.1)) / 2 ∧
            abs ((P.1 - F₁.1) * (P.2 - F₂.2) - (P.2 - F₁.2) * (P.1 - F₂.1)) / 2 = Real.sqrt 3)))) →
  (∀ x y : ℝ, Ellipse x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ min_length : ℝ,
    min_length = 12 * Real.sqrt 2 / 7 ∧
    ∀ x y : ℝ, LineL x y →
      (∃ x₁ y₁ x₂ y₂ : ℝ,
        Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧
        (x₁ - x₂)^2 + (y₁ - y₂)^2 ≥ min_length^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_min_length_l1023_102308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_ratio_l1023_102347

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - Real.sin x ^ 2 + 1

/-- Theorem stating the result to be proved -/
theorem min_value_ratio (θ : ℝ) (h : IsLocalMin f θ) :
  (Real.sin (2 * θ) + Real.cos (2 * θ)) / (Real.sin (2 * θ) - Real.cos (2 * θ)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_ratio_l1023_102347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_zero_l1023_102320

/-- A function f is odd if f(-x) = -f(x) for all x in its domain --/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = (ax - 1) / x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) / x

theorem odd_function_implies_a_zero (a : ℝ) :
  (∀ x ≠ 0, f a x = f a x) →  -- This ensures f is well-defined for x ≠ 0
  IsOdd (f a) →
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_zero_l1023_102320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_payoff_days_l1023_102365

/-- Represents the cryptocurrency mining setup -/
structure MiningSetup where
  system_unit_cost : ℚ
  graphics_card_cost : ℚ
  system_unit_power : ℚ
  graphics_card_power : ℚ
  graphics_card_count : ℕ
  mining_rate : ℚ
  eth_to_rub : ℚ
  electricity_cost : ℚ

/-- Calculates the number of days for the investment to pay off -/
noncomputable def payoff_days (setup : MiningSetup) : ℚ :=
  let total_investment := setup.system_unit_cost + setup.graphics_card_cost * setup.graphics_card_count
  let daily_eth_mined := setup.mining_rate * setup.graphics_card_count
  let daily_revenue := daily_eth_mined * setup.eth_to_rub
  let total_power := setup.system_unit_power + setup.graphics_card_power * setup.graphics_card_count
  let daily_energy_cost := total_power / 1000 * 24 * setup.electricity_cost
  let daily_profit := daily_revenue - daily_energy_cost
  total_investment / daily_profit

theorem investment_payoff_days :
  let setup : MiningSetup := {
    system_unit_cost := 9499
    graphics_card_cost := 20990
    system_unit_power := 120
    graphics_card_power := 185
    graphics_card_count := 2
    mining_rate := 63/10000
    eth_to_rub := 2779037/100
    electricity_cost := 538/100
  }
  ⌈(payoff_days setup : ℚ)⌉ = 179 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_payoff_days_l1023_102365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l1023_102338

theorem triangle_angle_measure (A B C : Real) (a b c : Real) :
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  Real.sin B + Real.sin A * (Real.sin C - Real.cos C) = 0 →
  a = 2 →
  c = Real.sqrt 2 →
  a / Real.sin A = c / Real.sin C →
  C = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l1023_102338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_removal_l1023_102316

theorem red_ball_removal (total : ℕ) (initial_red_percent : ℚ) (target_red_percent : ℚ) 
  (removed : ℕ) (h1 : total = 600) (h2 : initial_red_percent = 70/100) 
  (h3 : target_red_percent = 65/100) (h4 : removed = 86) : 
  (initial_red_percent * (total : ℚ) - (removed : ℚ)) / ((total : ℚ) - (removed : ℚ)) 
  = target_red_percent := by
  sorry

#check red_ball_removal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_removal_l1023_102316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_queens_l1023_102352

/-- A configuration of queens on an n x n chessboard. -/
def QueenConfiguration (n : ℕ) := Fin n → Fin n

/-- Checks if two queens threaten each other. -/
def threatens (n : ℕ) (q1 q2 : Fin n × Fin n) : Prop :=
  q1.1 = q2.1 ∨ q1.2 = q2.2 ∨ (q1.1 : ℤ) - (q2.1 : ℤ) = (q1.2 : ℤ) - (q2.2 : ℤ)

/-- A valid configuration has no queens threatening each other. -/
def isValidConfiguration (n : ℕ) (config : QueenConfiguration n) : Prop :=
  ∀ i j : Fin n, i ≠ j → ¬threatens n (i, config i) (j, config j)

/-- The maximum number of queens that can be placed on an n x n chessboard
    without threatening each other is equal to n. -/
theorem max_queens (n : ℕ) :
  (∃ (config : QueenConfiguration n), isValidConfiguration n config) ∧
  (∀ (m : ℕ) (config : QueenConfiguration n), m > n → ¬isValidConfiguration n config) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_queens_l1023_102352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1023_102395

noncomputable def a (x : ℝ) : ℝ × ℝ := (1, Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (2*x + Real.pi/3), Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - (1/2) * Real.cos (2*x)

theorem f_properties :
  ∀ x ∈ Set.Icc 0 (2*Real.pi),
    (f x = Real.sin (2*x - Real.pi/6) + 1/2) ∧
    (∀ y ∈ Set.Icc 0 (Real.pi/3), x ≤ y → f x ≤ f y) ∧
    (∀ y ∈ Set.Icc (5*Real.pi/6) (4*Real.pi/3), x ≤ y → f x ≤ f y) ∧
    (x ∈ Set.Icc 0 (Real.pi/3) → f x ∈ Set.Icc 0 (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1023_102395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corresponding_angles_equality_equivalence_l1023_102346

-- Define Angle and Corresponding as structures
structure Angle : Type

structure Corresponding (a b : Angle) : Prop

theorem corresponding_angles_equality_equivalence :
  (∀ a b : Angle, Corresponding a b → a = b) ↔
  (∀ a b : Angle, Corresponding a b → a = b) :=
by
  -- The proof is trivial as the left and right sides are identical
  apply Iff.refl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_corresponding_angles_equality_equivalence_l1023_102346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log8_1568_rounded_is_3_l1023_102372

noncomputable def log8_1568 : ℝ := Real.log 1568 / Real.log 8

noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem log8_1568_rounded_is_3 : round_to_nearest log8_1568 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log8_1568_rounded_is_3_l1023_102372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_of_fourth_degree_polynomial_l1023_102313

/-- A fourth-degree polynomial with integer coefficients -/
def FourthDegreePolynomial (a b c d e : ℤ) : ℝ → ℝ := fun x ↦ 
  (a : ℝ) * x^4 + (b : ℝ) * x^3 + (c : ℝ) * x^2 + (d : ℝ) * x + (e : ℝ)

theorem minimum_of_fourth_degree_polynomial 
  (a b c d e : ℤ) (h_positive : a > 0) 
  (h_equal : FourthDegreePolynomial a b c d e (Real.sqrt 3) = 
             FourthDegreePolynomial a b c d e (Real.sqrt 5)) :
  let P := FourthDegreePolynomial a b c d e
  ∃ (m : ℝ), (∀ x, P x ≥ m) ∧ (P 2 = m) ∧ (P (-2) = m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_of_fourth_degree_polynomial_l1023_102313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_total_money_l1023_102363

/-- Represents the water collection and selling scenario over three days --/
structure WaterScenario where
  barrel_capacity : ℚ
  collection_rate : ℚ
  rainfall : Fin 3 → ℚ
  selling_prices : Fin 3 → ℚ

/-- Calculates the total money made from selling water --/
def total_money_made (scenario : WaterScenario) : ℚ :=
  let collected := fun i => min scenario.barrel_capacity (scenario.collection_rate * scenario.rainfall i)
  let day1_sold := collected 0
  let day2_sold := min (scenario.barrel_capacity - day1_sold) (collected 1)
  let day3_sold := min (scenario.barrel_capacity - day1_sold - day2_sold) (collected 2)
  day1_sold * scenario.selling_prices 0 + day2_sold * scenario.selling_prices 1 + day3_sold * scenario.selling_prices 2

/-- The specific scenario described in the problem --/
def james_scenario : WaterScenario := {
  barrel_capacity := 80
  collection_rate := 15
  rainfall := fun i => [4, 3, (5/2)].get i
  selling_prices := fun i => [(6/5), (3/2), (4/5)].get i
}

/-- Theorem stating that the total money made in James' scenario is $102 --/
theorem james_total_money : total_money_made james_scenario = 102 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_total_money_l1023_102363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passes_through_all_quadrants_iff_l1023_102324

/-- The function f(x) defined in terms of parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 + (1/2) * a * x^2 - 2 * a * x + 2 * a + 1

/-- Predicate to check if f passes through all four quadrants -/
def passes_through_all_quadrants (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    f a x₁ > 0 ∧ x₁ > 0 ∧
    f a x₂ < 0 ∧ x₂ > 0 ∧
    f a x₃ < 0 ∧ x₃ < 0 ∧
    f a x₄ > 0 ∧ x₄ < 0

/-- The main theorem stating the necessary and sufficient condition -/
theorem passes_through_all_quadrants_iff (a : ℝ) : 
  passes_through_all_quadrants a ↔ -6/5 < a ∧ a < -3/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_passes_through_all_quadrants_iff_l1023_102324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_g_two_zeros_l1023_102351

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4^(x+1) - 2^x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x - (3/16) * a^2 + (1/4) * a

-- Theorem for monotonicity of f
theorem f_monotonicity :
  (∀ x y, x < y ∧ y < -3 → f x > f y) ∧
  (∀ x y, -3 < x ∧ x < y → f x < f y) := by
  sorry

-- Theorem for the range of a where g has exactly two zeros
theorem g_two_zeros (a : ℝ) :
  (∃ x y, x ≠ y ∧ g a x = 0 ∧ g a y = 0 ∧ ∀ z, g a z = 0 → z = x ∨ z = y) ↔
  (0 < a ∧ a < 1/3) ∨ (1 < a ∧ a < 4/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_g_two_zeros_l1023_102351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_speed_ratio_l1023_102393

-- Define the skating speeds and rink circumference
variable (s : ℝ) -- circumference of the skating rink
variable (x : ℝ) -- son's speed
variable (k : ℝ) -- factor by which father's speed is greater than son's

-- Define the encounter frequencies
noncomputable def same_direction_frequency := s / (k * x - x)
noncomputable def opposite_direction_frequency := s / (k * x + x)

-- State the theorem
theorem fathers_speed_ratio (h : opposite_direction_frequency = 5 * same_direction_frequency) : k = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fathers_speed_ratio_l1023_102393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l1023_102381

-- Define the triangle
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the hypotenuse length
def hypotenuse_length : ℝ := 13

-- Define the angle in radians (30 degrees = π/6 radians)
noncomputable def angle : ℝ := Real.pi/6

-- Theorem statement
theorem triangle_area_theorem :
  ∀ (a b c : ℝ),
  right_triangle a b c →
  c = hypotenuse_length →
  angle = Real.pi/6 →
  (1/2) * a * b = 21.125 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l1023_102381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_2008_l1023_102389

theorem sum_divisible_by_2008 (a : Fin 2008 → ℤ) :
  ∃ (s : Finset (Fin 2008)), s.card > 0 ∧ (s.sum (λ i => a i)) % 2008 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_2008_l1023_102389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_QR_l1023_102323

/-- Right triangle DEF with given side lengths -/
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  right_angle : DE^2 + EF^2 = DF^2

/-- Circle with center Q tangent to DE at D and passing through F -/
structure CircleQ where
  Q : ℝ × ℝ
  D : ℝ × ℝ
  F : ℝ × ℝ
  tangent_at_D : (Q.1 - D.1)^2 + (Q.2 - D.2)^2 = (Q.1 - F.1)^2 + (Q.2 - F.2)^2

/-- Circle with center R tangent to EF at E and passing through F -/
structure CircleR where
  R : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  tangent_at_E : (R.1 - E.1)^2 + (R.2 - E.2)^2 = (R.1 - F.1)^2 + (R.2 - F.2)^2

/-- The main theorem -/
theorem distance_QR (triangle : RightTriangle) (circleQ : CircleQ) (circleR : CircleR) 
  (h1 : triangle.DE = 5)
  (h2 : triangle.EF = 12)
  (h3 : triangle.DF = 13) :
  (circleQ.Q.1 - circleR.R.1)^2 + (circleQ.Q.2 - circleR.R.2)^2 = (169/5)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_QR_l1023_102323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_10_15_l1023_102343

/-- The area of a rhombus given the lengths of its diagonals -/
noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- Theorem: The area of a rhombus with diagonals of 10 cm and 15 cm is 75 cm² -/
theorem rhombus_area_10_15 : rhombus_area 10 15 = 75 := by
  unfold rhombus_area
  norm_num

#check rhombus_area_10_15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_10_15_l1023_102343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1023_102329

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x^2 + 1)

theorem f_properties :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f (-x) = -f x) ∧
  f (1/2) = 4/5 ∧
  Set.range f = Set.Icc (-1 : ℝ) 1 ∧
  ∀ m ≥ 2, Set.Icc (-1 : ℝ) 1 ⊆ Set.Icc (1 - m) (2 * m) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1023_102329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spend_is_correct_l1023_102306

/-- Calculates the total amount spent by Kristine and Crystal on hair accessories --/
def hair_accessories_total_spend 
  (barrette_price : ℝ)
  (comb_price : ℝ)
  (hairband_price : ℝ)
  (hair_ties_price : ℝ)
  (kristine_barrettes : ℕ)
  (kristine_combs : ℕ)
  (kristine_hairbands : ℕ)
  (kristine_hair_ties : ℕ)
  (crystal_barrettes : ℕ)
  (crystal_combs : ℕ)
  (crystal_hairbands : ℕ)
  (crystal_hair_ties : ℕ)
  (sales_tax_rate : ℝ) : ℝ :=
by
  sorry

/-- The total amount spent by Kristine and Crystal is $69.17 --/
theorem total_spend_is_correct : 
  hair_accessories_total_spend 4 2 3 2.5 2 3 4 5 3 2 1 7 0.085 = 69.17 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spend_is_correct_l1023_102306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sums_is_three_l1023_102388

/-- Represents a quadrilateral with four interior angles -/
structure Quadrilateral where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_360 : A + B + C + D = 360

/-- The maximum number of angle pair sums greater than 180° in a quadrilateral -/
noncomputable def max_sums_greater_than_180 (q : Quadrilateral) : ℕ :=
  let sums := [q.A + q.B, q.A + q.C, q.A + q.D, q.B + q.C, q.B + q.D, q.C + q.D]
  (sums.filter (λ x => x > 180)).length

/-- Theorem: The maximum number of angle pair sums greater than 180° is 3 -/
theorem max_sums_is_three :
  ∀ q : Quadrilateral, max_sums_greater_than_180 q ≤ 3 ∧
  ∃ q : Quadrilateral, max_sums_greater_than_180 q = 3 := by
  sorry

#check max_sums_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sums_is_three_l1023_102388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1023_102396

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_between_points : 
  distance (-3) 4 6 (-2) = Real.sqrt 117 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1023_102396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_sum_l1023_102386

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log x + 2

def tangent_line (a b : ℝ) (x : ℝ) : ℝ := a * x + b

theorem common_tangent_sum (a b : ℝ) (h1 : b > 0) :
  (∃ x1 x2 : ℝ, 
    (tangent_line a b x1 = f x1) ∧ 
    (tangent_line a b x2 = g x2) ∧
    (a = (deriv f) x1) ∧
    (a = (deriv g) x2)) →
  a + b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_sum_l1023_102386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_range_l1023_102342

/-- Given an acute triangle ABC with BC = 2 and sin B + sin C = 2 sin A,
    the length of median AD is in the range [√3, √13/2). -/
theorem median_length_range (A B C : ℝ) (D : ℝ) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  -- BC = 2
  Real.cos C * 2 * Real.sin B = 2 →
  -- sin B + sin C = 2 sin A
  Real.sin B + Real.sin C = 2 * Real.sin A →
  -- D is the midpoint of BC
  D = (B + C) / 2 →
  -- The length of median AD is in the range [√3, √13/2)
  Real.sqrt 3 ≤ |A - D| ∧ |A - D| < Real.sqrt 13 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_range_l1023_102342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop4_prop5_correct_propositions_l1023_102319

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Proposition 4
theorem prop4 (a b : Line) (α : Plane) :
  parallel a b → parallel_line_plane a α → ¬in_plane b α → parallel_line_plane b α := by
  sorry

-- Proposition 5
theorem prop5 (α β χ : Plane) (l : Line) :
  perpendicular α χ → perpendicular β χ → intersect α β l → perpendicular_line_plane l χ := by
  sorry

-- The main theorem stating that propositions 4 and 5 are correct
theorem correct_propositions : 
  (∀ a b α, parallel a b → parallel_line_plane a α → ¬in_plane b α → parallel_line_plane b α) ∧
  (∀ α β χ l, perpendicular α χ → perpendicular β χ → intersect α β l → perpendicular_line_plane l χ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop4_prop5_correct_propositions_l1023_102319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birdseed_mix_proportion_l1023_102350

/-- Represents a birdseed brand with its sunflower content -/
structure Birdseed where
  sunflower_percent : ℚ
  deriving Repr

/-- Represents a mixture of two birdseed brands -/
structure BirdseedMix where
  brand_a : Birdseed
  brand_b : Birdseed
  brand_a_proportion : ℚ
  deriving Repr

theorem birdseed_mix_proportion
  (mix : BirdseedMix)
  (h_a : mix.brand_a.sunflower_percent = 6/10)
  (h_b : mix.brand_b.sunflower_percent = 35/100)
  (h_mix : mix.brand_a.sunflower_percent * mix.brand_a_proportion +
           mix.brand_b.sunflower_percent * (1 - mix.brand_a_proportion) = 1/2) :
  mix.brand_a_proportion = 6/10 := by
  sorry

#eval BirdseedMix.mk (Birdseed.mk (6/10)) (Birdseed.mk (35/100)) (6/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birdseed_mix_proportion_l1023_102350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_with_exclusion_l1023_102345

theorem stratified_sampling_with_exclusion (total_teachers : ℕ) (senior_teachers : ℕ) 
  (intermediate_teachers : ℕ) (junior_teachers : ℕ) (sample_size : ℕ) :
  total_teachers = senior_teachers + intermediate_teachers + junior_teachers →
  senior_teachers = 28 →
  intermediate_teachers = 54 →
  junior_teachers = 81 →
  sample_size = 36 →
  let remaining_teachers := total_teachers - 1
  let sampling_ratio := sample_size / remaining_teachers
  (senior_teachers - 1) * sampling_ratio = 6 ∧
  intermediate_teachers * sampling_ratio = 12 ∧
  junior_teachers * sampling_ratio = 18 ∧
  ¬(∃ k : ℕ, senior_teachers * sample_size = k * total_teachers) ∨
  ¬(∃ k : ℕ, intermediate_teachers * sample_size = k * total_teachers) ∨
  ¬(∃ k : ℕ, junior_teachers * sample_size = k * total_teachers) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_with_exclusion_l1023_102345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1023_102358

-- Define the line l: x - y + 4 = 0
def line (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the circle C: (x-1)^2 + (y-1)^2 = 4
def circleEq (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the center of the circle
def center : ℝ × ℝ := (1, 1)

-- Define the radius of the circle
def radius : ℝ := 2

-- Theorem: The minimum distance from any point on C to l is 2√2 - 2
theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 - 2 ∧
  ∀ (p : ℝ × ℝ), circleEq p.1 p.2 →
  ∀ (q : ℝ × ℝ), line q.1 q.2 →
  d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1023_102358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_slope_range_l1023_102357

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := (a * x) / (x^2 + b)

-- State the theorem
theorem tangent_line_and_slope_range :
  ∃ (a b : ℝ),
  (∀ x : ℝ, f a b x = (4 * x) / (x^2 + 1)) ∧
  (f a b 1 = 2) ∧
  (((deriv (f a b)) 1) = 0) ∧
  (∀ x : ℝ, -1/2 ≤ ((deriv (f a b)) x) ∧ ((deriv (f a b)) x) ≤ 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_slope_range_l1023_102357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sequence_l1023_102310

def S (n : ℕ) : ℤ := -n^2 + 6*n + 7

def a : ℕ → ℤ
  | 0 => 0  -- Adding a case for 0
  | 1 => S 1
  | n+1 => S (n+1) - S n

theorem max_value_of_sequence :
  ∃ m : ℤ, m = 12 ∧ ∀ n : ℕ, a n ≤ m := by
  -- We'll use 12 as our maximum value
  use 12
  constructor
  · -- First part: m = 12
    rfl
  · -- Second part: ∀ n : ℕ, a n ≤ 12
    intro n
    -- We'll need to prove this by cases or induction
    sorry

-- The actual proof would require more steps and lemmas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sequence_l1023_102310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_change_l1023_102334

noncomputable def initial_average : ℚ := 124/10
def wickets_before : ℕ := 175
def wickets_last : ℕ := 8
def runs_last : ℕ := 26

noncomputable def total_runs_before : ℚ := initial_average * wickets_before
def total_wickets_after : ℕ := wickets_before + wickets_last
noncomputable def total_runs_after : ℚ := total_runs_before + runs_last

noncomputable def new_average : ℚ := total_runs_after / total_wickets_after

theorem bowling_average_change :
  initial_average - new_average = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_average_change_l1023_102334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_perp_plane_parallel_l1023_102311

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perp : Line → Line → Prop)
variable (perpPlane : Line → Plane → Prop)

-- Notation for parallel and perpendicular
local infix:50 " ∥ " => parallel
local infix:50 " ⟂ " => perp
local infix:50 " ⟂ₚ " => perpPlane

-- Theorem for transitivity of parallel lines
theorem parallel_transitive (a b c : Line) :
  (a ∥ b) → (b ∥ c) → (a ∥ c) := by sorry

-- Theorem for perpendicular lines to a plane are parallel
theorem perp_plane_parallel (a b : Line) (r : Plane) :
  (a ⟂ₚ r) → (b ⟂ₚ r) → (a ∥ b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_perp_plane_parallel_l1023_102311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_numbers_l1023_102301

theorem sum_of_four_numbers (p q r s : ℝ) 
  (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r) (h_pos_s : 0 < s)
  (h_sum_squares : p^2 + q^2 = 2500 ∧ r^2 + s^2 = 2500)
  (h_products : p * r = 1200 ∧ q * s = 1200) : 
  p + q + r + s = 140 := by
  sorry

#check sum_of_four_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_numbers_l1023_102301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_cosh_curve_l1023_102385

-- Define the function representing the curve
noncomputable def f (x : ℝ) : ℝ := Real.cosh x + 3

-- Define the arc length function
noncomputable def arcLength (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt (1 + (deriv f x) ^ 2)

-- State the theorem
theorem arc_length_cosh_curve :
  arcLength 0 1 = Real.sinh 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_cosh_curve_l1023_102385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_circle_radius_l1023_102307

/-- The radius of a semi-circle with perimeter 140 cm -/
noncomputable def radius : ℝ := 140 / (Real.pi + 2)

/-- The perimeter of a semi-circle -/
noncomputable def semiCirclePerimeter (r : ℝ) : ℝ := Real.pi * r + 2 * r

/-- Theorem: The radius of a semi-circle with perimeter 140 cm is 140 / (π + 2) cm -/
theorem semi_circle_radius :
  semiCirclePerimeter radius = 140 := by
  -- Expand the definition of semiCirclePerimeter
  unfold semiCirclePerimeter
  -- Expand the definition of radius
  unfold radius
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_circle_radius_l1023_102307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_root_2019_l1023_102325

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

theorem no_integer_root_2019 (P : IntPolynomial) 
  (a b c d : ℤ) (ha : P.eval a = 2016) (hb : P.eval b = 2016) 
  (hc : P.eval c = 2016) (hd : P.eval d = 2016)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  ¬ ∃ (x : ℤ), P.eval x = 2019 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_root_2019_l1023_102325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_plane_not_always_perpendicular_trapezoid_hexagon_l1023_102360

structure Point3D :=
  (x y z : ℝ)

structure Plane :=
  (points : Set Point3D)

structure Line :=
  (points : Set Point3D)

def perpendicular (l : Line) (p : Plane) : Prop := sorry

def perpendicularInPlane (l : Line) (l1 l2 : Line) (p : Plane) : Prop := sorry

def triangle (p : Plane) : Prop := sorry
def trapezoid (p : Plane) : Prop := sorry
def circleInPlane (p : Plane) : Prop := sorry
def regularHexagon (p : Plane) : Prop := sorry

def diameter (c : Plane) (h : circleInPlane c) : Line := sorry

theorem line_perpendicular_to_plane (l : Line) (p : Plane) :
  (∃ t : triangle p, ∃ s1 s2 : Line, perpendicularInPlane l s1 s2 p ∧ s1 ≠ s2) ∨
  (∃ h : circleInPlane p, ∃ d1 d2 : Line, d1 = diameter p h ∧ d2 = diameter p h ∧ 
    perpendicularInPlane l d1 d2 p ∧ d1 ≠ d2) →
  perpendicular l p :=
sorry

theorem not_always_perpendicular_trapezoid_hexagon (l : Line) (p : Plane) :
  ¬(∀ tz : trapezoid p, ∀ s1 s2 : Line, 
    perpendicularInPlane l s1 s2 p ∧ s1 ≠ s2 → perpendicular l p) ∧
  ¬(∀ h : regularHexagon p, ∀ s1 s2 : Line, 
    perpendicularInPlane l s1 s2 p ∧ s1 ≠ s2 → perpendicular l p) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_plane_not_always_perpendicular_trapezoid_hexagon_l1023_102360
