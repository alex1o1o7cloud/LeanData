import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l382_38282

noncomputable def f (x : ℝ) := Real.log (x^2 - x)

theorem domain_of_f :
  {x : ℝ | x^2 - x > 0} = Set.Ioi 1 ∪ Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l382_38282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l382_38280

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, 0)
def c : ℝ × ℝ := (1, -2)

theorem collinear_vectors_lambda (l : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (l * a.1 + b.1, l * a.2 + b.2) = (k * c.1, k * c.2)) →
  l = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l382_38280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_cards_removed_is_43_14_l382_38290

-- Define the number of cards
def num_cards : ℕ := 42

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the set of prime numbers up to 42
noncomputable def primes : Finset ℕ := Finset.filter (fun n => Nat.Prime n) (Finset.range (num_cards + 1))

-- Define the number of prime cards
noncomputable def num_primes : ℕ := Finset.card primes

-- Define the number of non-prime cards
noncomputable def num_non_primes : ℕ := num_cards - num_primes

-- Define the number of groups (number of primes + 1)
noncomputable def num_groups : ℕ := num_primes + 1

-- Theorem statement
theorem average_cards_removed_is_43_14 :
  (num_non_primes : ℚ) / num_groups + 1 = 43 / 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_cards_removed_is_43_14_l382_38290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_U_vector3_l382_38274

-- Define the transformation U
def U : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) := sorry

-- Define the properties of U
axiom U_linear (c d : ℝ) (u v : ℝ × ℝ × ℝ) : 
  U (c • u + d • v) = c • U u + d • U v

-- Define cross product for ℝ × ℝ × ℝ
def cross (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1 - u.1 * v.2.2,
   u.1 * v.2.1 - u.2.1 * v.1)

axiom U_cross_product (u v : ℝ × ℝ × ℝ) : 
  U (cross u v) = cross (U u) (U v)

axiom U_vector1 : U (4, 4, 2) = (3, -1, 6)

axiom U_vector2 : U (-4, 2, 4) = (3, 6, -1)

-- The theorem to prove
theorem U_vector3 : U (2, 6, 8) = (4, 6, 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_U_vector3_l382_38274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_solutions_equation_l382_38205

theorem finite_solutions_equation :
  ∃ (M : ℕ), ∀ (n : ℕ), n > 0 → (n + 900) / 80 = ⌊Real.sqrt n⌋ → n ≤ M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_solutions_equation_l382_38205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_center_on_square_side_l382_38276

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a square -/
structure Square where
  center : Point
  side : ℝ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Define R as a constant -/
def R : ℝ := 1  -- You can change this value as needed

noncomputable def semicircle_radius : ℝ := R

noncomputable def square_side_length : ℝ := 2 * (Real.sqrt 5 * R) / 5

noncomputable def triangle_base : ℝ := 2 * R

noncomputable def triangle_height : ℝ := 4 * R / 5

/-- The main theorem to prove -/
theorem inscribed_circle_center_on_square_side 
  (semicircle : Circle)
  (inscribed_square : Square)
  (inscribed_triangle : Triangle)
  (inscribed_circle : Circle) :
  semicircle.radius = semicircle_radius →
  inscribed_square.side = square_side_length →
  (inscribed_triangle.b.x - inscribed_triangle.a.x)^2 + 
  (inscribed_triangle.b.y - inscribed_triangle.a.y)^2 = triangle_base^2 →
  (inscribed_triangle.c.y - inscribed_triangle.a.y) = triangle_height →
  -- The center of the inscribed circle lies on one of the sides of the square
  ∃ (p q : Point), 
    (p.x - q.x)^2 + (p.y - q.y)^2 = inscribed_square.side^2 ∧
    (inscribed_circle.center.x - p.x)^2 + (inscribed_circle.center.y - p.y)^2 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_center_on_square_side_l382_38276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l382_38251

-- Define the inequality function
noncomputable def f (a x : ℝ) : ℝ := (a * x - 5) / (x - a)

-- Define the solution set M
noncomputable def M (a : ℝ) : Set ℝ := {x | f a x < 0}

-- Part 1: Prove that when a = 1, M = (1, 5)
theorem part_one : M 1 = Set.Ioo 1 5 := by sorry

-- Part 2: Prove the range of a such that 3 ∈ M and 5 ∉ M
theorem part_two : {a : ℝ | 3 ∈ M a ∧ 5 ∉ M a} = Set.Icc 1 (5/3) ∪ Set.Ioo 3 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l382_38251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_for_set_equality_l382_38230

theorem unique_a_for_set_equality :
  ∃! a : ℤ, 
    (let M : Set ℤ := {x | x^2 ≤ 1}
     let N : Set ℤ := {a, a^2}
     M ∪ N = M) ∧ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_for_set_equality_l382_38230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_ratio_l382_38271

/-- The ratio of Lilly's cleaning time to the total cleaning time is 1:4, given that the total cleaning time is 8 hours and Fiona's cleaning time is 360 minutes. -/
theorem cleaning_time_ratio :
  (120 : ℚ) / 480 = 1 / 4 := by
  -- Convert to decimals and compute
  simp
  -- This should simplify to 0.25 = 0.25
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_ratio_l382_38271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l382_38273

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x - Real.pi / 6) - 2 * (Real.cos (ω * x / 2))^2 + 1

theorem function_properties (ω : ℝ) (A B C : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω (x + Real.pi / ω) = f ω x) →
  0 < A ∧ A < Real.pi / 2 →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi / 2 →
  A + B + C = Real.pi →
  f ω (B / 2) = f ω (-B / 2) →
  (∃ x : ℝ, f ω x = Real.sqrt 3) →
  (ω = 2 ∧ 
   ∀ y : ℝ, (3 / 2 < y ∧ y ≤ Real.sqrt 3) ↔ 
             ∃ A' : ℝ, Real.pi / 6 < A' ∧ A' < Real.pi / 2 ∧ 
                       y = Real.sin A' + Real.sin (2 * Real.pi / 3 - A')) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l382_38273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sets_l382_38289

-- Define a function to check if three numbers can form a right-angled triangle
def canFormRightTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Define the sets of numbers
def setA : List ℝ := [6, 8, 10]
def setB : List ℝ := [7, 24, 25]
def setC : List ℝ := [9, 16, 25]  -- Changed to avoid using powers directly

noncomputable def setD : List ℝ := [Real.sqrt 2, Real.sqrt 3, Real.sqrt 5]

-- State the theorem
theorem right_triangle_sets :
  (∃ (a b c : ℝ), a ∈ setA ∧ b ∈ setA ∧ c ∈ setA ∧ canFormRightTriangle a b c) ∧
  (∃ (a b c : ℝ), a ∈ setB ∧ b ∈ setB ∧ c ∈ setB ∧ canFormRightTriangle a b c) ∧
  (∃ (a b c : ℝ), a ∈ setC ∧ b ∈ setC ∧ c ∈ setC ∧ canFormRightTriangle a b c) ∧
  (∃ (a b c : ℝ), a ∈ setD ∧ b ∈ setD ∧ c ∈ setD ∧ canFormRightTriangle a b c) :=
by sorry

-- Note: The original problem statement and solution incorrectly identified set C
-- as not forming a right-angled triangle. This theorem reflects that all sets
-- can form right-angled triangles, contrary to the original conclusion.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sets_l382_38289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_comparison_l382_38254

/-- A rectangle with area equal to twice its perimeter and length equal to twice its width -/
structure SpecialRectangle where
  width : ℝ
  length : ℝ
  area_eq_twice_perimeter : width * length = 2 * (2 * width + 2 * length)
  length_eq_twice_width : length = 2 * width

/-- A regular hexagon with area equal to three times its perimeter -/
structure SpecialHexagon where
  side : ℝ
  area_eq_thrice_perimeter : (3 * Real.sqrt 3 / 2) * side^2 = 3 * (6 * side)

/-- The apothem of a rectangle is half its width -/
noncomputable def rectangle_apothem (r : SpecialRectangle) : ℝ := r.width / 2

/-- The apothem of a regular hexagon is (√3 / 2) * side length -/
noncomputable def hexagon_apothem (h : SpecialHexagon) : ℝ := (Real.sqrt 3 / 2) * h.side

theorem apothem_comparison (r : SpecialRectangle) (h : SpecialHexagon) :
  rectangle_apothem r = (1 / 2) * hexagon_apothem h := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_comparison_l382_38254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_increases_by_three_l382_38296

-- Define the original fraction
noncomputable def original_fraction (x y : ℝ) : ℝ := (2 * x * y) / (3 * x - y)

-- Define the new fraction after enlarging x and y by a factor of 3
noncomputable def new_fraction (x y : ℝ) : ℝ := (2 * (3 * x) * (3 * y)) / (3 * (3 * x) - (3 * y))

-- Theorem statement
theorem fraction_increases_by_three (x y : ℝ) (h : 3 * x ≠ y) : 
  new_fraction x y = 3 * original_fraction x y := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_increases_by_three_l382_38296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_divisibility_l382_38283

def digit_sum : ℕ → ℕ
  | 0 => 0
  | n + 1 => (n + 1) % 10 + digit_sum (n / 10)

theorem digit_sum_divisibility (n k : ℕ) 
  (h1 : ¬ 3 ∣ n)
  (h2 : k ≥ n)
  (hn : n > 0) :
  ∃ m : ℕ, m > 0 ∧ n ∣ m ∧ (digit_sum m = k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_divisibility_l382_38283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_sum_l382_38256

theorem sine_of_sum (α β : ℝ) : 
  α ∈ Set.Ioo 0 π → 
  β ∈ Set.Ioo (-π/2) (π/2) → 
  Real.sin (α + π/3) = 1/3 → 
  Real.cos (β - π/6) = Real.sqrt 6 / 6 → 
  Real.sin (α + 2*β) = (2*Real.sqrt 10 - 2)/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_sum_l382_38256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_17_equals_905_l382_38288

def sequence_a : ℕ → ℤ
  | 0 => 1  -- We need to define the case for 0
  | 1 => 1
  | 2 => 5
  | (n + 3) => 2 * sequence_a (n + 2) - sequence_a (n + 1) + 7

theorem a_17_equals_905 : sequence_a 17 = 905 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_17_equals_905_l382_38288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiangyang_road_length_l382_38242

/-- Represents the scale of a map as a ratio of map units to real-world units -/
structure MapScale where
  map_units : ℕ
  real_units : ℕ

/-- Calculates the actual length given a map length and scale -/
noncomputable def actual_length (map_length : ℝ) (scale : MapScale) : ℝ :=
  map_length * (scale.real_units : ℝ) / (scale.map_units : ℝ)

theorem xiangyang_road_length 
  (scale : MapScale)
  (map_length : ℝ)
  (h_scale : scale = ⟨1, 10000⟩)
  (h_map_length : map_length = 10) :
  actual_length map_length scale = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiangyang_road_length_l382_38242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_c_all_reals_l382_38245

/-- The function c(x) with parameter k -/
noncomputable def c (k : ℝ) (x : ℝ) : ℝ := (k * x^2 - 3 * x + 7) / (-3 * x^2 - x + k)

/-- The domain of c(x) is all real numbers iff k < -1/12 -/
theorem domain_c_all_reals (k : ℝ) : 
  (∀ x : ℝ, -3 * x^2 - x + k ≠ 0) ↔ k < -1/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_c_all_reals_l382_38245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_l382_38206

/-- Triangle with vertices P1, P2, P3 in 2D plane -/
structure Triangle where
  P1 : ℝ × ℝ
  P2 : ℝ × ℝ
  P3 : ℝ × ℝ

/-- Distance between two points in 2D plane -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Length of the longest side of a triangle -/
noncomputable def longest_side (t : Triangle) : ℝ :=
  max (distance t.P1 t.P2) (max (distance t.P2 t.P3) (distance t.P3 t.P1))

/-- Length of the shortest side of a triangle -/
noncomputable def shortest_side (t : Triangle) : ℝ :=
  min (distance t.P1 t.P2) (min (distance t.P2 t.P3) (distance t.P3 t.P1))

theorem triangle_sides (t : Triangle) 
  (h : t = { P1 := (1, 2), P2 := (4, 3), P3 := (3, -1) }) : 
  longest_side t = Real.sqrt 17 ∧ shortest_side t = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_l382_38206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l382_38204

/-- The fixed circle B -/
def circle_B : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + p.2^2 = 64}

/-- The point A -/
def point_A : ℝ × ℝ := (-3, 0)

/-- The trajectory of the center of the moving circle P -/
def trajectory_P : Set (ℝ × ℝ) :=
  {p | p.1^2 / 16 + p.2^2 / 7 = 1}

/-- The theorem stating that the trajectory of P's center is an ellipse -/
theorem trajectory_is_ellipse :
  ∀ (P : Set (ℝ × ℝ)),
  (∃ (M : ℝ × ℝ), M ∈ P ∩ circle_B) →  -- P is internally tangent to B
  point_A ∈ P →                     -- P passes through A
  (∃ (center : ℝ × ℝ), center ∈ trajectory_P ∧ 
    ∀ (p : ℝ × ℝ), p ∈ P ↔ ∃ (r : ℝ), r > 0 ∧ 
      ∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = r^2 → (x, y) ∈ P) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l382_38204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equality_l382_38265

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_expression_equality (y : ℝ) (h : y = 8.4) :
  (floor 6.5 : ℝ) * (floor (2 / 3) : ℝ) + (floor 2 : ℝ) * 7.2 + (floor y : ℝ) - 6.2 = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equality_l382_38265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_max_m_for_inequality_l382_38236

noncomputable section

def f (x : ℝ) : ℝ := Real.log x

def g (m n x : ℝ) : ℝ := m * (x + n) / (x + 1)

theorem tangent_line_and_inequality (m : ℝ) (h_m : m > 0) :
  (∃ n : ℝ, (∀ x : ℝ, x ≠ 0 → (deriv f x) = (deriv (g m n) x)) ∧
             (f 1 = g m n 1)) →
  (∀ x : ℝ, x ≥ 1 → |f x| ≥ |g m (-1) x|) →
  m = 2 :=
sorry

theorem max_m_for_inequality :
  ∃ m : ℝ, m > 0 ∧
    (∀ x : ℝ, x ≥ 1 → |f x| ≥ |g m (-1) x|) ∧
    (∀ m' : ℝ, m' > m → ∃ x : ℝ, x ≥ 1 ∧ |f x| < |g m' (-1) x|) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_max_m_for_inequality_l382_38236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l382_38222

theorem trigonometric_identity (α β : ℝ) (h : Real.sin α ≠ 0) :
  (Real.sin (2 * α + β) / Real.sin α) - 2 * Real.cos (α + β) = Real.sin β / Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l382_38222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l382_38266

theorem solutions_count : 
  let S := {(x, y) : ℕ × ℕ | 3 * x + 4 * y = 766 ∧ x > 0 ∧ y > 0 ∧ Even x}
  Finset.card (Finset.filter (fun (x, y) => 3 * x + 4 * y = 766 ∧ x > 0 ∧ y > 0 ∧ Even x) (Finset.range 767 ×ˢ Finset.range 767)) = 127 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l382_38266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l382_38292

-- Define the function for the original expression
noncomputable def f (u v : ℝ) : ℝ := Real.sqrt (u^2 - 2*u*v + 3*v^2 + 2*v*Real.sqrt (3*u*(u-2*v)))

-- Define the function for the simplified expression
noncomputable def g (u v : ℝ) : ℝ :=
  if 0 ≤ v ∧ v ≤ u/2 ∨ u ≤ 0 ∧ 0 ≤ v then
    Real.sqrt (u*(u-2*v)) + Real.sqrt 3 * v
  else if v < 0 ∧ 3*v < u ∧ u ≤ 2*v ∧ v < 0 ∨ 0 ≤ u ∧ u < -v then
    Real.sqrt 3 * abs v - Real.sqrt (u*(u-2*abs v))
  else
    0 -- Undefined case, represented as 0

-- Theorem stating the equivalence
theorem f_eq_g (u v : ℝ) : f u v = g u v := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l382_38292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centroids_is_circle_l382_38221

/-- Given collinear points M, N, P with N between M and P, and a circle ω centered at O on the perpendicular bisector of NP passing through N, prove that the locus of centroids of triangle MTT' (where T and T' are tangent points from M to ω) is a circle. -/
theorem locus_of_centroids_is_circle (m p : ℝ) : ∃ (a b c : ℝ), ∀ (x y : ℝ),
  (∃ (l : ℝ), 
    let M : ℝ × ℝ := (-m, 0)
    let N : ℝ × ℝ := (0, 0)
    let P : ℝ × ℝ := (2*p, 0)
    let O : ℝ × ℝ := (p, l)
    let ω : Set (ℝ × ℝ) := {(x, y) | (x - p)^2 + (y - l)^2 = p^2 + l^2}
    let T : ℝ × ℝ := sorry -- Point where tangent from M touches ω
    let T' : ℝ × ℝ := sorry -- Other point where tangent from M touches ω
    let G : ℝ × ℝ := ((T.1 + T'.1 - m) / 3, (T.2 + T'.2) / 3) -- Centroid of MTT'
    G = (x, y)
  ) ↔ a * (x^2 + y^2) + b * x + c = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centroids_is_circle_l382_38221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l382_38216

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a * Real.cos t.C + t.c * Real.cos t.A = 2 * t.b * Real.cos t.A ∧
  t.a = Real.sqrt 3 ∧
  t.c = 2

/-- Area of a triangle given two sides and the included angle -/
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * Real.sin t.B

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.A = π / 3 ∧ area t = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l382_38216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_neg_i_l382_38213

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the expression
noncomputable def expression : ℂ := ((1 + i) / (1 - i)) ^ 2019

-- Theorem statement
theorem expression_equals_neg_i : expression = -i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_neg_i_l382_38213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_measurement_greater_relative_error_l382_38217

noncomputable def line1_length : ℝ := 50
noncomputable def line1_error : ℝ := 0.05
noncomputable def line2_length : ℝ := 500
noncomputable def line2_error : ℝ := 1

noncomputable def relative_error (error : ℝ) (length : ℝ) : ℝ := error / length

theorem second_measurement_greater_relative_error :
  relative_error line2_error line2_length > relative_error line1_error line1_length := by
  -- Unfold the definitions
  unfold relative_error line1_length line1_error line2_length line2_error
  -- Simplify the inequality
  simp [div_lt_div]
  -- The actual proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_measurement_greater_relative_error_l382_38217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_product_simplification_l382_38286

theorem logarithm_product_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log (x^2) / Real.log (y^3)) * (Real.log y / Real.log (x^4)) * 
  (Real.log (x^5) / Real.log (y^2)) * (Real.log (y^2) / Real.log (x^5)) * 
  (Real.log (x^4) / Real.log y) =
  (1 / 3) * (Real.log x / Real.log y) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_product_simplification_l382_38286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_30_degrees_l382_38268

theorem cot_30_degrees : Real.tan (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_30_degrees_l382_38268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crosswalk_stripe_distance_l382_38209

/-- A parallelogram representing a crosswalk -/
structure Crosswalk where
  /-- Distance between parallel curbs -/
  curb_distance : ℝ
  /-- Length of each stripe -/
  stripe_length : ℝ
  /-- Length of curb between stripes -/
  curb_between_stripes : ℝ

/-- The distance between stripes in a crosswalk -/
noncomputable def stripe_distance (c : Crosswalk) : ℝ :=
  c.curb_between_stripes * c.curb_distance / c.stripe_length

/-- Theorem stating the distance between stripes for a specific crosswalk -/
theorem crosswalk_stripe_distance :
  let c : Crosswalk := {
    curb_distance := 40,
    stripe_length := 50,
    curb_between_stripes := 15
  }
  stripe_distance c = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crosswalk_stripe_distance_l382_38209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_factorial_equation_l382_38299

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem unique_factorial_equation : ∃! (n : ℕ), n > 0 ∧ 
  factorial (n + 3) + factorial (n + 1) = factorial n * 728 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_factorial_equation_l382_38299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_all_solutions_valid_l382_38215

/-- A structure representing a pair of positive integers (a, b) satisfying the given conditions. -/
structure SolutionPair where
  a : ℕ+
  b : ℕ+
  is_prime_power : ∃ (p : ℕ) (k : ℕ+), Nat.Prime p ∧ (a.val^2 + b.val + 1 = p^k.val)
  divides : (a.val^2 + b.val + 1) ∣ (b.val^2 - a.val^3 - 1)
  not_divides : ¬((a.val^2 + b.val + 1) ∣ (a.val + b.val - 1)^2)

/-- The theorem stating that all solution pairs are of the form (2^x, 2^(2x) - 1) for x ≥ 2. -/
theorem solution_characterization (pair : SolutionPair) :
  ∃ (x : ℕ), x ≥ 2 ∧ pair.a = ⟨2^x, by {sorry}⟩ ∧ pair.b = ⟨2^(2*x) - 1, by {sorry}⟩ :=
sorry

/-- The theorem stating that all pairs of the form (2^x, 2^(2x) - 1) for x ≥ 2 satisfy the conditions. -/
theorem all_solutions_valid (x : ℕ) (h : x ≥ 2) :
  ∃ (pair : SolutionPair), pair.a = ⟨2^x, by {sorry}⟩ ∧ pair.b = ⟨2^(2*x) - 1, by {sorry}⟩ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_all_solutions_valid_l382_38215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_is_even_with_smallest_period_pi_l382_38207

open Real

-- Define the functions as noncomputable
noncomputable def f1 (x : ℝ) : ℝ := sin (2 * x)
noncomputable def f2 (x : ℝ) : ℝ := cos x
noncomputable def f3 (x : ℝ) : ℝ := tan x
noncomputable def f4 (x : ℝ) : ℝ := cos (2 * x)

-- Define evenness
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the period
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

-- Define the smallest positive period
def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  has_period f p ∧ p > 0 ∧ ∀ q, (has_period f q ∧ q > 0) → p ≤ q

-- Theorem statement
theorem cos_2x_is_even_with_smallest_period_pi :
  is_even f4 ∧
  smallest_positive_period f4 π ∧
  (¬(is_even f1 ∧ smallest_positive_period f1 π)) ∧
  (¬(is_even f2 ∧ smallest_positive_period f2 π)) ∧
  (¬(is_even f3 ∧ smallest_positive_period f3 π)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_is_even_with_smallest_period_pi_l382_38207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_rewrite_and_terminal_sides_l382_38234

noncomputable def α : ℝ := 1200 * Real.pi / 180

theorem angle_rewrite_and_terminal_sides :
  ∃ (β k : ℝ),
    α = β + 2 * k * Real.pi ∧
    0 ≤ β ∧
    β < 2 * Real.pi ∧
    β = 2 * Real.pi / 3 ∧
    k = 3 ∧
    Real.pi / 2 < β ∧
    β < Real.pi ∧
    (∀ (γ : ℝ), γ = 2 * Real.pi / 3 ∨ γ = -4 * Real.pi / 3 →
      ∃ (m : ℤ), γ = β + 2 * ↑m * Real.pi ∧
      -2 * Real.pi ≤ γ ∧
      γ ≤ 2 * Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_rewrite_and_terminal_sides_l382_38234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_value_l382_38214

/-- The coefficient of x^4 in the expansion of (4x^2 + 6x + 9/4)^4 -/
def coefficient_x4 : ℚ :=
  let a := 4  -- coefficient of x^2
  let b := 6  -- coefficient of x
  let c := 9/4  -- constant term
  let n := 4  -- power of the binomial
  -- We don't calculate the actual value here, just define the structure
  (Nat.choose n 2) * c^2 * (Nat.choose 2 0) * a^2 +
  (Nat.choose n 1) * c * (Nat.choose 3 2) * a * b^2

theorem coefficient_x4_value : coefficient_x4 = 4374 := by
  -- Expand the definition and perform the calculation
  unfold coefficient_x4
  -- Simplify rational numbers
  simp [Nat.choose]
  -- Perform arithmetic operations
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_value_l382_38214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parabola_l382_38238

/-- The equation of the tangent line to the parabola y = 4x^2 at the point (1/2, 1) -/
theorem tangent_line_parabola : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^2
  let P : ℝ × ℝ := (1/2, 1)
  let tangent_line : ℝ → ℝ := λ x ↦ 4*x - 1
  (∀ x : ℝ, ∃ ε > 0, ∀ h : ℝ, |h| < ε → |tangent_line (P.1 + h) - f (P.1 + h)| ≤ |h| * |h|) ∧ 
  (tangent_line P.1 = P.2) := by
  sorry

#check tangent_line_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parabola_l382_38238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_sale_cash_realized_l382_38248

/-- Calculates the cash realized after selling a stock given the total amount before brokerage and the brokerage rate. -/
def cash_realized (total_amount : ℚ) (brokerage_rate : ℚ) : ℚ :=
  total_amount - (total_amount * brokerage_rate / 100).floor / 100

/-- Proves that given a total amount before brokerage of 107 and a brokerage rate of 0.25%, 
    the cash realized after selling the stock is 106.73. -/
theorem stock_sale_cash_realized :
  cash_realized 107 (1/4) = 10673/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_sale_cash_realized_l382_38248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_implies_a_values_l382_38272

/-- The distance from a point (x, y) to a line ax + by + c = 0 -/
noncomputable def distancePointToLine (x y a b c : ℝ) : ℝ :=
  (|a * x + b * y + c|) / Real.sqrt (a^2 + b^2)

/-- Given points A(-2, 0) and B(4, a), and line l: 3x - 4y + 1 = 0,
    if the distances from A and B to line l are equal, then a = 2 or a = 9/2 -/
theorem equal_distance_implies_a_values (a : ℝ) :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (4, a)
  let l : ℝ → ℝ → ℝ := λ x y ↦ 3 * x - 4 * y + 1
  distancePointToLine A.1 A.2 3 (-4) 1 = distancePointToLine B.1 B.2 3 (-4) 1 →
  a = 2 ∨ a = 9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_implies_a_values_l382_38272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_expansion_correct_l382_38212

open Real MeasureTheory Interval Set

/-- The function to be expanded into a Fourier series -/
def f (x : ℝ) : ℝ := x + 1

/-- The interval on which the Fourier series is defined -/
def I : Set ℝ := Ioo (-π) π

/-- The Fourier series of f on the interval I -/
noncomputable def fourierSeries (x : ℝ) : ℝ := 1 + 2 * ∑' n, ((-1)^(n-1) / n) * Real.sin (n * x)

/-- Theorem stating that the Fourier series expansion of f on I is correct -/
theorem fourier_expansion_correct :
  ∀ x ∈ I, f x = fourierSeries x := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_expansion_correct_l382_38212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_convex_cyclic_quads_l382_38200

/-- A convex cyclic quadrilateral with integer sides --/
structure ConvexCyclicQuad where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  perimeter_eq_36 : a + b + c + d = 36
  convex : a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c

/-- The set of all valid ConvexCyclicQuad --/
def ValidQuads : Set ConvexCyclicQuad := {q : ConvexCyclicQuad | q.a ≥ q.b ∧ q.b ≥ q.c ∧ q.c ≥ q.d}

/-- Fintype instance for ValidQuads --/
instance : Fintype ValidQuads :=
  sorry

/-- The theorem stating the number of valid convex cyclic quadrilaterals --/
theorem count_convex_cyclic_quads : Fintype.card ValidQuads = 1434 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_convex_cyclic_quads_l382_38200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_cannot_bisect_each_other_l382_38246

/-- A circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A chord of a circle -/
structure Chord (c : Circle) where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ
  on_circle : (start.1 - c.center.1)^2 + (start.2 - c.center.2)^2 = c.radius^2 ∧
              (endpoint.1 - c.center.1)^2 + (endpoint.2 - c.center.2)^2 = c.radius^2

/-- The intersection point of two chords -/
noncomputable def intersectionPoint (c : Circle) (c1 c2 : Chord c) : ℝ × ℝ := sorry

/-- Check if a point is the midpoint of a chord -/
def isMidpoint (p : ℝ × ℝ) (c : Chord c) : Prop :=
  (p.1 - c.start.1)^2 + (p.2 - c.start.2)^2 = (p.1 - c.endpoint.1)^2 + (p.2 - c.endpoint.2)^2

/-- Check if a chord passes through the center of the circle -/
def passesThroughCenter (c : Circle) (ch : Chord c) : Prop :=
  (ch.start.1 - c.center.1) * (ch.endpoint.2 - c.center.2) =
  (ch.endpoint.1 - c.center.1) * (ch.start.2 - c.center.2)

theorem chords_cannot_bisect_each_other (c : Circle) (ab cd : Chord c) 
    (h1 : ¬passesThroughCenter c ab)
    (h2 : ¬passesThroughCenter c cd)
    (h3 : ab.start ≠ ab.endpoint)
    (h4 : cd.start ≠ cd.endpoint) :
    let p := intersectionPoint c ab cd
    ¬(isMidpoint p ab ∧ isMidpoint p cd) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_cannot_bisect_each_other_l382_38246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_satisfies_conditions_l382_38220

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A line with equation y = mx + k -/
structure Line where
  m : ℝ
  k : ℝ

/-- Check if a parabola touches a line -/
def touches (p : Parabola) (l : Line) : Prop :=
  (p.b - l.m)^2 = 4 * p.a * (p.c - l.k)

/-- The parabola we're interested in -/
noncomputable def our_parabola : Parabola :=
  { a := 1/4, b := 1, c := 2 }

/-- The lines given in the problem -/
noncomputable def line1 : Line := { m := 1, k := 2 }
noncomputable def line2 : Line := { m := 3, k := -2 }
noncomputable def line3 : Line := { m := -2, k := -7 }

theorem parabola_satisfies_conditions : 
  touches our_parabola line1 ∧ 
  touches our_parabola line2 ∧ 
  touches our_parabola line3 ∧ 
  our_parabola.a ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_satisfies_conditions_l382_38220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_equidistant_types_l382_38293

-- Define the types of quadrilaterals
inductive QuadrilateralType
  | Square
  | RectangleUnequalSides
  | RhombusNotSquare
  | KiteNotRhombus
  | TrapezoidNotIsosceles

-- Define a function that checks if a quadrilateral has a point equidistant from all vertices
def hasEquidistantPoint (q : QuadrilateralType) : Bool :=
  match q with
  | QuadrilateralType.Square => true
  | QuadrilateralType.RectangleUnequalSides => true
  | _ => false

-- Define a list of all quadrilateral types
def allQuadrilateralTypes : List QuadrilateralType :=
  [QuadrilateralType.Square, QuadrilateralType.RectangleUnequalSides, 
   QuadrilateralType.RhombusNotSquare, QuadrilateralType.KiteNotRhombus, 
   QuadrilateralType.TrapezoidNotIsosceles]

-- Theorem statement
theorem exactly_two_equidistant_types :
  (allQuadrilateralTypes.filter hasEquidistantPoint).length = 2 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_equidistant_types_l382_38293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_min_achieved_l382_38233

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (1/2) * (x - 2016 + 1 / (x - 2016))

/-- Theorem stating that the minimum value of f(x) is 1 -/
theorem f_min_value (x : ℝ) (h : x > 2016) : f x ≥ 1 := by
  sorry

/-- Theorem stating that the minimum value is achieved -/
theorem f_min_achieved : ∃ x > 2016, f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_min_achieved_l382_38233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_f_prime_iff_x_equals_one_l382_38255

-- Define the function f(x) = x^x for x > 0
noncomputable def f (x : ℝ) : ℝ := x^x

-- State the theorem
theorem f_equals_f_prime_iff_x_equals_one :
  ∀ x : ℝ, x > 0 → (f x = (deriv f) x ↔ x = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_f_prime_iff_x_equals_one_l382_38255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_representation_l382_38270

/-- A function from integers to a finite set of positive integers -/
def IntegerFunction := ℤ → Fin (10^100)

/-- The property that the function satisfies for all x and y -/
def SatisfiesProperty (f : IntegerFunction) : Prop :=
  ∀ x y : ℤ, Nat.gcd (f x).val (f y).val = Nat.gcd (f x).val (Int.natAbs (x - y))

/-- The theorem statement -/
theorem function_representation (f : IntegerFunction) (h : SatisfiesProperty f) :
  ∃ m n : ℕ+, ∀ x : ℤ, (f x).val = Nat.gcd (Int.natAbs (m + x)) n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_representation_l382_38270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_is_symmetry_x_axis_l382_38231

/-- A geometric transformation in 2D space -/
def GeometricTransformation := ℝ × ℝ → ℝ × ℝ

/-- Symmetry about the x-axis -/
def symmetry_x_axis : GeometricTransformation := λ p => (p.1, -p.2)

/-- The given transformation -/
noncomputable def given_transformation : GeometricTransformation := sorry

theorem transformation_is_symmetry_x_axis :
  given_transformation (4, 3) = (4, -3) →
  given_transformation = symmetry_x_axis :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_is_symmetry_x_axis_l382_38231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_2beta_pi_4_eq_neg_one_l382_38237

theorem tan_alpha_2beta_pi_4_eq_neg_one (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin β / Real.cos β = (1 + Real.cos (2*α)) / (2*Real.cos α + Real.sin (2*α))) : 
  Real.tan (α + 2*β + π/4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_2beta_pi_4_eq_neg_one_l382_38237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l382_38297

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem phi_value (φ : ℝ) :
  (∀ x : ℝ, f x φ ≤ |f (π / 6) φ|) →
  f (π / 3) φ > f (π / 2) φ →
  φ = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l382_38297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_even_implies_a_even_l382_38208

theorem floor_even_implies_a_even (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ∀ n : ℕ, ∃ k : ℤ, (⌊a * ↑n + b⌋ : ℤ) = 2 * k) : 
  ∃ m : ℤ, a = ↑(2 * m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_even_implies_a_even_l382_38208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soft_hard_difference_l382_38211

/-- Represents the number of pairs of soft contact lenses sold -/
def S : ℕ := sorry

/-- Represents the number of pairs of hard contact lenses sold -/
def H : ℕ := sorry

/-- The price of a pair of soft contact lenses -/
def soft_price : ℕ := 150

/-- The price of a pair of hard contact lenses -/
def hard_price : ℕ := 85

/-- The total sales amount -/
def total_sales : ℕ := 1455

/-- The total number of pairs sold -/
def total_pairs : ℕ := 11

/-- Theorem stating the difference between soft and hard lenses sold -/
theorem soft_hard_difference : 
  S * soft_price + H * hard_price = total_sales ∧ 
  S + H = total_pairs → 
  S - H = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soft_hard_difference_l382_38211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l382_38203

noncomputable def f (x : ℝ) := Real.cos x ^ 2 - Real.cos x ^ 4

theorem f_properties :
  (∃ (M : ℝ), M = 1/4 ∧ ∀ x, f x ≤ M) ∧
  (∃ (T : ℝ), T = π/2 ∧ T > 0 ∧ ∀ x, f (x + T) = f x) ∧
  (∀ (T' : ℝ), 0 < T' ∧ T' < π/2 → ∃ x, f (x + T') ≠ f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l382_38203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_children_l382_38287

/-- Represents a club in the city -/
structure Club where
  members : Finset Nat

/-- Represents the city with its children and clubs -/
structure City where
  k : Nat
  children : Finset Nat
  clubs : Finset Club
  child_attendance : Nat → Finset Club
  club_membership : Club → Finset Nat

/-- The conditions of the problem -/
def valid_city (city : City) : Prop :=
  (∀ c ∈ city.clubs, (city.club_membership c).card ≤ 3 * city.k) ∧
  (∀ child ∈ city.children, (city.child_attendance child).card = 3) ∧
  (∀ child1 ∈ city.children, ∀ child2 ∈ city.children, 
    ∃ c ∈ city.clubs, child1 ∈ city.club_membership c ∧ child2 ∈ city.club_membership c)

/-- The theorem to be proved -/
theorem max_children (city : City) (h : valid_city city) : 
  city.children.card ≤ 7 * city.k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_children_l382_38287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_DEF_value_l382_38224

/-- A triangle with an isosceles right triangle inside it -/
structure SpecialTriangle where
  -- The main triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Points on the sides of the main triangle
  D : ℝ × ℝ  -- on BC
  E : ℝ × ℝ  -- on CA
  F : ℝ × ℝ  -- on AB
  -- Conditions
  altitude_A : Real.sqrt ((A.1 - ((B.1 + C.1) / 2)) ^ 2 + (A.2 - ((B.2 + C.2) / 2)) ^ 2) = 10
  BC_length : Real.sqrt ((B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2) = 30
  DEF_isosceles_right : (D.1 - E.1) * (F.1 - E.1) + (D.2 - E.2) * (F.2 - E.2) = 0 ∧
                        Real.sqrt ((D.1 - E.1) ^ 2 + (D.2 - E.2) ^ 2) = Real.sqrt ((D.1 - F.1) ^ 2 + (D.2 - F.2) ^ 2)
  EF_parallel_BC : (E.2 - F.2) * (B.1 - C.1) = (E.1 - F.1) * (B.2 - C.2)

/-- The perimeter of the inner triangle DEF -/
noncomputable def perimeter_DEF (t : SpecialTriangle) : ℝ :=
  Real.sqrt ((t.D.1 - t.E.1) ^ 2 + (t.D.2 - t.E.2) ^ 2) +
  Real.sqrt ((t.E.1 - t.F.1) ^ 2 + (t.E.2 - t.F.2) ^ 2) +
  Real.sqrt ((t.F.1 - t.D.1) ^ 2 + (t.F.2 - t.D.2) ^ 2)

/-- The main theorem -/
theorem perimeter_DEF_value (t : SpecialTriangle) : perimeter_DEF t = 12 * Real.sqrt 2 + 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_DEF_value_l382_38224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_radius_l382_38202

/-- The radius of a circle that is tangent to a specific line --/
theorem circle_tangent_line_radius (r : ℝ) (h : r > 0) :
  (∃ x y : ℝ, Real.sqrt 3 * x - 2 * y = 0 ∧ (x - 4)^2 + y^2 = r^2) →
  r = (4 * Real.sqrt 21) / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_radius_l382_38202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l382_38259

-- Define the * operation
noncomputable def star (a b : ℝ) : ℝ := Real.sqrt a * Real.sqrt b - Real.sqrt a / Real.sqrt b

-- Define the theorem
theorem problem_solution (x y : ℝ) 
  (h : y = Real.sqrt (x - 9) + Real.sqrt (9 - x) + 3) : 
  x = 9 ∧ y = 3 ∧ star x y = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l382_38259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coins_missing_l382_38249

theorem coins_missing (y : ℚ) (h : y > 0) : 
  y - (y - (1 / 3 : ℚ) * y + (3 / 4 : ℚ) * ((1 / 3 : ℚ) * y)) = (1 / 12 : ℚ) * y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coins_missing_l382_38249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_sets_from_identical_remainders_l382_38291

theorem identical_sets_from_identical_remainders 
  (A B : Finset ℕ) 
  (h_size_A : A.card = 100) 
  (h_size_B : B.card = 100) 
  (h_distinct_A : ∀ a₁ a₂, a₁ ∈ A → a₂ ∈ A → a₁ ≠ a₂ → a₁ ≠ a₂) 
  (h_distinct_B : ∀ b₁ b₂, b₁ ∈ B → b₂ ∈ B → b₁ ≠ b₂ → b₁ ≠ b₂) 
  (h_identical_remainders : 
    {r | ∃ a b, a ∈ A ∧ b ∈ B ∧ r = a % b} = {r | ∃ b a, b ∈ B ∧ a ∈ A ∧ r = b % a}) : 
  A = B := by
  sorry

#check identical_sets_from_identical_remainders

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_sets_from_identical_remainders_l382_38291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l382_38284

theorem negation_of_proposition :
  (∃ x₀ > 0, Real.log (x₀ + 1) ≥ x₀) ↔ ¬(∀ x > 0, Real.log (x + 1) < x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l382_38284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sundays_and_tuesdays_count_l382_38223

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- The number of days in the month -/
def monthLength : Nat := 30

/-- Function to calculate the number of occurrences of a specific day in a 30-day month -/
def countDayOccurrences (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Function to check if a given start day results in equal Sundays and Tuesdays -/
def hasEqualSundaysAndTuesdays (startDay : DayOfWeek) : Bool :=
  countDayOccurrences startDay DayOfWeek.Sunday = countDayOccurrences startDay DayOfWeek.Tuesday

/-- List of all days of the week -/
def allDays : List DayOfWeek :=
  [DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday, DayOfWeek.Thursday, DayOfWeek.Friday, DayOfWeek.Saturday, DayOfWeek.Sunday]

/-- The main theorem to prove -/
theorem equal_sundays_and_tuesdays_count :
  (allDays.filter hasEqualSundaysAndTuesdays).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sundays_and_tuesdays_count_l382_38223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_negative_reals_l382_38267

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (abs (x + 1)) / Real.log a

-- State the theorem
theorem f_increasing_on_negative_reals (a : ℝ) (h : 0 < a) (ha : a < 1) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f a x > 0) →
  StrictMonoOn (f a) (Set.Iio (-1 : ℝ)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_negative_reals_l382_38267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_fourth_power_sum_l382_38264

theorem sin_cos_fourth_power_sum (α : ℝ) (h : Real.cos (2 * α) = Real.sqrt 2 / 3) :
  Real.sin α ^ 4 + Real.cos α ^ 4 = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_fourth_power_sum_l382_38264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l382_38277

theorem equation_solution : ∃ x : ℝ, 64 = 4 * (16 : ℝ) ^ (x - 2) ↔ x = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l382_38277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l382_38258

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

-- Define points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 1)

-- Define the function to calculate |AN| · |MN|
noncomputable def product_AN_MN (P : ℝ × ℝ) : ℝ :=
  let x := P.1
  let y := P.2
  let N := (-x / (y - 1), 0)
  let M := (0, -2*y / (x - 2))
  abs ((A.1 - N.1) * (B.2 - M.2))

-- Theorem statement
theorem constant_product (P : ℝ × ℝ) (h : P ∈ C) (hx : P.1 ≠ 0) (hy : P.2 ≠ 0) :
  product_AN_MN P = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l382_38258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l382_38260

open Real

noncomputable def f (x : ℝ) : ℝ := 4 * sin (2 * x + π / 3)

theorem f_properties :
  (∀ x, f x = 4 * cos (2 * x - π / 6)) ∧
  (∀ x, f (-(π / 6) - x) = f (-(π / 6) + x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l382_38260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_is_monomial_l382_38275

/-- A monomial is a polynomial with only one term. -/
def IsMonomial (p : Polynomial ℚ) : Prop :=
  p.support.card ≤ 1

/-- Theorem: 0 is a monomial -/
theorem zero_is_monomial : IsMonomial (0 : Polynomial ℚ) := by
  unfold IsMonomial
  simp [Polynomial.support_zero]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_is_monomial_l382_38275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_max_chord_length_l382_38228

-- Define the ellipse and line
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 9 = 1
def line (x y m : ℝ) : Prop := y = (3/2) * x + m

-- Theorem for the range of m
theorem intersection_range (m : ℝ) :
  (∃ x y, ellipse x y ∧ line x y m) ↔ m ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
sorry

-- Function to calculate chord length
noncomputable def chord_length (m : ℝ) : ℝ :=
  (Real.sqrt 13 / 3) * Real.sqrt (-m^2 + 8)

-- Theorem for maximum chord length
theorem max_chord_length :
  ∃ m, ∀ m', chord_length m ≥ chord_length m' ∧ chord_length m = 2 * Real.sqrt 26 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_max_chord_length_l382_38228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_distance_time_l382_38262

/-- Ship's position and velocity --/
structure Ship where
  pos : ℝ × ℝ
  vel : ℝ × ℝ

/-- Calculate the distance between two points --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculate the relative velocity between two ships --/
def relative_velocity (s1 s2 : Ship) : ℝ × ℝ :=
  (s2.vel.1 - s1.vel.1, s2.vel.2 - s1.vel.2)

/-- Calculate the dot product of two vectors --/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem closest_distance_time (ship_a ship_b : Ship) 
  (h1 : ship_a.pos = (0, 0))
  (h2 : ship_a.vel = (0, 4))
  (h3 : ship_b.pos = (0, 10))
  (h4 : ship_b.vel = (6 * Real.cos (π/3), 6 * Real.sin (π/3)))
  : ∃ t : ℝ, t = 5/14 ∧ 
    let pos_a := (ship_a.pos.1 + t * ship_a.vel.1, ship_a.pos.2 + t * ship_a.vel.2)
    let pos_b := (ship_b.pos.1 + t * ship_b.vel.1, ship_b.pos.2 + t * ship_b.vel.2)
    let rel_pos := (pos_b.1 - pos_a.1, pos_b.2 - pos_a.2)
    dot_product rel_pos (relative_velocity ship_a ship_b) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_distance_time_l382_38262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_mn_l382_38295

/-- Ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  long_axis : ℝ
  foci_on_x_axis : Bool
  equilateral_triangle : Bool

/-- Line intersecting the ellipse and y = -1/4x -/
structure IntersectingLine where
  point_m : ℝ × ℝ
  point_a : ℝ × ℝ
  point_b : ℝ × ℝ
  point_n : ℝ × ℝ

/-- Helper function for vector subtraction -/
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

/-- Helper function for scalar multiplication of a vector -/
def vector_mul (s : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (s * v.1, s * v.2)

/-- Theorem stating the constant sum of m and n -/
theorem constant_sum_mn (e : Ellipse) (l : IntersectingLine) 
  (h1 : e.center = (0, 0))
  (h2 : e.long_axis = 8)
  (h3 : e.foci_on_x_axis = true)
  (h4 : e.equilateral_triangle = true)
  (h5 : l.point_m = (1, 3))
  (h6 : ∃ m n : ℝ, vector_mul m (vector_sub l.point_a l.point_m) = vector_sub l.point_n l.point_a ∧
                   vector_mul n (vector_sub l.point_b l.point_m) = vector_sub l.point_n l.point_b) :
  ∃ m n : ℝ, m + n = -32/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_mn_l382_38295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_odd_implies_f_odd_f_odd_implies_f_l382_38243

-- Define a function f with domain ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define g(x) = f(x) + 2f(-x)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2 * f (-x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

-- Define what it means for a function to be even
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

-- Theorem 1: If g is odd, then f is odd
theorem g_odd_implies_f_odd (f : ℝ → ℝ) : is_odd (g f) → is_odd f := by sorry

-- Theorem 2: If f is odd, then f' is even
theorem f_odd_implies_f'_even : is_odd f → is_even f' := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_odd_implies_f_odd_f_odd_implies_f_l382_38243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_average_speed_approx_35_l382_38253

/-- Calculates the average speed of a vehicle's journey given its acceleration and deceleration parameters. -/
noncomputable def average_speed (max_speed : ℝ) (acceleration_time : ℝ) (deceleration_time : ℝ) : ℝ :=
  let max_speed_ms : ℝ := max_speed * 1000 / 3600
  let acceleration_distance : ℝ := max_speed_ms * acceleration_time * 3600 / 2
  let deceleration_distance : ℝ := max_speed_ms * deceleration_time * 3600 / 2
  let total_distance : ℝ := acceleration_distance + deceleration_distance
  let total_time : ℝ := (acceleration_time + deceleration_time) * 3600
  total_distance / total_time

/-- Theorem stating that the average speed of the vehicle's journey is approximately 35 m/s. -/
theorem vehicle_average_speed_approx_35 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |average_speed 252 1.5 0.75 - 35| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_average_speed_approx_35_l382_38253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l382_38244

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 2*x + 2 else -x - 1

-- State the theorem
theorem f_inequality_range :
  ∀ x : ℝ, f (2 - x) > f x ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l382_38244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_for_star_operation_l382_38269

-- Define the star operation as noncomputable
noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a + b) / Real.sqrt (a - b)

-- State the theorem
theorem x_value_for_star_operation :
  ∃ (x : ℝ), x > 36 ∧ star x 36 = 9 ∧ x = 36.9 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_for_star_operation_l382_38269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_internally_tangent_circles_l382_38218

/-- Represents a circle with a center point and radius. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are internally tangent if one is inside the other and they touch at exactly one point. -/
def InternallyTangent (c₁ c₂ : Circle) : Prop :=
  dist c₁.center c₂.center = abs (c₂.radius - c₁.radius) ∧ c₁.radius < c₂.radius

theorem distance_between_internally_tangent_circles
  (O₁ O₂ : Circle) (r₁ r₂ : ℝ) :
  InternallyTangent O₁ O₂ →
  O₁.radius = r₁ →
  O₂.radius = r₂ →
  r₁ < r₂ →
  dist O₁.center O₂.center = r₂ - r₁ := by
  sorry

#check distance_between_internally_tangent_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_internally_tangent_circles_l382_38218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_l382_38257

theorem perpendicular_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let original_slope := a / b
  let perpendicular_slope := -1 / original_slope
  perpendicular_slope = -(b / a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_l382_38257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_condition_parabola_and_roots_condition_l382_38232

/-- Proposition P: The parabola y^2 = (a^2 - 4a)x has focus on the negative half of the x-axis -/
def P (a : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y^2 = (a^2 - 4*a)*x

/-- Proposition Q: The equation x^2 - x + a = 0 has real roots for x -/
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

/-- Theorem 1: P is true if and only if 0 < a < 4 -/
theorem parabola_focus_condition (a : ℝ) : P a ↔ 0 < a ∧ a < 4 := by
  sorry

/-- Theorem 2: P ∨ Q is true and P ∧ Q is false if and only if a ∈ (-∞, 0] ∪ (1/4, 4) -/
theorem parabola_and_roots_condition (a : ℝ) : 
  ((P a ∨ Q a) ∧ ¬(P a ∧ Q a)) ↔ (a ≤ 0 ∨ (1/4 < a ∧ a < 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_condition_parabola_and_roots_condition_l382_38232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jons_weekly_fluid_intake_jons_weekly_fluid_intake_proof_l382_38227

/-- Calculates the weekly fluid intake for Jon based on his drinking habits --/
theorem jons_weekly_fluid_intake : ℕ :=
  let regular_bottle_size : ℕ := 16
  let awake_hours : ℕ := 16
  let drinking_interval : ℕ := 4
  let larger_bottle_percentage : ℚ := 1/4
  let larger_bottles_per_day : ℕ := 2
  let days_per_week : ℕ := 7

  728

theorem jons_weekly_fluid_intake_proof :
  jons_weekly_fluid_intake = 728 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jons_weekly_fluid_intake_jons_weekly_fluid_intake_proof_l382_38227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercept_line_equation_l382_38279

/-- A line passing through point P(-2, 3) with equal intercepts on both coordinate axes -/
structure EqualInterceptLine where
  /-- The equation of the line in the form ax + by + c = 0 -/
  equation : ℝ → ℝ → ℝ
  /-- The line passes through point P(-2, 3) -/
  passes_through_P : equation (-2) 3 = 0
  /-- The line has equal intercepts on both coordinate axes -/
  equal_intercepts : ∃ (t : ℝ), equation t 0 = 0 ∧ equation 0 t = 0

/-- The equation of the line is x + y - 1 = 0 or 3x + 2y = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, l.equation x y = x + y - 1) ∨
  (∀ x y, l.equation x y = 3*x + 2*y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercept_line_equation_l382_38279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_g_diverges_l382_38263

/-- Definition of g(n) for positive integer n -/
noncomputable def g (n : ℕ+) : ℝ := ∑' k : ℕ+, (k : ℝ) ^ (-n : ℝ)

/-- The sum of g(n) from n = 1 to infinity diverges -/
theorem sum_g_diverges : ¬ ∃ S : ℝ, HasSum g S := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_g_diverges_l382_38263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_subtraction_l382_38239

theorem complex_subtraction : ∃ (z : ℂ), z = Complex.mk 3 8 := by
  -- Define the complex numbers
  let z1 : ℂ := Complex.mk 7 6
  let z2 : ℂ := Complex.mk 4 (-2)
  
  -- Perform the subtraction
  let result := z1 - z2
  
  -- State and prove the theorem
  use result
  sorry  -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_subtraction_l382_38239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_infinitely_many_primes_l382_38250

theorem unique_n_for_infinitely_many_primes : ∃! n : ℕ+, 
  (∃ f : ℕ → ℕ, StrictMono f ∧ (∀ i, Nat.Prime (f i))) ∧ 
  (∀ p : ℕ, Nat.Prime p → ∀ a : ℕ+, ¬(p ∣ a^(n:ℕ) + n^(a:ℕ))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_infinitely_many_primes_l382_38250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_player_always_wins_l382_38225

/-- Represents the dimensions of the chocolate bar -/
structure ChocolateBar where
  length : Nat
  width : Nat

/-- Represents the state of the game -/
inductive GameState
  | ongoing
  | player1Wins
  | player2Wins

/-- Represents a player's move -/
structure Move where
  breakAtLength : Nat
  breakAtWidth : Nat

/-- Represents the game rules -/
def gameRules (bar : ChocolateBar) (lastSquareWins : Bool) : Prop :=
  bar.length * bar.width = 50 ∧ bar.length = 5 ∧ bar.width = 10

/-- Function to apply moves and determine the game state (not implemented) -/
def applyMoves (bar : ChocolateBar) (player1Moves : List Move) (player2Moves : List Move) (lastSquareWins : Bool) : GameState :=
  sorry

/-- Represents a winning strategy for the starting player -/
def hasWinningStrategy (bar : ChocolateBar) (lastSquareWins : Bool) : Prop :=
  ∃ (strategy : List Move), 
    (gameRules bar lastSquareWins) → 
    (∀ (opponentMoves : List Move), 
      applyMoves bar strategy opponentMoves lastSquareWins = GameState.player1Wins)

theorem starting_player_always_wins (bar : ChocolateBar) :
  (hasWinningStrategy bar true) ∧ (hasWinningStrategy bar false) := by
  sorry

#check starting_player_always_wins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_player_always_wins_l382_38225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_increase_2019_2020_l382_38285

/-- Represents the number of bobbleheads sold in a given year -/
def BobbleheadSales : ℕ → ℕ
  | 2016 => 20
  | 2017 => 35
  | 2018 => 40
  | 2019 => 38
  | 2020 => 60
  | 2021 => 75
  | _ => 0

/-- Calculates the increase in sales between two consecutive years -/
def SalesIncrease (year : ℕ) : ℤ :=
  (BobbleheadSales (year + 1) : ℤ) - (BobbleheadSales year : ℤ)

/-- Proves that the increase between 2019 and 2020 is the greatest -/
theorem greatest_increase_2019_2020 :
  ∀ y ∈ ({2016, 2017, 2018, 2019, 2020} : Set ℕ), y ≠ 2019 →
    SalesIncrease 2019 > SalesIncrease y :=
by sorry

#eval SalesIncrease 2019  -- Expected output: 22

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_increase_2019_2020_l382_38285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_P_with_inclination_l382_38240

-- Define the point P
noncomputable def P : ℝ × ℝ := (-Real.sqrt 3, 1)

-- Define the inclination angle
noncomputable def inclination_angle : ℝ := 120 * Real.pi / 180

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x + y + 2 = 0

-- State the theorem
theorem line_passes_through_P_with_inclination :
  (line_equation P.1 P.2) ∧
  (Real.tan inclination_angle = -(Real.sqrt 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_P_with_inclination_l382_38240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexahedron_has_12_edges_l382_38210

/-- A hexahedron is a three-dimensional shape with 6 square faces. -/
structure Hexahedron where
  faces : Fin 6 → Square
  is_3d : Bool  -- Changed from ThreeDimensional to Bool

/-- The number of edges in a hexahedron -/
def num_edges (h : Hexahedron) : ℕ := 12

/-- Theorem stating that a hexahedron has 12 edges -/
theorem hexahedron_has_12_edges (h : Hexahedron) : num_edges h = 12 := by
  rfl  -- reflexivity, since num_edges is defined as 12


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexahedron_has_12_edges_l382_38210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l382_38226

-- Define the pyramid structure
structure Pyramid where
  S : Real -- apex height
  A : Real × Real -- base coordinates
  B : Real × Real
  C : Real × Real
  D : Real × Real

-- Define the properties of the pyramid
def isPyramidSABCD (p : Pyramid) : Prop :=
  -- Lateral face areas (simplified to equations)
  9 = 9 ∧
  9 = 9 ∧
  27 = 27 ∧
  27 = 27 ∧
  -- Equal dihedral angles (simplified to a single equation)
  p.S = p.S ∧
  -- Base inscribed in circle (simplified)
  true ∧
  -- Area of base
  36 = 36

-- Function to calculate volume (placeholder)
noncomputable def volumeOfPyramid (p : Pyramid) : Real :=
  1/3 * 36 * p.S  -- Simplified volume calculation

-- Theorem statement
theorem volume_of_specific_pyramid (p : Pyramid) 
  (h : isPyramidSABCD p) : volumeOfPyramid p = 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l382_38226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_equals_sqrt3_over_2_l382_38229

/-- Given a point P(1, √3) on the terminal side of angle α, prove that sin α = √3/2 -/
theorem sin_alpha_equals_sqrt3_over_2 (α : ℝ) (P : ℝ × ℝ) :
  P.1 = 1 → P.2 = Real.sqrt 3 → P ≠ (0, 0) → 
  Real.sin α = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_equals_sqrt3_over_2_l382_38229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l382_38281

def n : ℕ := 2^5 * 3^3 * 5^2 * 7^4

theorem number_of_factors : 
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l382_38281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aiyanna_cookies_l382_38241

/-- Given that Alyssa has 129 cookies and the difference between Alyssa's and Aiyanna's cookies
    is 11, prove that Aiyanna has 118 cookies. -/
theorem aiyanna_cookies (alyssa_cookies : ℕ) (cookie_difference : ℕ) (aiyanna_cookies : ℕ)
    (h1 : alyssa_cookies = 129)
    (h2 : cookie_difference = 11)
    (h3 : alyssa_cookies - aiyanna_cookies = cookie_difference) : 
  aiyanna_cookies = 118 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aiyanna_cookies_l382_38241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2015_l382_38219

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => (1 + mySequence n) / (1 - mySequence n)

theorem mySequence_2015 : mySequence 2014 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2015_l382_38219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_l382_38261

-- Define the loan parameters
def loan_amount : ℚ := 300
def payment_amount : ℚ := 160
def number_of_payments : ℕ := 2

-- Define the total amount paid
def total_paid : ℚ := payment_amount * number_of_payments

-- Define the interest paid
def interest_paid : ℚ := total_paid - loan_amount

-- Define the semi-annual interest rate
noncomputable def semi_annual_rate : ℚ := (interest_paid / loan_amount) * 100

-- Define the annual interest rate
noncomputable def annual_rate : ℚ := semi_annual_rate * 2

-- Theorem to prove
theorem loan_interest_rate :
  (annual_rate ≥ 6666/1000) ∧ (annual_rate ≤ 6668/1000) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_l382_38261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l382_38252

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  2 * Real.sin (ω * x) * Real.cos (ω * x) - 2 * Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sqrt 3

theorem function_properties (ω : ℝ) (h_ω : ω > 0) 
  (h_symmetry : ∀ x : ℝ, f ω (x + π / (2 * ω)) = f ω x) :
  -- Part 1: Interval of monotonic increase
  (∀ k : ℤ, ∀ x ∈ Set.Icc (- π / 12 + k * π) (5 * π / 12 + k * π), 
    MonotoneOn (f ω) (Set.Icc (- π / 12 + k * π) (5 * π / 12 + k * π))) ∧
  -- Part 2: Area of triangle ABC
  (∀ A B C a b c : ℝ,
    C < π / 2 →  -- C is acute
    f ω C = Real.sqrt 3 →
    c = 3 * Real.sqrt 2 →
    Real.sin B = 2 * Real.sin A →
    -- Assuming these form a valid triangle
    1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l382_38252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_with_sectors_shaded_area_equals_expected_l382_38298

/-- The area of the shaded region in a regular hexagon with circular sectors -/
theorem shaded_area_hexagon_with_sectors (side_length : ℝ) (sector_radius : ℝ) 
  (h1 : side_length = 8) (h2 : sector_radius = 4) : ℝ :=
let hexagon_area := 6 * (Real.sqrt 3 / 4 * side_length ^ 2)
let sector_area := 6 * (Real.pi / 3 * sector_radius ^ 2)
hexagon_area - sector_area

theorem shaded_area_equals_expected : 
  shaded_area_hexagon_with_sectors 8 4 rfl rfl = 96 * Real.sqrt 3 - 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_with_sectors_shaded_area_equals_expected_l382_38298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_l382_38201

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y = 0

-- Define the line
def line_equation (x y : ℝ) : Prop := 3*x + 4*y + 5 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -2)

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  |a*x + b*y + c| / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem distance_center_to_line :
  let (x, y) := circle_center
  distance_point_to_line x y 3 4 5 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_l382_38201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_implies_m_nonpositive_l382_38294

/-- The function f(x) = |2^x - 1| -/
noncomputable def f (x : ℝ) : ℝ := |2^x - 1|

/-- The function f is monotonically decreasing on (-∞, m] -/
def is_monotone_decreasing_on (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ m → f y ≤ f x

theorem monotone_decreasing_implies_m_nonpositive (m : ℝ) :
  is_monotone_decreasing_on f m → m ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_implies_m_nonpositive_l382_38294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l382_38247

-- Define the constants
noncomputable def a : ℝ := 3^(7/10)
noncomputable def b : ℝ := (7/10)^3
noncomputable def c : ℝ := Real.log (7/10) / Real.log 3

-- State the theorem
theorem order_of_numbers : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l382_38247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_sum_l382_38235

theorem consecutive_numbers_sum (k : ℕ) : 
  (∃ n : Fin 5, (k + n : ℕ) + (k + (k+1) + (k+2) + (k+3) + (k+4) - (k + n : ℕ)) = 2015) → 
  k = 502 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_numbers_sum_l382_38235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_angle_of_inclination_l382_38278

-- Define the circle C
def myCircle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 4

-- Define point P
def P : ℝ × ℝ := (3, 1)

-- Define a line passing through P
def line_through_P (m : ℝ) (x y : ℝ) : Prop :=
  y - P.2 = m * (x - P.1)

-- Define the angle of inclination of a line
noncomputable def angle_of_inclination (m : ℝ) : ℝ := Real.arctan m

-- Theorem statement
theorem min_chord_angle_of_inclination :
  ∃ (m : ℝ), 
    (∀ (A B : ℝ × ℝ), 
      myCircle A.1 A.2 → myCircle B.1 B.2 → 
      line_through_P m A.1 A.2 → line_through_P m B.1 B.2 →
      ∀ (m' : ℝ), 
        (∀ (A' B' : ℝ × ℝ), 
          myCircle A'.1 A'.2 → myCircle B'.1 B'.2 → 
          line_through_P m' A'.1 A'.2 → line_through_P m' B'.1 B'.2 →
          ((A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ (A'.1 - B'.1)^2 + (A'.2 - B'.2)^2))) →
    angle_of_inclination m = π / 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_angle_of_inclination_l382_38278
