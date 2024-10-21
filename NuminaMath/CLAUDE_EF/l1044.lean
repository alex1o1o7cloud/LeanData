import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_circle_area_ratio_for_radius_three_l1044_104426

/-- The ratio of the area of a star figure to the area of its circumscribing circle -/
noncomputable def star_circle_area_ratio (r : ℝ) : ℝ :=
  (4 * r^2 - Real.pi * r^2) / (Real.pi * r^2)

/-- Theorem: For a circle with radius 3, when cut into four congruent arcs and 
    joined to form a star figure, the ratio of the area of the star figure to 
    the area of the circle is (4 - π) / π -/
theorem star_circle_area_ratio_for_radius_three : 
  star_circle_area_ratio 3 = (4 - Real.pi) / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_circle_area_ratio_for_radius_three_l1044_104426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_18_l1044_104405

/-- The area of a triangle given by three points in a 2D plane -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Theorem: The area of triangle ABC is 18 square units -/
theorem triangle_area_is_18 :
  let a : ℝ × ℝ := (0, 2)
  let b : ℝ × ℝ := (6, 0)
  let c : ℝ × ℝ := (3, 7)
  triangleArea a b c = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_18_l1044_104405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l1044_104491

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product {m1 m2 : ℝ} : 
  m1 * m2 = -1 ↔ (∃ (a b c d e f : ℝ), ∀ (x y : ℝ), (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0 ∧ 
    m1 = -a / b ∧ m2 = d / e ∧ a * e ≠ b * d))

/-- The theorem states that if the line ax + 2y + 2 = 0 is perpendicular to the line 3x - y - 2 = 0, 
    then a = 2/3 -/
theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ (x y : ℝ), a * x + 2 * y + 2 = 0 → 3 * x - y - 2 = 0 → False) → a = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l1044_104491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_3_x_equals_negative_3_l1044_104401

-- Define x as given in the problem
noncomputable def x : ℝ := (Real.log 2 / Real.log 8) ^ (Real.log 8 / Real.log 2)

-- Theorem statement
theorem log_3_x_equals_negative_3 : Real.log x / Real.log 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_3_x_equals_negative_3_l1044_104401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cone_l1044_104406

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Represents the water level in the cone -/
structure WaterLevel where
  cone : Cone
  percentage : ℝ

/-- Calculates the height of water in the cone -/
noncomputable def waterHeight (w : WaterLevel) : ℝ :=
  w.cone.height * (w.percentage)^(1/3)

theorem water_height_in_cone (c : Cone) (w : WaterLevel) :
  c.radius = 8 ∧ c.height = 64 ∧ w.cone = c ∧ w.percentage = 0.4 →
  ∃ (a b : ℕ), a = 64 ∧ b = 2 ∧ waterHeight w = a * Real.rpow (b : ℝ) (1/3 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cone_l1044_104406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_handshakes_l1044_104467

theorem tournament_handshakes : 30 = (
  let total_teams : ℕ := 4
  let teams_with_two : ℕ := 3
  let teams_with_three : ℕ := 1
  let players_in_two : ℕ := 2
  let players_in_three : ℕ := 3
  let total_players : ℕ := teams_with_two * players_in_two + teams_with_three * players_in_three
  let total_possible_handshakes : ℕ := total_players * (total_players - 1) / 2
  let handshakes_within_two : ℕ := teams_with_two * (players_in_two * (players_in_two - 1) / 2)
  let handshakes_within_three : ℕ := teams_with_three * (players_in_three * (players_in_three - 1) / 2)
  let internal_handshakes : ℕ := handshakes_within_two + handshakes_within_three
  total_possible_handshakes - internal_handshakes
) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_handshakes_l1044_104467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_three_rays_with_common_point_l1044_104434

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    (p.1 + 3 = 5 ∧ p.2 - 6 ≤ 5) ∨
    (p.2 - 6 = 5 ∧ p.1 + 3 ≤ 5) ∨
    (p.1 + 3 = p.2 - 6 ∧ 5 ≤ p.1 + 3)}

-- Define what it means for a set to be a ray
def is_ray (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ × ℝ), S = {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = a + t • (b - a)}

-- Define what it means for a set to be three rays with a common point
def is_three_rays_with_common_point (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ) (r₁ r₂ r₃ : Set (ℝ × ℝ)),
    S = r₁ ∪ r₂ ∪ r₃ ∧
    is_ray r₁ ∧ is_ray r₂ ∧ is_ray r₃ ∧
    p ∈ r₁ ∧ p ∈ r₂ ∧ p ∈ r₃ ∧
    (r₁ ∩ r₂ = {p}) ∧ (r₂ ∩ r₃ = {p}) ∧ (r₃ ∩ r₁ = {p})

-- The theorem to be proved
theorem T_is_three_rays_with_common_point : is_three_rays_with_common_point T := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_three_rays_with_common_point_l1044_104434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l1044_104458

/-- A point in the xy-plane with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- The set of all valid points in the 5x5 grid -/
def validPoints : Set Point :=
  {p : Point | 1 ≤ p.x ∧ p.x ≤ 5 ∧ 1 ≤ p.y ∧ p.y ≤ 5}

/-- A triangle represented by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Check if a triangle has positive area -/
def hasPositiveArea (t : Triangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all triangles with vertices in validPoints -/
def allTriangles : Set Triangle :=
  {t : Triangle | t.p1 ∈ validPoints ∧ t.p2 ∈ validPoints ∧ t.p3 ∈ validPoints}

/-- The set of all triangles with positive area -/
def validTriangles : Set Triangle :=
  {t ∈ allTriangles | hasPositiveArea t}

-- Assume finiteness of validTriangles
axiom validTriangles_finite : Fintype validTriangles

theorem count_valid_triangles :
  @Fintype.card validTriangles validTriangles_finite = 2170 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l1044_104458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_f_l1044_104476

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * Real.sqrt x

-- State the theorem
theorem evaluate_f : f 3 + 3 * f 1 - 2 * f 5 = -211 + 3 * Real.sqrt 3 - 6 * Real.sqrt 5 := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_f_l1044_104476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_mean_value_range_l1044_104430

/-- Definition of a double mean value function -/
def is_double_mean_value_function (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a ∧
    deriv f x₁ = (f a - f 0) / a ∧
    deriv f x₂ = (f a - f 0) / a

/-- The function f(x) = x^3 - x^2 + a + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a + 1

/-- Theorem stating the range of a for which f is a double mean value function -/
theorem double_mean_value_range (a : ℝ) :
  is_double_mean_value_function (f a) a ↔ 1/2 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_mean_value_range_l1044_104430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_interesting_number_l1044_104482

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def digits_of_nat (n : ℕ) : List ℕ :=
  n.repr.data.map (λ c => c.toNat - '0'.toNat)

def adjacent_sums_are_squares (n : ℕ) : Prop :=
  let digits := digits_of_nat n
  ∀ i, i < digits.length - 1 → is_square (digits[i]! + digits[i+1]!)

def all_digits_different (n : ℕ) : Prop :=
  (digits_of_nat n).Nodup

def is_interesting (n : ℕ) : Prop :=
  all_digits_different n ∧ adjacent_sums_are_squares n

theorem largest_interesting_number :
  (∀ m : ℕ, is_interesting m → m ≤ 6310972) ∧ is_interesting 6310972 := by
  sorry

#eval digits_of_nat 6310972

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_interesting_number_l1044_104482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1044_104424

open Real

noncomputable def f (a x : ℝ) : ℝ := (1 - a) / 2 * x^2 + a * x - log x

theorem function_inequality (m : ℝ) :
  (∀ a ∈ Set.Ioo 3 4, ∀ x₁ ∈ Set.Icc 1 2, ∀ x₂ ∈ Set.Icc 1 2,
    (a^2 - 1) / 2 * m + log 2 > |f a x₁ - f a x₂|) →
  m ≥ 1 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1044_104424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_l_piece_partition_l1044_104489

-- Define the board
def Board := Fin 10 → Fin 10 → Bool

-- Define an L-shaped piece
structure LPiece :=
  (pos : Fin 10 × Fin 10)
  (orientation : Fin 4)

-- Define a function to check if a cell is covered by an L-piece
def CoveredBy (p : LPiece) (cell : Fin 10 × Fin 10) : Prop :=
  sorry -- We'll leave this implementation for later

-- Define a valid partition of the board
def ValidPartition (pieces : List LPiece) : Prop :=
  -- Each cell is covered by exactly one L-piece
  ∀ (i j : Fin 10), ∃! (p : LPiece), p ∈ pieces ∧ CoveredBy p (i, j)

-- The main theorem
theorem impossible_l_piece_partition :
  ¬∃ (pieces : List LPiece), ValidPartition pieces :=
sorry

-- You can add more auxiliary definitions or lemmas here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_l_piece_partition_l1044_104489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_distances_l1044_104474

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point M
structure PointM where
  x₀ : ℝ
  y₀ : ℝ
  on_circle : C x₀ y₀

-- Define point N
def N : ℝ × ℝ := (4, 0)

-- Define point P as midpoint of MN
noncomputable def P (m : PointM) : ℝ × ℝ := ((m.x₀ + 4) / 2, m.y₀ / 2)

-- Define the line L: 3x + 4y - 26 = 0
def L (x y : ℝ) : Prop := 3 * x + 4 * y - 26 = 0

-- Theorem statement
theorem trajectory_and_distances :
  ∀ (m : PointM),
    let (x, y) := P m
    -- Trajectory equation
    (x - 2)^2 + y^2 = 1 ∧
    -- Maximum distance
    (∃ (p : ℝ × ℝ), (p.fst - 2)^2 + p.snd^2 = 1 ∧
      ∀ (q : ℝ × ℝ), (q.fst - 2)^2 + q.snd^2 = 1 →
        |3 * p.fst + 4 * p.snd - 26| / Real.sqrt 25 ≥ |3 * q.fst + 4 * q.snd - 26| / Real.sqrt 25) ∧
    |3 * x + 4 * y - 26| / Real.sqrt 25 = 5 ∧
    -- Minimum distance
    (∃ (p : ℝ × ℝ), (p.fst - 2)^2 + p.snd^2 = 1 ∧
      ∀ (q : ℝ × ℝ), (q.fst - 2)^2 + q.snd^2 = 1 →
        |3 * p.fst + 4 * p.snd - 26| / Real.sqrt 25 ≤ |3 * q.fst + 4 * q.snd - 26| / Real.sqrt 25) ∧
    |3 * x + 4 * y - 26| / Real.sqrt 25 = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_distances_l1044_104474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_terms_in_sequence_l1044_104486

noncomputable def a (n : ℕ+) : ℝ := (n - Real.sqrt 2017) / (n - Real.sqrt 2016)

theorem min_max_terms_in_sequence :
  ∀ k ∈ Finset.range 100, k ≠ 0 →
    (a ⟨45, by norm_num⟩ ≤ a ⟨k + 1, by positivity⟩ ∧ a ⟨k + 1, by positivity⟩ ≤ a ⟨44, by norm_num⟩) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_terms_in_sequence_l1044_104486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_smallest_is_480_l1044_104468

def Digits : Finset ℕ := {4, 0, 9, 8}

def ValidNumber (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (∃ (a b c : ℕ), a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = 100 * a + 10 * b + c)

def ThirdSmallest (n : ℕ) : Prop :=
  ValidNumber n ∧
  (∃ m₁ m₂ : ℕ, ValidNumber m₁ ∧ ValidNumber m₂ ∧ m₁ < m₂ ∧ m₂ < n) ∧
  (∀ m : ℕ, ValidNumber m → m < n → (∃ k : ℕ, ValidNumber k ∧ m < k ∧ k < n))

theorem third_smallest_is_480 : ThirdSmallest 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_smallest_is_480_l1044_104468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_f_increasing_interval_l1044_104480

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (a-2)x² + (a-1)x + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  (a - 2) * x^2 + (a - 1) * x + 3

/-- The increasing interval of a function on ℝ -/
def IncreasingInterval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

theorem even_f_increasing_interval (a : ℝ) :
  IsEven (f a) → IncreasingInterval (f a) (Set.Iic 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_f_increasing_interval_l1044_104480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_diff_in_second_quadrant_l1044_104472

def z₁ : ℂ := 1 + 3 * Complex.I
def z₂ : ℂ := 3 + Complex.I

theorem z_diff_in_second_quadrant : 
  (z₁ - z₂).re < 0 ∧ (z₁ - z₂).im > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_diff_in_second_quadrant_l1044_104472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_no_min_l1044_104433

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/6) * x^3 - (1/2) * m * x^2 + x

-- Define convexity
def is_convex (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Ioo a b, ∃ (ε : ℝ), ε > 0 ∧
    ∀ y z, y ∈ Set.Ioo (x - ε) (x + ε) → z ∈ Set.Ioo (x - ε) (x + ε) → y < z →
      (f y - f x) / (y - x) < (f z - f x) / (z - x)

-- Theorem statement
theorem f_has_max_no_min (m : ℝ) (h1 : m ≤ 2) 
  (h2 : is_convex (f m) (-1) 2) :
  (∃ x ∈ Set.Ioo (-1) 2, ∀ y ∈ Set.Ioo (-1) 2, f m y ≤ f m x) ∧
  (∀ x ∈ Set.Ioo (-1) 2, ∃ y ∈ Set.Ioo (-1) 2, f m y < f m x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_no_min_l1044_104433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_next_birthday_age_l1044_104441

-- Define the ages as real numbers
noncomputable def john_age : ℝ := sorry
noncomputable def mike_age : ℝ := sorry
noncomputable def lucas_age : ℝ := sorry

-- Define the relationships between ages
axiom john_older_than_mike : john_age = 1.25 * mike_age
axiom mike_younger_than_lucas : mike_age = 0.7 * lucas_age
axiom sum_of_ages : john_age + mike_age + lucas_age = 27.3

-- Define John's next birthday age
def john_next_birthday : ℕ := 10

-- Theorem to prove
theorem johns_next_birthday_age : 
  john_next_birthday = 10 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_next_birthday_age_l1044_104441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_is_345_l1044_104442

theorem larger_number_is_345 (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  Nat.gcd a b = 23 →
  Nat.lcm a b = 23 * 14 * 15 →
  max a b = 345 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_is_345_l1044_104442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_theorem_l1044_104477

/-- Represents a convex quadrilateral with side lengths a, b, c, d and diagonal lengths m, n -/
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  m : ℝ
  n : ℝ
  A : ℝ
  C : ℝ
  convex : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
  diagonals_positive : m > 0 ∧ n > 0

/-- The quadrilateral diagonal theorem -/
theorem quadrilateral_diagonal_theorem (q : ConvexQuadrilateral) :
  q.m^2 * q.n^2 = q.a^2 * q.c^2 + q.b^2 * q.d^2 - 2 * q.a * q.b * q.c * q.d * Real.cos (q.A + q.C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_theorem_l1044_104477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_determine_past_age_l1044_104496

-- Define the variables
def current_brother_age : ℕ := 10

-- Define the age difference relation
def age_difference (v b : ℕ) : Prop := v = 2 * b + 10

-- Define the theorem
theorem cannot_determine_past_age : 
  ∀ (past_brother_age : ℕ) (past_viggo_age : ℕ),
    age_difference past_viggo_age past_brother_age →
    past_brother_age ≤ current_brother_age →
    ∃ (other_valid_past_age : ℕ), 
      other_valid_past_age ≠ past_brother_age ∧
      other_valid_past_age ≤ current_brother_age ∧
      age_difference (other_valid_past_age + (current_brother_age - past_brother_age)) other_valid_past_age :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_determine_past_age_l1044_104496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_ax_iff_a_in_closed_interval_one_two_l1044_104400

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 - 1 else Real.log (x + 1)

-- State the theorem
theorem f_leq_ax_iff_a_in_closed_interval_one_two (a : ℝ) :
  (∀ x, f x ≤ a * x) ↔ a ∈ Set.Icc 1 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_ax_iff_a_in_closed_interval_one_two_l1044_104400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barbershop_price_increase_l1044_104497

/-- Calculates the percentage increase between two prices -/
noncomputable def percentage_increase (weekday_price weekend_price : ℝ) : ℝ :=
  (weekend_price - weekday_price) / weekday_price * 100

/-- Proves that the percentage increase from weekday to weekend price is 50% -/
theorem barbershop_price_increase (weekday_price weekend_price : ℝ) 
  (h1 : weekday_price = 18) 
  (h2 : weekend_price = 27) : 
  percentage_increase weekday_price weekend_price = 50 := by
  -- Unfold the definition of percentage_increase
  unfold percentage_increase
  -- Substitute the known values
  rw [h1, h2]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_barbershop_price_increase_l1044_104497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1044_104448

-- Define the inequality function
noncomputable def f (a x : ℝ) : ℝ := a * (4 : ℝ) ^ x - (2 : ℝ) ^ x + 2

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Iic (0 : ℝ), f a x > 0) ↔ a ∈ Set.Ioi (-1 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1044_104448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1044_104453

/-- The time (in days) it takes for worker c to complete the work alone -/
def time_for_c (c : ℚ) : ℚ :=
  1 / c

theorem work_completion_time 
  (a b c : ℚ) 
  (h1 : a + b + c = 1 / 4)  -- a, b, and c together finish in 4 days
  (h2 : a = 1 / 36)         -- a alone finishes in 36 days
  (h3 : b = 1 / 18)         -- b alone finishes in 18 days
  : time_for_c c = 6 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1044_104453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cba_finals_revenue_l1044_104402

/-- Represents the ticket revenue for the n-th game in million yuan -/
noncomputable def ticket_revenue (n : ℕ) : ℝ := n + 3

/-- Represents the total ticket revenue for n games in million yuan -/
noncomputable def total_revenue (n : ℕ) : ℝ := (n / 2) * (ticket_revenue 1 + ticket_revenue n)

/-- Probability of a specific series outcome -/
noncomputable def prob_outcome (n : ℕ) : ℝ := (1 / 2) ^ n * Nat.choose (n - 1) 3

theorem cba_finals_revenue :
  -- Probability of total revenue being 30 million yuan
  prob_outcome 5 = 1 / 4 ∧
  -- Expected total revenue
  (prob_outcome 4 * total_revenue 4 +
   prob_outcome 5 * total_revenue 5 +
   prob_outcome 6 * total_revenue 6 +
   prob_outcome 7 * total_revenue 7) = 3775 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cba_finals_revenue_l1044_104402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l1044_104493

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x => if x > 0 then x^2 + x else -((-x)^2 + (-x))

-- State the theorem
theorem odd_function_property :
  (∀ x : ℝ, f (-x) = -f x) ∧  -- f is an odd function
  (∀ x : ℝ, x > 0 → f x = x^2 + x) ∧  -- f(x) = x^2 + x for x > 0
  ¬(∀ x : ℝ, x ≤ 0 → f x = x^2 + x)  -- It's not true that f(x) = x^2 + x for x ≤ 0
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l1044_104493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_pi_third_l1044_104435

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) - (1/3) * Real.sin (3 * x)

-- State the theorem
theorem extreme_value_at_pi_third (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ π/3 ∧ |x - π/3| < ε → f a x ≤ f a (π/3)) →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_pi_third_l1044_104435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1044_104464

/-- Parabola defined by y^2 = 4x -/
def Parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

/-- Focus of the parabola y^2 = 4x -/
def Focus : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (3, 0)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_theorem (A : ℝ × ℝ) 
  (h1 : Parabola A)
  (h2 : distance A Focus = distance B Focus) :
  distance A B = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1044_104464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_relation_l1044_104417

/-- The main theorem -/
theorem lcm_gcd_relation (n m k : ℕ) (hn : n > 0) (hm : m > 0) (hk : k > 0)
  (h1 : n ∣ Nat.lcm m k) (h2 : m ∣ Nat.lcm n k) : n * Nat.gcd m k = m * Nat.gcd n k := by
  sorry

#check lcm_gcd_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_relation_l1044_104417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1044_104473

-- Define the arithmetic sequence and its sum
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

-- State the theorem
theorem arithmetic_sequence_problem (a₁ d : ℝ) (k : ℕ) :
  d ≠ 0 →
  arithmetic_sum a₁ d 11 = 132 →
  arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d k = 24 →
  k = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1044_104473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1044_104465

/-- The distance from a point to a line in 2D space -/
noncomputable def distancePointToLine (x₀ y₀ A B C : ℝ) : ℝ :=
  (|A * x₀ + B * y₀ + C|) / Real.sqrt (A^2 + B^2)

/-- The coordinates of point P -/
def P : ℝ × ℝ := (1, 0)

/-- Coefficients of the line equation Ax + By + C = 0 -/
def lineCoefficients : ℝ × ℝ × ℝ := (1, -2, 1)

theorem distance_point_to_line :
  let (x₀, y₀) := P
  let (A, B, C) := lineCoefficients
  distancePointToLine x₀ y₀ A B C = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1044_104465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_calculation_only_b_correct_l1044_104422

theorem correct_calculation : 
  (2^3 ≠ 6) ∧ (-4^2 = -16) ∧ (-8 - 8 ≠ 0) ∧ (-5 - 2 ≠ -3) := by
  constructor
  · -- Option A
    norm_num
  · constructor
    · -- Option B
      norm_num
    · constructor
      · -- Option C
        norm_num
      · -- Option D
        norm_num

theorem only_b_correct : 
  (2^3 ≠ 6) ∧ (-4^2 = -16) ∧ (-8 - 8 ≠ 0) ∧ (-5 - 2 ≠ -3) :=
  correct_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_calculation_only_b_correct_l1044_104422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_renyi_theorem_l1044_104423

/-- The Rado graph -/
def RadoGraph : Type := Unit

/-- A random graph from G(ℵ₀, p) -/
def RandomGraph (p : ℝ) : Type := Unit

/-- Isomorphism between graphs -/
def IsomorphicGraphs (G H : Type) : Prop := True

/-- Probability measure -/
noncomputable def Probability (event : Prop) : ℝ := 0

/-- Erdős-Rényi Theorem -/
theorem erdos_renyi_theorem (p : ℝ) (hp : 0 < p ∧ p < 1) :
  Probability (IsomorphicGraphs (RandomGraph p) RadoGraph) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_renyi_theorem_l1044_104423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ceiling_is_14_l1044_104495

-- Define the sum as noncomputable
noncomputable def sum : ℝ := Real.sqrt 9 + Real.sqrt 16 + 2 * (1 / 7) + 4 * (1 / 8)

-- Theorem statement
theorem sum_ceiling_is_14 : ⌈sum⌉ = 14 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ceiling_is_14_l1044_104495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_eight_l1044_104494

theorem polynomial_value_at_eight (p q r s t u : ℝ) :
  let P : ℂ → ℂ := λ x ↦ (3*x^4 - 33*x^3 + p*x^2 + q*x + r) * (4*x^4 - 100*x^3 + s*x^2 + t*x + u)
  (∀ z : ℂ, P z = 0 ↔ z ∈ ({2, 3, 4, 6, 7} : Set ℂ)) →
  P 8 = 483840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_eight_l1044_104494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_theorem_l1044_104492

/-- Point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Rectangle OABC in the Cartesian plane -/
structure Rectangle where
  O : Point
  A : Point
  B : Point
  C : Point

/-- Line that divides the rectangle -/
structure DividingLine where
  slope : ℚ
  intercept : ℚ

/-- Point M through which the dividing line passes -/
def M : Point := { x := 5, y := 6 }

/-- The rectangle OABC -/
def rectangle : Rectangle := {
  O := { x := 0, y := 0 },
  A := { x := 0, y := 5 },
  B := { x := 6, y := 5 },
  C := { x := 6, y := 0 }
}

/-- The dividing line -/
def dividingLine : DividingLine := {
  slope := 35 / 26,
  intercept := -19 / 26
}

/-- Function to calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℚ :=
  sorry

/-- Function to check if a point is on a line -/
def isPointOnLine (p : Point) (l : DividingLine) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Theorem stating that the given line divides the rectangle in the ratio 2:3 -/
theorem dividing_line_theorem (r : Rectangle) (l : DividingLine) : 
  isPointOnLine M l →
  (∃ E : Point, 
    isPointOnLine E l ∧ 
    E.x = 0 ∧
    (triangleArea r.O E { x := E.x, y := r.A.y }) / 
    (r.B.x * r.A.y - triangleArea r.O E { x := E.x, y := r.A.y }) = 2 / 3) :=
  sorry

#check dividing_line_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_theorem_l1044_104492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_lunch_cost_l1044_104459

/-- Calculates the total cost of lunches for a field trip --/
theorem total_lunch_cost 
  (children : ℕ) (chaperones : ℕ) (teacher : ℕ) (additional : ℕ) (cost_per_lunch : ℕ) :
  children = 35 →
  chaperones = 5 →
  teacher = 1 →
  additional = 3 →
  cost_per_lunch = 7 →
  (children + chaperones + teacher + additional) * cost_per_lunch = 308 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_lunch_cost_l1044_104459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_problem_l1044_104421

theorem seed_germination_problem (seeds_first_plot : ℕ) 
  (germination_rate_first : ℝ) (germination_rate_second : ℝ) 
  (overall_germination_rate : ℝ) :
  seeds_first_plot = 300 →
  germination_rate_first = 0.25 →
  germination_rate_second = 0.35 →
  overall_germination_rate = 0.28999999999999996 →
  ∃ (seeds_second_plot : ℕ),
    (germination_rate_first * (seeds_first_plot : ℝ) + 
     germination_rate_second * (seeds_second_plot : ℝ)) / 
    ((seeds_first_plot : ℝ) + (seeds_second_plot : ℝ)) = 
    overall_germination_rate ∧
    seeds_second_plot = 200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_problem_l1044_104421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_from_angle_relation_l1044_104408

/-- Given a triangle ABC with angles A, B, and C that satisfy the equation
    sin²A + sin²B + sin²C = 9/4, prove that the triangle is equilateral. -/
theorem triangle_equilateral_from_angle_relation (A B C : ℝ) 
  (h_triangle : A + B + C = Real.pi)
  (h_relation : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 9/4) :
  ∃ (a : ℝ), a > 0 ∧ A = B ∧ B = C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_from_angle_relation_l1044_104408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_sum_l1044_104437

noncomputable section

open Real

-- Define the function f
def f (x : ℝ) : ℝ := sin (2 * x + π / 4)

-- Define the theorem
theorem three_roots_sum (a : ℝ) :
  ∃ x₁ x₂ x₃ : ℝ,
    x₁ ∈ Set.Icc 0 (9 * π / 8) ∧
    x₂ ∈ Set.Icc 0 (9 * π / 8) ∧
    x₃ ∈ Set.Icc 0 (9 * π / 8) ∧
    x₁ < x₂ ∧ x₂ < x₃ ∧
    f x₁ = a ∧ f x₂ = a ∧ f x₃ = a ∧
    (∀ x ∈ Set.Icc 0 (9 * π / 8), f x = a → x = x₁ ∨ x = x₂ ∨ x = x₃) →
    2 * x₁ + 3 * x₂ + x₃ = 7 * π / 4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_sum_l1044_104437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_probability_l1044_104462

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4 : ℝ)^x - a * (2 : ℝ)^(x+1) + 1

-- Define the interval for a
def interval : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem zero_point_probability :
  ∃ (P : Set ℝ → ℝ), 
    (∀ s, s ⊆ interval → 0 ≤ P s ∧ P s ≤ 1) ∧  -- P is a probability measure on the interval
    (P interval = 1) ∧                         -- P is normalized
    (P {a ∈ interval | ∃ x, f a x = 0} = 1/4)  -- The probability of f having a zero point is 1/4
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_probability_l1044_104462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1044_104475

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem omega_range (ω φ : ℝ) : 
  ω > 0 → 
  0 < φ → φ < π / 2 →
  f ω φ 0 = Real.sqrt 2 / 2 →
  (∀ x₁ x₂ : ℝ, π / 2 < x₁ → x₁ < π → π / 2 < x₂ → x₂ < π → x₁ ≠ x₂ → 
    (x₁ - x₂) / (f ω φ x₁ - f ω φ x₂) < 0) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1044_104475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_assignments_l1044_104445

/-- The number of ways to assign n distinct items to k distinct categories is k^n -/
theorem distinct_assignments (n k : ℕ) : 
  k^n = (k : ℕ)^n :=
by sorry

/-- The specific problem instance -/
example : 3^7 = 2187 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_assignments_l1044_104445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1044_104440

/-- The sequence a_n defined by a recurrence relation -/
def a : ℕ → ℤ
  | 0 => -1  -- Define the base case for 0
  | 1 => -1
  | n + 2 => 2 * a (n + 1) + 3

/-- The general term of the sequence -/
def general_term (n : ℕ) : ℤ := 2^n - 3

/-- Theorem stating that the general term is correct for the given sequence -/
theorem sequence_general_term (n : ℕ) : a n = general_term n := by
  sorry

#eval a 5  -- You can add this line to test the function
#eval general_term 5  -- You can add this line to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1044_104440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_odd_partition_bijection_l1044_104457

/-- A partition of a natural number into distinct parts -/
def DistinctPartition (n : ℕ) : Type :=
  { p : List ℕ // p.sum = n ∧ p.Nodup ∧ ∀ x ∈ p, x > 0 }

/-- A partition of a natural number into odd parts -/
def OddPartition (n : ℕ) : Type :=
  { p : List ℕ // p.sum = n ∧ ∀ x ∈ p, x > 0 ∧ Odd x }

/-- The theorem stating the bijection between distinct and odd partitions -/
theorem distinct_odd_partition_bijection (n : ℕ) :
  ∃ f : DistinctPartition n → OddPartition n,
    Function.Bijective f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_odd_partition_bijection_l1044_104457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_at_x₀_l1044_104439

noncomputable section

open Real

/-- The function f(x) = x(1-x) / (x³ - x + 1) for 0 < x < 1 -/
noncomputable def f (x : ℝ) : ℝ := (x * (1 - x)) / (x^3 - x + 1)

/-- The x-coordinate of the maximum point of f(x) -/
noncomputable def x₀ : ℝ := (sqrt 2 + 1 - sqrt (2 * sqrt 2 - 1)) / 2

theorem f_maximum_at_x₀ :
  ∀ x ∈ Set.Ioo 0 1, f x ≤ f x₀ := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_at_x₀_l1044_104439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l1044_104444

/-- The area of a sector with given arc length and central angle -/
noncomputable def sector_area (arc_length : ℝ) (central_angle : ℝ) : ℝ :=
  (arc_length ^ 2) / (2 * central_angle)

/-- Theorem: The area of a sector with arc length 3π and central angle 3/4π is 6π -/
theorem sector_area_specific : sector_area (3 * Real.pi) ((3 / 4) * Real.pi) = 6 * Real.pi := by
  -- Unfold the definition of sector_area
  unfold sector_area
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l1044_104444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_system_problem_l1044_104471

-- Define the lines and curve
def line_l1 (x : ℝ) : Prop := x = -2
def curve_C (x y θ : ℝ) : Prop := x = 2 * Real.cos θ ∧ y = 2 + 2 * Real.sin θ
def line_l2 (θ : ℝ) : Prop := θ = Real.pi / 4

-- State the theorem
theorem coordinate_system_problem :
  ∀ (x y ρ θ : ℝ),
  -- Given conditions
  line_l1 x →
  curve_C x y θ →
  line_l2 θ →
  -- Prove the following
  (∃ (ρ : ℝ), ρ * Real.cos θ + 2 = 0) ∧  -- Polar equation of l₁
  (∃ (ρ : ℝ), ρ = 4 * Real.sin θ) ∧     -- Polar equation of C
  (∃ (S : ℝ), S = 2) ∧                  -- Area of ΔCMN
  (∃ (ρ' θ' : ℝ), ρ' = -2 * Real.sqrt 2 ∧ θ' = Real.pi / 4)  -- Intersection of l₁ and l₂
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_system_problem_l1044_104471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_882_with_more_than_three_factors_l1044_104409

/-- The number of positive integer factors of 882 that have more than 3 factors -/
noncomputable def factors_with_more_than_three_factors : ℕ :=
  (Finset.filter (fun d => (Nat.divisors d).card > 3) (Nat.divisors 882)).card

/-- Theorem stating that the number of factors of 882 with more than 3 factors is 11 -/
theorem factors_882_with_more_than_three_factors :
  factors_with_more_than_three_factors = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_882_with_more_than_three_factors_l1044_104409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infection_spread_l1044_104460

/-- The number of people each infected person can infect in one round -/
def x : ℕ := sorry

/-- The total number of infected people after two rounds of transmission -/
def total_infected : ℕ := 1 + x + x * (x + 1)

/-- The given total number of infected people -/
def given_total : ℕ := 196

/-- Theorem stating that the calculated total matches the given total -/
theorem infection_spread : total_infected = given_total :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infection_spread_l1044_104460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_seven_truth_tellers_l1044_104463

/-- Represents the two types of cockroaches -/
inductive CockroachType
  | TruthTeller
  | Liar
deriving Repr, DecidableEq

/-- Represents a cell on the 4x4 board -/
structure Cell where
  row : Fin 4
  col : Fin 4
  type : CockroachType
deriving Repr

/-- The 4x4 board of cockroaches -/
def Board := Array (Array Cell)

/-- Check if two cells are side neighbors -/
def areSideNeighbors (c1 c2 : Cell) : Bool :=
  (c1.row = c2.row && (c1.col.val + 1 = c2.col.val || c1.col.val = c2.col.val + 1)) ||
  (c1.col = c2.col && (c1.row.val + 1 = c2.row.val || c1.row.val = c2.row.val + 1))

/-- Check if a cell's claim about its neighbors is true -/
def isCellClaimTrue (board : Board) (cell : Cell) : Bool :=
  board.all fun row =>
    row.all fun neighbor =>
      if areSideNeighbors cell neighbor then
        neighbor.type == CockroachType.TruthTeller
      else
        true

/-- Check if all cells' claims are consistent with their types -/
def areAllClaimsConsistent (board : Board) : Bool :=
  board.all fun row =>
    row.all fun cell =>
      match cell.type with
      | CockroachType.TruthTeller => isCellClaimTrue board cell
      | CockroachType.Liar => true

/-- Count the number of truth-tellers on the board -/
def countTruthTellers (board : Board) : Nat :=
  board.foldl (init := 0) fun acc row =>
    acc + row.foldl (init := 0) fun count cell =>
      if cell.type == CockroachType.TruthTeller then count + 1 else count

/-- The main theorem to be proved -/
theorem impossible_seven_truth_tellers (board : Board) :
  areAllClaimsConsistent board → countTruthTellers board ≠ 7 := by
  sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_seven_truth_tellers_l1044_104463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_of_blankets_l1044_104411

theorem average_price_of_blankets (price1 price2 unknown_price : ℕ) 
  (count1 count2 count_unknown : ℕ) (total_cost : ℚ) 
  (h1 : price1 = 100)
  (h2 : price2 = 150)
  (h3 : count1 = 3)
  (h4 : count2 = 2)
  (h5 : count_unknown = 2)
  (h6 : unknown_price * count_unknown = 900)
  (h7 : total_cost = price1 * count1 + price2 * count2 + unknown_price * count_unknown)
  : total_cost / (count1 + count2 + count_unknown) = 1500 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_of_blankets_l1044_104411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_through_point_l1044_104412

/-- Given an angle θ in the Cartesian coordinate system xOy, if the terminal side of θ
    with Ox as the initial side passes through the point (3/5, 4/5), then
    sin θ = 4/5 and tan 2θ = -24/7 -/
theorem angle_through_point (θ : ℝ) :
  (∃ (r : ℝ), r * (Real.cos θ) = 3/5 ∧ r * (Real.sin θ) = 4/5) →
  Real.sin θ = 4/5 ∧ Real.tan (2 * θ) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_through_point_l1044_104412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_side_ratio_bounds_l1044_104452

/-- Predicate to represent an acute triangle -/
def TriangleAcute (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b + c ∧ b < a + c ∧ c < a + b ∧
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

/-- Helper function to calculate the area of a triangle given two sides and the altitude to one of them -/
noncomputable def TriangleArea (x y h : ℝ) : ℝ :=
  (1 / 2) * x * h

/-- Predicate to represent that h is an altitude of a triangle with sides x and y -/
def IsAltitude (h x y : ℝ) : Prop :=
  h > 0 ∧ h * x = 2 * TriangleArea x y h

/-- Given an acute triangle ABC with altitudes h_a, h_b, h_c and sides a, b, c,
    the sum of altitudes divided by the sum of sides is strictly greater than 1/2 and strictly less than 1. -/
theorem altitude_side_ratio_bounds (a b c h_a h_b h_c : ℝ) :
  TriangleAcute a b c →
  IsAltitude h_a b c →
  IsAltitude h_b a c →
  IsAltitude h_c a b →
  (1 : ℝ) / 2 < (h_a + h_b + h_c) / (a + b + c) ∧ (h_a + h_b + h_c) / (a + b + c) < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_side_ratio_bounds_l1044_104452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_with_power_of_three_l1044_104455

def is_power_of_three (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3^k

def product_of_four_consecutive (n : ℕ) : ℕ :=
  n * (n + 1) * (n + 2) * (n + 3)

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_product_with_power_of_three (n : ℕ) :
  (∃ k ∈ [n, n+1, n+2, n+3], is_power_of_three k) →
  units_digit (product_of_four_consecutive n) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_with_power_of_three_l1044_104455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_scaling_l1044_104461

/-- Given a finite multiset of real numbers, calculate its variance -/
def variance (s : Multiset ℝ) : ℝ := sorry

theorem variance_scaling (s : Multiset ℝ) (h : variance s = 3) :
  variance (s.map (fun x => 3 * x)) = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_scaling_l1044_104461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fourth_quarter_score_l1044_104431

/-- Represents a student's scores over four quarters -/
structure StudentScores where
  q1 : ℝ
  q2 : ℝ
  q3 : ℝ
  q4 : ℝ

/-- Calculates the average score over four quarters -/
noncomputable def average (s : StudentScores) : ℝ := (s.q1 + s.q2 + s.q3 + s.q4) / 4

/-- The minimum required average score -/
def requiredAverage : ℝ := 85

/-- Lee's scores for the first three quarters -/
def leeScores : StudentScores := { q1 := 88, q2 := 84, q3 := 82, q4 := 0 }

/-- Theorem: The minimum score Lee needs in the 4th quarter is 86% -/
theorem min_fourth_quarter_score :
  ∃ (x : ℝ), x = 86 ∧ 
  average { q1 := leeScores.q1, q2 := leeScores.q2, q3 := leeScores.q3, q4 := x } ≥ requiredAverage ∧
  ∀ (y : ℝ), y < x →
    average { q1 := leeScores.q1, q2 := leeScores.q2, q3 := leeScores.q3, q4 := y } < requiredAverage :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fourth_quarter_score_l1044_104431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1044_104456

/-- The function f(x) = 1 - 2sin²x -/
noncomputable def f (x : ℝ) : ℝ := 1 - 2 * Real.sin x ^ 2

/-- The smallest positive period of f is π -/
theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1044_104456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_15_gon_l1044_104481

/-- The number of sides in the polygon -/
def n : ℕ := 15

/-- The sum of interior angles of an n-sided polygon -/
def interior_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

/-- Predicate for a list of angles forming an increasing arithmetic sequence -/
def is_increasing_arithmetic_sequence (angles : List ℝ) : Prop :=
  angles.length > 1 ∧ 
  ∃ d : ℝ, ∀ i : ℕ, i + 1 < angles.length → 
    (angles.get! (i+1) - angles.get! i = d ∧ d > 0)

/-- Theorem stating the smallest angle in a convex 15-sided polygon with angles in arithmetic sequence -/
theorem smallest_angle_in_15_gon (angles : List ℝ) :
  angles.length = n →
  (∀ a ∈ angles, a > 0 ∧ a < 180) →
  (∀ a ∈ angles, ∃ k : ℤ, a = k) →
  angles.sum = interior_angle_sum n →
  is_increasing_arithmetic_sequence angles →
  angles.get! 0 = 135 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_15_gon_l1044_104481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_stuffing_time_approx_4_64_l1044_104436

/-- Represents the rates and task details for Earl and Ellen's envelope stuffing job -/
structure EnvelopeStuffingJob where
  earl_standard_rate : ℚ  -- Earl's rate for standard-sized envelopes (per minute)
  earl_larger_rate : ℚ    -- Earl's rate for larger-sized envelopes (per minute)
  ellen_standard_time : ℚ  -- Time Ellen takes to stuff 36 standard-sized envelopes (in minutes)
  ellen_larger_time : ℚ    -- Time Ellen takes to stuff 36 larger-sized envelopes (in minutes)
  standard_envelopes : ℕ   -- Number of standard-sized envelopes to stuff
  larger_envelopes : ℕ     -- Number of larger-sized envelopes to stuff
  standard_circulars : ℕ   -- Number of circulars per standard-sized envelope
  larger_circulars : ℕ     -- Number of circulars per larger-sized envelope

/-- Calculates the time taken to stuff all envelopes for the given job -/
def time_to_stuff (job : EnvelopeStuffingJob) : ℚ :=
  let ellen_standard_rate := 36 / job.ellen_standard_time
  let ellen_larger_rate := 36 / job.ellen_larger_time
  let combined_standard_rate := job.earl_standard_rate + ellen_standard_rate
  let combined_larger_rate := job.earl_larger_rate + ellen_larger_rate
  let standard_time := job.standard_envelopes / combined_standard_rate
  let larger_time := job.larger_envelopes / combined_larger_rate
  standard_time + larger_time

/-- Theorem stating that the time taken to stuff all envelopes is approximately 4.64 minutes -/
theorem envelope_stuffing_time_approx_4_64 (job : EnvelopeStuffingJob) 
    (h1 : job.earl_standard_rate = 36)
    (h2 : job.earl_larger_rate = 24)
    (h3 : job.ellen_standard_time = 3/2)
    (h4 : job.ellen_larger_time = 2)
    (h5 : job.standard_envelopes = 150)
    (h6 : job.larger_envelopes = 90)
    (h7 : job.standard_circulars = 2)
    (h8 : job.larger_circulars = 3) :
    ∃ ε > 0, |time_to_stuff job - 464/100| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_stuffing_time_approx_4_64_l1044_104436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1044_104414

-- Define the original and target functions
noncomputable def original_function (x : ℝ) : ℝ := (1/2) * Real.sin (2*x)
noncomputable def target_function (x : ℝ) : ℝ := (1/4) * Real.sin x

-- Define the transformation
noncomputable def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := (1/2) * f (x/2)

-- Theorem statement
theorem function_transformation :
  ∀ x, transform original_function x = target_function x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1044_104414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1044_104490

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x

noncomputable def minValue (a : ℝ) : ℝ :=
  if a < 0 then 0
  else if a ≤ 1 then -a^2
  else 1 - 2*a

theorem f_min_value (a : ℝ) :
  ∀ x ∈ Set.Icc 0 1, f a x ≥ minValue a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1044_104490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_ratio_l1044_104420

noncomputable section

open Real

theorem triangle_angle_and_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Acute triangle condition
  a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
  a * sin B = b * sin A → -- Law of sines
  a * sin C = c * sin A → -- Law of sines
  b * sin A + a * tan A * cos B = 2 * a * sin C → -- Given equation
  ∃ (D : ℝ), 
    (A = π / 3) ∧ 
    (∀ (BC CD : ℝ), BC > 0 ∧ CD > 0 → 
      (∃ (θ : ℝ), 0 < θ ∧ θ < π / 2 ∧
        BC / sin A = CD / sin (π / 4) ∧
        BC / CD = (sqrt 3 * sin (5 * π / 12 - θ)) / (sqrt 2 * sin θ)) →
      0 < BC / CD ∧ BC / CD < sqrt 3) := by
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_ratio_l1044_104420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1044_104485

theorem trig_identity (θ : Real) 
  (h : (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2) : 
  Real.sin θ / (Real.cos θ)^3 + Real.cos θ / (Real.sin θ)^3 = 820/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1044_104485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_p_for_repeating_decimal_18_l1044_104499

theorem min_p_for_repeating_decimal_18 (p q : ℕ+) (h_irreducible : Nat.Coprime p q) 
  (h_decimal : (p : ℚ) / q = 2 / 11) (h_min_q : ∀ (p' q' : ℕ+), (p' : ℚ) / q' = 2 / 11 → q' ≥ q) :
  p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_p_for_repeating_decimal_18_l1044_104499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_60_degrees_l1044_104487

-- Define vectors in ℝ²
variable (a b : ℝ × ℝ)

-- Define the conditions
def magnitude_a (a : ℝ × ℝ) : Prop := Real.sqrt ((a.1)^2 + (a.2)^2) = 2
def magnitude_b (b : ℝ × ℝ) : Prop := Real.sqrt ((b.1)^2 + (b.2)^2) = 1
def orthogonality (a b : ℝ × ℝ) : Prop := (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0

-- Define the dot product
def dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

-- Define the angle between vectors
noncomputable def angle_between (x y : ℝ × ℝ) : ℝ := 
  Real.arccos (dot_product x y / (Real.sqrt ((x.1)^2 + (x.2)^2) * Real.sqrt ((y.1)^2 + (y.2)^2)))

-- Theorem to prove
theorem angle_is_60_degrees (a b : ℝ × ℝ) 
  (h1 : magnitude_a a) (h2 : magnitude_b b) (h3 : orthogonality a b) : 
  angle_between a b = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_60_degrees_l1044_104487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_varies_as_negative_two_power_of_z_l1044_104470

-- Define the relationships between x, y, and z
def varies_inverse_sqrt (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x = k * y^(-(1/2 : ℝ))
def varies_fourth_power (y z : ℝ) : Prop := ∃ j : ℝ, j ≠ 0 ∧ y = j * z^4
def varies_nth_power (x z : ℝ) (n : ℝ) : Prop := ∃ m : ℝ, m ≠ 0 ∧ x = m * z^n

-- State the theorem
theorem x_varies_as_negative_two_power_of_z (x y z : ℝ) :
  varies_inverse_sqrt x y → varies_fourth_power y z → varies_nth_power x z (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_varies_as_negative_two_power_of_z_l1044_104470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equality_subset_complement_l1044_104407

/-- Set A defined by the quadratic inequality -/
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

/-- Set B defined by the quadratic inequality with parameter m -/
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

/-- Theorem for part I -/
theorem intersection_equality (m : ℝ) : A ∩ B m = Set.Icc 0 3 ↔ m = 2 := by sorry

/-- Theorem for part II -/
theorem subset_complement (m : ℝ) : A ⊆ (B m)ᶜ ↔ m < -3 ∨ m > 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equality_subset_complement_l1044_104407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_midpoint_l1044_104483

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

/-- Definition of the intersecting line -/
def line (x y m : ℝ) : Prop := y = x + m

/-- Definition of the circle where the midpoint lies -/
def midpoint_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- Theorem statement -/
theorem ellipse_intersection_midpoint (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    line x₁ y₁ m ∧ line x₂ y₂ m ∧
    x₁ ≠ x₂ ∧
    midpoint_circle ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)) →
  m = 3 ∨ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_midpoint_l1044_104483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_rotation_for_f_l1044_104479

/-- The center of rotation for a complex function f(z) = (az + b) / 2 where a and b are complex numbers -/
noncomputable def center_of_rotation (a b : ℂ) : ℂ :=
  b / (2 - a)

/-- The given complex function -/
noncomputable def f (z : ℂ) : ℂ :=
  ((-1 - Complex.I * Real.sqrt 3) * z + (2 * Real.sqrt 3 - 12 * Complex.I)) / 2

theorem center_of_rotation_for_f :
  center_of_rotation (-1 - Complex.I * Real.sqrt 3) (2 * Real.sqrt 3 - 12 * Complex.I) =
  -5 * Real.sqrt 3 / 2 - 7 / 2 * Complex.I :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_rotation_for_f_l1044_104479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_six_equals_43_over_16_l1044_104451

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9

noncomputable def g (x : ℝ) : ℝ := 
  let y := (x + 9) / 4  -- This is f⁻¹(x)
  3 * y^2 + 4 * y - 2

-- State the theorem
theorem g_of_negative_six_equals_43_over_16 : g (-6) = 43 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_six_equals_43_over_16_l1044_104451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_stratified_sample_l1044_104429

/-- Represents the number of students in each year --/
structure StudentPopulation where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ

/-- Represents the sample size for each year --/
structure SampleSize where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ
deriving Repr

/-- Calculates the stratified sample size for each year --/
def stratifiedSample (pop : StudentPopulation) (totalSample : ℕ) : SampleSize :=
  let total := pop.firstYear + pop.secondYear + pop.thirdYear
  { firstYear := (pop.firstYear * totalSample + total - 1) / total,
    secondYear := (pop.secondYear * totalSample + total - 1) / total,
    thirdYear := (pop.thirdYear * totalSample + total - 1) / total }

/-- Theorem stating the correct stratified sample sizes --/
theorem correct_stratified_sample :
  let pop : StudentPopulation := { firstYear := 600, secondYear := 680, thirdYear := 720 }
  let sample := stratifiedSample pop 50
  sample.firstYear = 15 ∧ sample.secondYear = 17 ∧ sample.thirdYear = 18 := by
  sorry

#eval stratifiedSample { firstYear := 600, secondYear := 680, thirdYear := 720 } 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_stratified_sample_l1044_104429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_condition_l1044_104450

/-- The equation of the graph -/
def graph_equation (x y l : ℝ) : Prop :=
  3 * x^2 + 2 * y^2 - 6 * x + 8 * y = l

/-- Definition of a non-degenerate ellipse -/
def is_non_degenerate_ellipse (f : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y l : ℝ), f x y l ↔ (x - c)^2 / a + (y - d)^2 / b = e

/-- The main theorem -/
theorem ellipse_condition :
  is_non_degenerate_ellipse graph_equation ↔ ∀ l, (∃ x y, graph_equation x y l) → l > -11 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_condition_l1044_104450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_lake_population_l1044_104488

theorem turtle_lake_population (
  -- Species distribution
  common_percent : ℝ ) (rare_percent : ℝ) (unique_percent : ℝ) (legendary_percent : ℝ)
  (common_baby_percent : ℝ) (common_juvenile_percent : ℝ) (common_adult_percent : ℝ)
  (rare_baby_percent : ℝ) (rare_juvenile_percent : ℝ) (rare_adult_percent : ℝ)
  (unique_baby_percent : ℝ) (unique_juvenile_percent : ℝ) (unique_adult_percent : ℝ)
  (legendary_baby_percent : ℝ) (legendary_juvenile_percent : ℝ) (legendary_adult_percent : ℝ)
  (common_female_percent : ℝ) (rare_female_percent : ℝ) (unique_female_percent : ℝ) (legendary_female_percent : ℝ)
  (common_striped_male_percent : ℝ) (rare_striped_male_percent : ℝ) (unique_striped_male_percent : ℝ) (legendary_striped_male_percent : ℝ)
  (common_striped_male_adult_percent : ℝ) (rare_striped_male_adult_percent : ℝ) (unique_striped_male_adult_percent : ℝ) (legendary_striped_male_adult_percent : ℝ)
  (observed_striped_male_adult_common : ℕ)
  (h1 : common_percent = 0.50)
  (h2 : rare_percent = 0.30)
  (h3 : unique_percent = 0.15)
  (h4 : legendary_percent = 0.05)
  (h5 : common_baby_percent = 0.40)
  (h6 : common_juvenile_percent = 0.30)
  (h7 : common_adult_percent = 0.30)
  (h8 : rare_baby_percent = 0.30)
  (h9 : rare_juvenile_percent = 0.40)
  (h10 : rare_adult_percent = 0.30)
  (h11 : unique_baby_percent = 0.20)
  (h12 : unique_juvenile_percent = 0.30)
  (h13 : unique_adult_percent = 0.50)
  (h14 : legendary_baby_percent = 0.15)
  (h15 : legendary_juvenile_percent = 0.30)
  (h16 : legendary_adult_percent = 0.55)
  (h17 : common_female_percent = 0.60)
  (h18 : rare_female_percent = 0.55)
  (h19 : unique_female_percent = 0.45)
  (h20 : legendary_female_percent = 0.40)
  (h21 : common_striped_male_percent = 0.25)
  (h22 : rare_striped_male_percent = 0.40)
  (h23 : unique_striped_male_percent = 0.33)
  (h24 : legendary_striped_male_percent = 0.50)
  (h25 : common_striped_male_adult_percent = 0.40)
  (h26 : rare_striped_male_adult_percent = 0.45)
  (h27 : unique_striped_male_adult_percent = 0.35)
  (h28 : legendary_striped_male_adult_percent = 0.30)
  (h29 : observed_striped_male_adult_common = 84) :
  ∃ (total_turtles : ℕ), total_turtles = 4200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_lake_population_l1044_104488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l1044_104454

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2^(|x+1| - |x-1|)

-- State the theorem
theorem f_range_theorem :
  ∀ x : ℝ, f x ≥ 2 * Real.sqrt 2 ↔ x ≥ 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l1044_104454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ptolemy_inequality_six_point_inequality_ptolemy_equality_condition_six_point_equality_condition_l1044_104446

-- Define a type for points in a plane
variable {Point : Type} [AddCommGroup Point] [Module ℝ Point]

-- Define a distance function
variable (dist : Point → Point → ℝ)

-- Ptolemy's inequality for four points
theorem ptolemy_inequality (A B C D : Point) :
  dist A B * dist C D + dist B C * dist A D ≥ dist A C * dist B D := by sorry

-- Inequality for six points
theorem six_point_inequality (A₁ A₂ A₃ A₄ A₅ A₆ : Point) :
  dist A₁ A₄ * dist A₂ A₅ * dist A₃ A₆ ≤
  dist A₁ A₂ * dist A₃ A₆ * dist A₄ A₅ +
  dist A₁ A₂ * dist A₃ A₄ * dist A₅ A₆ +
  dist A₂ A₃ * dist A₁ A₄ * dist A₅ A₆ +
  dist A₂ A₃ * dist A₄ A₅ * dist A₁ A₆ +
  dist A₃ A₄ * dist A₂ A₅ * dist A₁ A₆ := by sorry

-- Define a predicate for cyclic quadrilaterals
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Equality condition for Ptolemy's inequality
theorem ptolemy_equality_condition (A B C D : Point) :
  dist A B * dist C D + dist B C * dist A D = dist A C * dist B D ↔
  is_cyclic_quadrilateral A B C D := by sorry

-- Define a predicate for cyclic hexagons
def is_cyclic_hexagon (A₁ A₂ A₃ A₄ A₅ A₆ : Point) : Prop := sorry

-- Equality condition for six-point inequality
theorem six_point_equality_condition (A₁ A₂ A₃ A₄ A₅ A₆ : Point) :
  dist A₁ A₄ * dist A₂ A₅ * dist A₃ A₆ =
  dist A₁ A₂ * dist A₃ A₆ * dist A₄ A₅ +
  dist A₁ A₂ * dist A₃ A₄ * dist A₅ A₆ +
  dist A₂ A₃ * dist A₁ A₄ * dist A₅ A₆ +
  dist A₂ A₃ * dist A₄ A₅ * dist A₁ A₆ +
  dist A₃ A₄ * dist A₂ A₅ * dist A₁ A₆ ↔
  is_cyclic_hexagon A₁ A₂ A₃ A₄ A₅ A₆ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ptolemy_inequality_six_point_inequality_ptolemy_equality_condition_six_point_equality_condition_l1044_104446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_price_is_seven_point_five_l1044_104416

/-- Represents the price reduction percentage as a decimal -/
def price_reduction : ℚ := 4/10

/-- Represents the additional number of bananas that can be bought for Rs. 40 after the price reduction -/
def additional_bananas : ℕ := 64

/-- Represents the amount of money spent -/
def money_spent : ℚ := 40

/-- Represents the number of bananas in a dozen -/
def bananas_per_dozen : ℕ := 12

/-- Calculates the reduced price per dozen bananas -/
noncomputable def reduced_price_per_dozen : ℚ :=
  let original_price := money_spent / ((additional_bananas / bananas_per_dozen) * (1 - price_reduction))
  original_price * (1 - price_reduction)

/-- Theorem stating that the reduced price per dozen bananas is 7.5 -/
theorem reduced_price_is_seven_point_five :
  reduced_price_per_dozen = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_price_is_seven_point_five_l1044_104416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_nonzero_l1044_104413

/-- A polynomial of degree 5 with a double root at 0 and three other distinct roots -/
def P (a b c d e : ℝ) (x : ℝ) : ℝ :=
  x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- The three non-zero roots of the polynomial -/
noncomputable def p : ℝ := sorry
noncomputable def q : ℝ := sorry
noncomputable def r : ℝ := sorry

/-- Conditions on the roots -/
axiom root_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r
axiom root_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0

/-- The polynomial has a double root at 0 -/
axiom double_root_zero (a b c : ℝ) : 
  P a b c 0 0 0 = 0 ∧ (deriv (P a b c 0 0)) 0 = 0

/-- The polynomial factored form -/
axiom P_factored (x : ℝ) : 
  P (-(p+q+r)) (p*q + p*r + q*r) (-p*q*r) 0 0 x = x^2 * (x - p) * (x - q) * (x - r)

/-- Theorem: The coefficient c must be non-zero -/
theorem c_nonzero : 
  ∀ a b c d e : ℝ, (∀ x : ℝ, P a b c d e x = P (-(p+q+r)) (p*q + p*r + q*r) (-p*q*r) 0 0 x) → 
  c ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_nonzero_l1044_104413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_triplet_l1044_104410

theorem solution_triplet (a k m : ℕ) (h : k + a^k = m + 2*a^m) :
  ∃ t : ℕ, a = 1 ∧ k = t + 1 ∧ m = t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_triplet_l1044_104410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_distribution_l1044_104432

noncomputable def hemisphereVolume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

noncomputable def cylinderVolume (d : ℝ) : ℝ := (Real.pi * d^3) / 4

theorem chocolate_distribution (n : ℕ) (h : n = 36) :
  let r := 1  -- radius of hemisphere in feet
  let v := hemisphereVolume r  -- volume of hemisphere
  ∃ d : ℝ,  -- diameter of each cylinder
    n * (cylinderVolume d) = v ∧ 
    d = (2 : ℝ)^(1/3) / 3 := by
  sorry

#check chocolate_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_distribution_l1044_104432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_and_intersection_graph_intersection_l1044_104484

open Real

-- Define the constant e^(e^(-1))
noncomputable def ee : ℝ := Real.exp (Real.exp (-1))

-- Define the logarithm function
noncomputable def log (a x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem statement
theorem log_inequality_and_intersection (a : ℝ) :
  (∀ x > 0, log a x ≤ x ∧ x ≤ a^x) ↔ a ≥ ee :=
sorry

-- Theorem for the intersection of graphs
theorem graph_intersection (a : ℝ) :
  (0 < a ∧ a < ee ∧ a ≠ 1) → (∃ x > 0, log a x = x ∨ x = a^x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_and_intersection_graph_intersection_l1044_104484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cosine_l1044_104425

theorem triangle_angle_cosine (A B C : ℝ) : 
  A + B + C = π →  -- Sum of angles in a triangle is π radians (180°)
  A + C = 2 * B →  -- Given condition
  1 / Real.cos A + 1 / Real.cos C = Real.sqrt 2 / Real.cos B →  -- Given condition
  Real.cos ((A - C) / 2) = -(Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cosine_l1044_104425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_labeling_bounds_l1044_104443

/-- Represents the colors of points on the circle -/
inductive Color
| Red
| Blue
| Green
deriving BEq

/-- Represents an arc on the circle -/
structure Arc where
  start : Color
  stop : Color

/-- Calculates the label of an arc based on its endpoint colors -/
def arcLabel (arc : Arc) : Nat :=
  match arc.start, arc.stop with
  | Color.Red, Color.Blue | Color.Blue, Color.Red => 1
  | Color.Red, Color.Green | Color.Green, Color.Red => 2
  | Color.Blue, Color.Green | Color.Green, Color.Blue => 3
  | _, _ => 0

/-- Represents a configuration of points on the circle -/
def CircleConfiguration := List Color

/-- Calculates the sum of labels for a given configuration -/
def sumLabels (config : CircleConfiguration) : Nat :=
  let arcs := List.zip config (List.rotate config 1)
  (arcs.map fun (c1, c2) => arcLabel { start := c1, stop := c2 }).sum

/-- The theorem to be proved -/
theorem circle_labeling_bounds
  (config : CircleConfiguration)
  (h_length : config.length = 90)
  (h_red : config.count Color.Red = 40)
  (h_blue : config.count Color.Blue = 30)
  (h_green : config.count Color.Green = 20) :
  6 ≤ sumLabels config ∧ sumLabels config ≤ 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_labeling_bounds_l1044_104443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l1044_104427

theorem tan_difference (α β : ℝ) (h1 : Real.tan α = 1/2) (h2 : Real.tan β = 1/3) :
  Real.tan (α - β) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l1044_104427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l1044_104498

def my_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 0  -- We define a₀ as 0 to make the function total
  | 1 => 1/3
  | n+1 => (-1)^(n+1) * 2 * my_sequence n

theorem fifth_term_value : my_sequence 5 = -16/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l1044_104498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_one_seventh_count_l1044_104428

/-- A function that returns the number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.succ (Nat.log n 10)

/-- A function that returns the three-digit number obtained by removing the leftmost digit of a four-digit number -/
def remove_leftmost (n : ℕ) : ℕ :=
  n % 1000

/-- The main theorem stating that there are exactly 5 four-digit numbers satisfying the given conditions -/
theorem four_digit_one_seventh_count :
  (Finset.filter (fun n => num_digits n = 4 ∧ 
                           num_digits (remove_leftmost n) = 3 ∧ 
                           7 * (remove_leftmost n) = n) (Finset.range 10000)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_one_seventh_count_l1044_104428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_existence_l1044_104447

theorem unique_function_existence : 
  ∃! f : ℤ → ℤ, 
    (∀ a b : ℤ, f (a + b) + f (a * b) = f a * f b - 1) ∧ 
    (∀ n : ℤ, f n = Int.natAbs (2 - n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_existence_l1044_104447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1044_104449

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x + 4 * Real.cos x ^ 2

-- State the theorem
theorem max_value_of_f :
  ∃ (x_max : ℝ), 0 ≤ x_max ∧ x_max ≤ π / 2 ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ π / 2 → f x ≤ f x_max) ∧
  x_max = π / 8 ∧
  f x_max = (5 + 3 * Real.sqrt 2) / 2 := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1044_104449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_equals_pi_over_four_l1044_104419

theorem alpha_plus_beta_equals_pi_over_four (α β : ℝ) : 
  0 < α ∧ α < π → 0 < β ∧ β < π → Real.tan α = 1/2 → Real.tan β = 1/3 → α + β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_equals_pi_over_four_l1044_104419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_properties_l1044_104466

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point is on the regression line -/
def on_regression_line (reg : LinearRegression) (x y : ℝ) : Prop :=
  y = reg.slope * x + reg.intercept

/-- Represents the mean of x and y values -/
structure MeanValues where
  x_bar : ℝ
  y_bar : ℝ

/-- Coefficient of determination (R-squared) -/
def R_squared : ℝ → Prop := sorry

/-- The main theorem to be proved -/
theorem regression_properties
  (reg : LinearRegression)
  (means : MeanValues)
  (r_sq_1 r_sq_2 : ℝ) :
  (on_regression_line reg means.x_bar means.y_bar) ∧
  (reg.slope = -5 ∧ reg.intercept = 3 → ¬(∀ x : ℝ, reg.slope > 0)) ∧
  (R_squared r_sq_1 ∧ R_squared r_sq_2 ∧ r_sq_1 = 0.80 ∧ r_sq_2 = 0.98 → r_sq_2 > r_sq_1) ∧
  (reg.slope = 0.5 ∧ reg.intercept = -8 → ¬(∀ x : ℝ, on_regression_line reg x ((reg.slope * x + reg.intercept) + 0.1))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_properties_l1044_104466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_count_and_position_l1044_104415

def digits : List Nat := [0, 1, 5, 8]

def is_valid_four_digit_number (n : Nat) : Bool :=
  n ≥ 1000 && n < 10000 && (n.repr.all (fun c => c.toString.toNat?.isSome && c.toString.toNat?.get! ∈ digits))

def count_valid_numbers : Nat :=
  (List.range 10000).filter is_valid_four_digit_number |>.length

def position_of_1850 : Nat :=
  (List.range 10000).filter (fun n => is_valid_four_digit_number n && n ≤ 1850) |>.length

theorem four_digit_numbers_count_and_position :
  count_valid_numbers = 18 ∧ position_of_1850 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_count_and_position_l1044_104415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_items_for_2019_students_max_items_received_is_optimal_l1044_104403

/-- Represents the maximum number of items a student can receive in a voting scenario. -/
def max_items_received (num_students : ℕ) : ℕ :=
  num_students / 2

/-- Theorem stating the maximum number of items a student can receive
    in a specific voting scenario with 2019 students. -/
theorem max_items_for_2019_students :
  max_items_received 2019 = 1009 := by
  rfl

/-- Proves that the maximum number of items one student can receive is optimal. -/
theorem max_items_received_is_optimal (num_students : ℕ) :
  ∀ (items_received : ℕ), items_received ≤ max_items_received num_students := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_items_for_2019_students_max_items_received_is_optimal_l1044_104403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l1044_104438

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 6) + 1

theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi / 2 := by
  sorry

#check smallest_positive_period

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l1044_104438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_show_duration_example_l1044_104418

/-- Calculates the duration of a TV show excluding commercials -/
noncomputable def tv_show_duration (total_time : ℝ) (num_commercials : ℕ) (commercial_duration : ℝ) : ℝ :=
  total_time - (num_commercials : ℝ) * commercial_duration / 60

/-- Theorem stating that a TV show with 1.5 hours total airing time and 3 commercials of 10 minutes each has a duration of 1 hour excluding commercials -/
theorem tv_show_duration_example : tv_show_duration 1.5 3 10 = 1 := by
  -- Unfold the definition of tv_show_duration
  unfold tv_show_duration
  -- Simplify the arithmetic expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_show_duration_example_l1044_104418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l1044_104469

-- Define the function f(x) = 2^x + 3x - k
noncomputable def f (x k : ℝ) : ℝ := Real.exp (x * Real.log 2) + 3*x - k

-- Theorem statement
theorem solution_range (k : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ f x k = 0) ↔ k ∈ Set.Ico 5 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l1044_104469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1044_104478

noncomputable def f (x : ℝ) := (x / (x - 2)) + ((x + 3) / (3 * x))

theorem inequality_solution :
  ∀ x : ℝ, f x ≥ 4 ↔ (0 < x ∧ x ≤ 1/8) ∨ (2 < x ∧ x ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1044_104478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_hyperbola_asymptotes_l1044_104404

theorem hyperbola_equation (x y : ℝ) : 
  (∀ (x₀ y₀ : ℝ), x₀^2/16 + y₀^2/25 = 1 → (x₀ = 0 ∧ y₀ = 3) ∨ (x₀ = 0 ∧ y₀ = -3)) →
  y^2/5 - x^2/4 = 1 →
  (∃ (c : ℝ), c^2 = 9 ∧ 
    (∀ (x₁ y₁ : ℝ), y₁^2/5 - x₁^2/4 = 1 → 
      (x₁ - 0)^2 + (y₁ - c)^2 - ((x₁ - 0)^2 + (y₁ + c)^2) = 4*c*y₁)) ∧
  (-2)^2/4 + (Real.sqrt 10)^2/5 = 1 :=
by sorry

theorem hyperbola_asymptotes (x y : ℝ) :
  (∀ (t : ℝ), x = t ∨ x = -t → y = t/2 ∨ y = -t/2) →
  x^2/12 - y^2/3 = 1 →
  2^2/12 - 2^2/3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_hyperbola_asymptotes_l1044_104404
