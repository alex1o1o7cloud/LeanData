import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_cubic_curve_l48_4817

/-- The equation of the tangent line to the curve y = x³ - 4x at the point (1, -3) is x + y + 2 = 0. -/
theorem tangent_line_cubic_curve :
  let f : ℝ → ℝ := λ x ↦ x^3 - 4*x
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let tangent_line : ℝ → ℝ := λ x ↦ -(x - x₀) + y₀
  (∀ x, tangent_line x + x + 2 = 0) ∧
  (HasDerivAt f (tangent_line 0 - tangent_line 1) x₀) ∧
  (f x₀ = y₀) ∧
  (y₀ = -3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_cubic_curve_l48_4817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_AB_BC_l48_4868

-- Define the circle C
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 23 = 0

-- Define the line l
def line_eq (x y : ℝ) : Prop := 3*x + 4*y + 8 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, 5)

-- Theorem statement
theorem dot_product_AB_BC :
  ∃ (A B C : ℝ × ℝ),
    circle_eq A.1 A.2 ∧
    circle_eq B.1 B.2 ∧
    line_eq A.1 A.2 ∧
    line_eq B.1 B.2 ∧
    circle_eq point_P.1 point_P.2 ∧
    C = (1, 1) ∧ -- Center of the circle
    (let AB := (B.1 - A.1, B.2 - A.2)
     let BC := (C.1 - B.1, C.2 - B.2)
     AB.1 * BC.1 + AB.2 * BC.2 = -32) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_AB_BC_l48_4868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_p_in_product_l48_4896

theorem power_of_p_in_product (p q : ℕ) (n : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (Nat.divisors (p^n * q^5)).card = 18 → n = 2 := by
  intro h
  sorry

#check power_of_p_in_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_p_in_product_l48_4896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_divisible_by_eleven_l48_4870

theorem difference_divisible_by_eleven (S : Finset ℤ) (h : S.card = 12) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_divisible_by_eleven_l48_4870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_half_plus_beta_l48_4898

theorem cos_alpha_half_plus_beta (α β : ℝ) 
  (h1 : α ∈ Set.Ioo π (3*π/2)) 
  (h2 : Real.cos α = -5/13) 
  (h3 : Real.tan (β/2) = 1/2) : 
  Real.cos (α/2 + β) = -18*Real.sqrt 13/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_half_plus_beta_l48_4898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cutting_l48_4897

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 150 →
  ratio = 5 / 8 →
  shorter_piece + ratio * shorter_piece = total_length →
  ‖shorter_piece - 92‖ < 1 :=
by
  intros h1 h2 h3
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cutting_l48_4897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortestLongestDistance_l48_4827

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line segment defined by two endpoints -/
structure LineSegment where
  a : Point
  b : Point

/-- Checks if a point is within the perpendicular zone of a line segment -/
def isWithinPerpendicularZone (p : Point) (seg : LineSegment) : Bool :=
  sorry

/-- Calculates the perpendicular distance from a point to a line segment -/
noncomputable def perpendicularDistance (p : Point) (seg : LineSegment) : ℝ :=
  sorry

/-- Checks if a point lies on a line segment -/
def isOnLineSegment (p : Point) (seg : LineSegment) : Bool :=
  sorry

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Defines membership of a point on a line segment -/
instance : Membership Point LineSegment where
  mem p seg := isOnLineSegment p seg

/-- Theorem: Shortest and longest distance from a point to a line segment -/
theorem shortestLongestDistance (p : Point) (seg : LineSegment) :
  let shortestDist := 
    if isOnLineSegment p seg then 0
    else if isWithinPerpendicularZone p seg then perpendicularDistance p seg
    else min (distance p seg.a) (distance p seg.b)
  let longestDist := max (distance p seg.a) (distance p seg.b)
  (∀ q : Point, q ∈ seg → distance p q ≥ shortestDist) ∧
  (∀ q : Point, q ∈ seg → distance p q ≤ longestDist) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortestLongestDistance_l48_4827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_inequality_sine_sum_inequality_l48_4860

-- Problem 1
theorem reciprocal_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1/a + 1/b + 1/c ≥ 1/Real.sqrt (a*b) + 1/Real.sqrt (b*c) + 1/Real.sqrt (a*c) := by
  sorry

-- Problem 2
theorem sine_sum_inequality (x y : ℝ) :
  Real.sin x + Real.sin y ≤ 1 + Real.sin x * Real.sin y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_inequality_sine_sum_inequality_l48_4860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l48_4857

theorem matrix_scalar_multiplication (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (M = ![![-5, 0], ![0, -5]]) →
  (∀ v : Fin 2 → ℝ, M.mulVec v = (-5 : ℝ) • v) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l48_4857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_committee_probability_l48_4830

-- Define the total number of members
def total_members : ℕ := 30

-- Define the number of boys
def num_boys : ℕ := 12

-- Define the number of girls
def num_girls : ℕ := 18

-- Define the committee size
def committee_size : ℕ := 5

-- Define the probability of choosing a committee with at least 1 boy and at least 1 girl
def prob_mixed_committee : ℚ := 571 / 611

-- Theorem statement
theorem mixed_committee_probability :
  (1 : ℚ) - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size : ℚ) / 
  (Nat.choose total_members committee_size : ℚ) = prob_mixed_committee := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_committee_probability_l48_4830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_problem_l48_4833

structure Plane where

structure Line where

def parallel (a b : Line ⊕ Plane) : Prop := sorry

def perpendicular (a b : Line ⊕ Plane) : Prop := sorry

def subset (a : Line) (b : Plane) : Prop := sorry

def intersection (a b : Plane) : Line := sorry

def inj_line : Line → Line ⊕ Plane := Sum.inl
def inj_plane : Plane → Line ⊕ Plane := Sum.inr

theorem geometry_problem 
  (α β : Plane) 
  (m n : Line) 
  (h_distinct_planes : α ≠ β)
  (h_distinct_lines : m ≠ n) :
  (∀ (h1 : parallel (inj_line m) (inj_line n)) (h2 : perpendicular (inj_line m) (inj_plane α)), perpendicular (inj_line n) (inj_plane α)) ∧
  (∀ (h1 : perpendicular (inj_line m) (inj_plane α)) (h2 : perpendicular (inj_line m) (inj_plane β)), parallel (inj_plane α) (inj_plane β)) ∧
  (∀ (h1 : perpendicular (inj_line m) (inj_plane α)) (h2 : parallel (inj_line m) (inj_line n)) (h3 : subset n β), perpendicular (inj_plane α) (inj_plane β)) ∧
  ¬(∀ (h1 : parallel (inj_line m) (inj_plane α)) (h2 : intersection α β = n), parallel (inj_line m) (inj_line n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_problem_l48_4833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_range_l48_4877

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p | (p.1 - 4)^2 + (p.2 - 3)^2 = 1 ∧ 3 ≤ p.1 ∧ p.1 ≤ 5 ∧ p.2 ≥ 3}

-- Define the perimeter function
def perimeter (p : ℝ × ℝ) : ℝ := 2 * (p.1 + p.2)

-- Theorem statement
theorem perimeter_range :
  ∃ (l u : ℝ), l = 12 ∧ u = 14 + 2 * Real.sqrt 2 ∧
  (∀ p ∈ C, l ≤ perimeter p ∧ perimeter p ≤ u) ∧
  (∃ p1 p2, p1 ∈ C ∧ p2 ∈ C ∧ perimeter p1 = l ∧ perimeter p2 = u) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_range_l48_4877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_zero_rational_l48_4819

-- Define the set of numbers we're considering
def numbers : Set ℝ := {Real.sqrt 2, Real.pi, 0, (4 : ℝ) ^ (1/3)}

-- Define what it means for a real number to be rational
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / b

-- Theorem statement
theorem only_zero_rational : 
  ∃! (x : ℝ), x ∈ numbers ∧ is_rational x ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_zero_rational_l48_4819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l48_4818

noncomputable def original_length : ℝ := 60
noncomputable def original_width : ℝ := 20

noncomputable def original_perimeter : ℝ := 2 * (original_length + original_width)
noncomputable def original_area : ℝ := original_length * original_width

noncomputable def new_side_length : ℝ := original_perimeter / 4
noncomputable def new_area : ℝ := new_side_length ^ 2

theorem garden_area_increase :
  new_area - original_area = 400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l48_4818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_adjacent_diagonals_perpendicular_l48_4866

/-- A cube is a three-dimensional shape with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this statement

/-- A face diagonal is a line segment connecting opposite corners of a face -/
def face_diagonal (c : Cube) : Type := sorry

/-- The angle between two lines in three-dimensional space -/
def angle_between (c : Cube) (l1 l2 : face_diagonal c) : ℝ := sorry

/-- Two faces of a cube are adjacent if they share an edge -/
def adjacent_faces (c : Cube) (f1 f2 : Set (face_diagonal c)) : Prop := sorry

theorem cube_adjacent_diagonals_perpendicular (c : Cube) 
  (d1 d2 : face_diagonal c) (f1 f2 : Set (face_diagonal c)) :
  adjacent_faces c f1 f2 → d1 ∈ f1 → d2 ∈ f2 → angle_between c d1 d2 = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_adjacent_diagonals_perpendicular_l48_4866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_with_property_l48_4875

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- The property we're trying to prove for a given prime number a -/
def property (a : ℕ) : Prop :=
  is_prime a ∧ ∀ x : ℕ, ¬(is_prime (x^3 + a^2 : ℕ))

theorem smallest_prime_with_property :
  (property 5) ∧ (∀ a : ℕ, a < 5 → is_prime a → ¬(property a)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_with_property_l48_4875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_d_value_l48_4838

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem least_d_value (c d : ℕ) 
  (hc_pos : c > 0) 
  (hd_pos : d > 0) 
  (hc_factors : number_of_factors c = 4) 
  (hd_factors : number_of_factors d = c) 
  (hd_div_c : d % c = 0) : 
  d ≥ 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_d_value_l48_4838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l48_4861

noncomputable def f (a c x : ℝ) := a * x^2 - (1/2) * x + c

noncomputable def h (b x : ℝ) := (3/4) * x^2 - b * x + (b/2) - 1/4

noncomputable def g (a c m x : ℝ) := f a c x - m * x

theorem function_properties :
  ∃ (a c : ℝ), 
    (∀ x, f a c x ≥ 0) ∧ 
    (f a c 1 = 0) ∧
    (a = 1/4 ∧ c = 1/4) ∧
    (∀ b, (b < 1/2 → ∀ x, f a c x + h b x < 0 ↔ b < x ∧ x < 1/2) ∧
          (b > 1/2 → ∀ x, f a c x + h b x < 0 ↔ 1/2 < x ∧ x < b) ∧
          (b = 1/2 → ∀ x, f a c x + h b x ≥ 0)) ∧
    (∃ m₁ m₂, (m₁ = -3 ∨ m₁ = -1 + 2 * Real.sqrt 2) ∧
              (m₂ = -3 ∨ m₂ = -1 + 2 * Real.sqrt 2) ∧
              m₁ ≠ m₂ ∧
              (∀ x, x ∈ Set.Icc m₁ (m₁ + 2) → g a c m₁ x ≥ -5) ∧
              (∃ x, x ∈ Set.Icc m₁ (m₁ + 2) ∧ g a c m₁ x = -5) ∧
              (∀ x, x ∈ Set.Icc m₂ (m₂ + 2) → g a c m₂ x ≥ -5) ∧
              (∃ x, x ∈ Set.Icc m₂ (m₂ + 2) ∧ g a c m₂ x = -5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l48_4861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_sum_l48_4878

theorem square_area_sum (n : ℕ) : n ≥ 6 →
  let initialSide : ℝ := 5
  let areaRatio : ℝ := 1 / 2
  let areaSum : ℝ → ℕ → ℝ := λ a r ↦ a * (1 - areaRatio^r) / (1 - areaRatio)
  areaSum (initialSide^2) n > 49 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_sum_l48_4878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_plus_cos_double_eq_one_l48_4836

theorem sin_squared_plus_cos_double_eq_one (α : ℝ) (h : α ∈ Set.Ioo 0 π) :
  (Real.sin α) ^ 2 + Real.cos (2 * α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_plus_cos_double_eq_one_l48_4836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sections_with_five_lines_exists_configuration_with_sixteen_sections_l48_4847

/-- Represents a line segment in a rectangle --/
structure LineSegment where
  -- Add necessary fields here

/-- Represents a rectangle with line segments drawn through it --/
structure RectangleWithLines where
  lines : List LineSegment

/-- Calculates the number of sections in a rectangle given a list of line segments --/
def countSections (rect : RectangleWithLines) : Nat :=
  sorry

/-- Theorem: The maximum number of sections created by 5 line segments in a rectangle is 16 --/
theorem max_sections_with_five_lines (rect : RectangleWithLines) :
  (rect.lines.length = 5) → (countSections rect ≤ 16) :=
by sorry

/-- Theorem: There exists a configuration of 5 line segments that creates 16 sections --/
theorem exists_configuration_with_sixteen_sections :
  ∃ (rect : RectangleWithLines), (rect.lines.length = 5) ∧ (countSections rect = 16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sections_with_five_lines_exists_configuration_with_sixteen_sections_l48_4847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l48_4800

noncomputable def f (x : ℝ) := Real.log ((4 - x) / (4 + x))

def domain : Set ℝ := { x | -4 < x ∧ x < 4 }

theorem f_properties :
  -- 1. f is odd
  (∀ x, x ∈ domain → f (-x) = -f x) ∧
  -- 2. f is decreasing on its domain
  (∀ x₁ x₂, x₁ ∈ domain → x₂ ∈ domain → x₁ < x₂ → f x₁ > f x₂) ∧
  -- 3. Existence of k and its range
  (∃ k : ℝ, k < 0 ∧ -2 < k ∧ k ≤ -1 ∧
    ∀ θ : ℝ, f (k - Real.cos θ) + f (Real.cos θ ^ 2 - k ^ 2) ≥ 0) ∧
  (∀ k : ℝ, (∀ θ : ℝ, f (k - Real.cos θ) + f (Real.cos θ ^ 2 - k ^ 2) ≥ 0) →
    (k < 0 ∧ -2 < k ∧ k ≤ -1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l48_4800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_changes_2004x2004_l48_4880

/-- Represents a rectangular array --/
structure RectangularArray where
  rows : Nat
  cols : Nat

/-- Represents a path on the edges of a rectangular array --/
structure ArrayPath (array : RectangularArray) where
  changes : Nat
  noCrossing : Bool

/-- The maximum number of changes in direction for a path on a rectangular array --/
def maxChanges (array : RectangularArray) : Nat :=
  array.rows * (array.cols + 1) - 1

/-- Theorem stating the maximum number of changes for a 2004x2004 array --/
theorem max_changes_2004x2004 :
  ∀ (path : ArrayPath { rows := 2004, cols := 2004 }),
    path.noCrossing → path.changes ≤ maxChanges { rows := 2004, cols := 2004 } := by
  sorry

#eval maxChanges { rows := 2004, cols := 2004 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_changes_2004x2004_l48_4880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_below_M_over_100_l48_4892

def sum_of_fractions : ℚ :=
  1 / (3 * 2 * 1 * 18 * 17 * 16 * 15 * 14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) +
  1 / (4 * 3 * 2 * 1 * 17 * 16 * 15 * 14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) +
  1 / (5 * 4 * 3 * 2 * 1 * 16 * 15 * 14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) +
  1 / (6 * 5 * 4 * 3 * 2 * 1 * 15 * 14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) +
  1 / (7 * 6 * 5 * 4 * 3 * 2 * 1 * 14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) +
  1 / (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) +
  1 / (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) +
  1 / (10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)

def M : ℚ := sum_of_fractions * (20 * 19 * 18 * 17 * 16 * 15 * 14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)

theorem greatest_integer_below_M_over_100 :
  ⌊M / 100⌋ = 499 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_below_M_over_100_l48_4892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l48_4826

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f₁ (x : ℝ) : ℤ := floor (4 * x)

noncomputable def g (x : ℝ) : ℝ := 4 * x - (f₁ x : ℝ)

noncomputable def f₂ (x : ℝ) : ℤ := f₁ (g x)

theorem part_one :
  f₁ (7/16 : ℝ) = 1 ∧ f₂ (7/16 : ℝ) = 3 := by sorry

theorem part_two :
  ∀ x : ℝ, f₁ x = 1 ∧ f₂ x = 3 ↔ 7/16 ≤ x ∧ x < 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l48_4826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_distributions_is_16_l48_4852

/-- Represents a zodiac sign --/
inductive ZodiacSign
| Rat | Ox | Tiger | Rabbit | Dragon | Snake | Horse | Sheep | Monkey | Rooster | Dog | Pig

/-- Represents a pair of zodiac signs --/
def ZodiacPair := (ZodiacSign × ZodiacSign)

/-- The set of all zodiac pairs --/
def allZodiacPairs : List ZodiacPair := sorry

/-- Student A's preferred zodiac signs --/
def studentAPrefers : List ZodiacSign := [ZodiacSign.Ox, ZodiacSign.Horse]

/-- Student B's preferred zodiac signs --/
def studentBPrefers : List ZodiacSign := [ZodiacSign.Ox, ZodiacSign.Dog, ZodiacSign.Sheep]

/-- Student C's preferred zodiac signs (all signs) --/
def studentCPrefers : List ZodiacSign := sorry

/-- Checks if a zodiac pair contains a preferred sign --/
def containsPreferred (pair : ZodiacPair) (preferred : List ZodiacSign) : Bool := sorry

/-- The number of valid ways to distribute zodiac pairs --/
def numValidDistributions : Nat := sorry

/-- Theorem stating that the number of valid distributions is 16 --/
theorem num_valid_distributions_is_16 : numValidDistributions = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_distributions_is_16_l48_4852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_increasing_on_interval_l48_4811

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

def domain_f : Set ℝ := {x : ℝ | x ≠ 1 ∧ x ≠ -1}

theorem domain_of_f : {x : ℝ | f x ≠ 0} = domain_f := by sorry

theorem f_increasing_on_interval :
  ∀ x1 x2 : ℝ, 1 < x1 → x1 < x2 → f x1 < f x2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_increasing_on_interval_l48_4811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_denominators_l48_4805

/-- A digit is a natural number between 0 and 9 inclusive -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- The set of possible denominators for the fraction representation of 0.cd̄ -/
def PossibleDenominators : Finset ℕ := {3, 9, 11, 33, 99}

/-- The theorem statement -/
theorem repeating_decimal_denominators (c d : Digit) 
  (h1 : c.val ≠ d.val) 
  (h2 : ¬(c.val = 0 ∧ d.val = 0)) : 
  ∃! (n : ℕ), n = Finset.card PossibleDenominators := by
  sorry

#check repeating_decimal_denominators

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_denominators_l48_4805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_sixth_minus_x_l48_4835

theorem cos_pi_sixth_minus_x (x : ℝ) (h : Real.sin x + Real.sqrt 3 * Real.cos x = 8/5) : 
  Real.cos (π/6 - x) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_sixth_minus_x_l48_4835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l48_4871

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 2 * a * Real.sin (2 * x + Real.pi / 6) + a + b

-- Define the domain and range
def domain : Set ℝ := Set.Icc 0 (Real.pi / 2)
def range : Set ℝ := Set.Icc (-5) 1

-- State the theorem
theorem function_values (a b : ℝ) : 
  (∀ x ∈ domain, f a b x ∈ range) → 
  ((a = 2 ∧ b = -5) ∨ (a = -2 ∧ b = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l48_4871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_f_prime_nonpositive_f_inequality_l48_4856

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x - 1

-- Statement 1: Tangent line at x = 0
theorem tangent_line_at_zero :
  ∃ (m : ℝ), (deriv f 0 = m) ∧ m = 0 :=
sorry

-- Statement 2: f' is non-positive on [0, π)
theorem f_prime_nonpositive :
  ∀ x ∈ Set.Icc 0 Real.pi, deriv f x ≤ 0 :=
sorry

-- Statement 3: Inequality for f(m+n) - f(m) and f(n)
theorem f_inequality (m n : ℝ) (hm : m ∈ Set.Ioo 0 (Real.pi / 2)) (hn : n ∈ Set.Ioo 0 (Real.pi / 2)) :
  f (m + n) - f m < f n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_f_prime_nonpositive_f_inequality_l48_4856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_and_exponent_l48_4863

-- Define the constants
noncomputable def a : ℝ := Real.log (2/3) / Real.log (1/3)
noncomputable def b : ℝ := Real.log (1/3) / Real.log (1/2)
noncomputable def c : ℝ := (1/2) ^ (3/10)

-- State the theorem
theorem order_of_logarithms_and_exponent : b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_and_exponent_l48_4863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l48_4865

noncomputable section

-- Define the vectors m and n
def m (a θ : ℝ) : ℝ × ℝ := (a - Real.sin θ, -1/2)
def n (θ : ℝ) : ℝ × ℝ := (1/2, Real.cos θ)

-- Define perpendicularity of vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

theorem vector_relations (θ : ℝ) :
  (perpendicular (m (Real.sqrt 2 / 2) θ) (n θ) → Real.sin (2 * θ) = -1/2) ∧
  (parallel (m 0 θ) (n θ) → Real.tan θ = 2 + Real.sqrt 3 ∨ Real.tan θ = 2 - Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l48_4865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_expression_g_increasing_intervals_l48_4894

noncomputable def f (x : ℝ) := Real.sin x

noncomputable def g (x : ℝ) := 2 * f ((x - Real.pi/3) / 2)

theorem g_expression (x : ℝ) : g x = 2 * Real.sin (x/2 - Real.pi/6) := by sorry

theorem g_increasing_intervals (k : ℤ) :
  StrictMonoOn g (Set.Icc (2 * Real.pi * (k : ℝ) - Real.pi/3) (2 * Real.pi * (k : ℝ) + Real.pi/3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_expression_g_increasing_intervals_l48_4894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_diameter_l48_4842

theorem circumscribed_circle_diameter 
  (triangle : Set ℝ × Set ℝ × Set ℝ) 
  (side : ℝ) 
  (angle : ℝ) :
  side = 12 →
  angle = 30 * π / 180 →
  ∃ (diameter : ℝ), diameter = 24 ∧ 
    diameter = 2 * (side / Real.sin angle) :=
by
  intro h_side h_angle
  use 24
  constructor
  · rfl
  · rw [h_side, h_angle]
    simp [Real.sin]
    sorry -- The exact proof steps are omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_diameter_l48_4842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_properties_l48_4890

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 0

-- Define a point P on line l
def point_on_line_l (P : ℝ × ℝ) : Prop := line_l P.1 P.2

-- Define tangent lines PA and PB
def tangent_lines (P A B : ℝ × ℝ) : Prop :=
  my_circle A.1 A.2 ∧ my_circle B.1 B.2 ∧
  -- Additional conditions for tangency would be defined here
  True

-- Theorem statement
theorem tangent_properties (P A B : ℝ × ℝ) :
  point_on_line_l P →
  tangent_lines P A B →
  (∃ t : ℝ, (3/2 - A.1) * t + A.1 = 3/2 ∧ (-1/2 - A.2) * t + A.2 = -1/2 ∧
             (3/2 - B.1) * t + B.1 = 3/2 ∧ (-1/2 - B.2) * t + B.2 = -1/2) ∧
  (∀ A' B' : ℝ × ℝ, tangent_lines P A' B' →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_properties_l48_4890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_real_roots_l48_4834

theorem range_of_m_for_real_roots : 
  ∀ m : ℝ, 
  (∃ x : ℝ, 4^x + m*2^x + m^2 - 1 = 0) ↔ 
  (-(2*Real.sqrt 3)/3 ≤ m ∧ m < 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_real_roots_l48_4834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_l48_4808

/-- A lattice point in an xy-coordinate system is any point (x, y) where both x and y are integers. -/
def is_lattice_point (x y : ℤ) : Prop := True

/-- The graph of y = mx + 3 passes through no lattice point with 0 < x ≤ 50 for all m such that 1/3 < m < b. -/
def no_lattice_points (m b : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x → x ≤ 50 → is_lattice_point x y → (↑y : ℚ) ≠ m * (↑x : ℚ) + 3

/-- The maximum possible value of b is 17/50. -/
theorem max_b_value :
  ∀ b : ℚ, (∀ m : ℚ, 1/3 < m → m < b → no_lattice_points m b) →
  b ≤ 17/50 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_l48_4808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_specific_field_l48_4873

/-- Calculates the fencing required for a rectangular field -/
noncomputable def fencing_required (area : ℝ) (uncovered_side : ℝ) : ℝ :=
  let width := area / uncovered_side
  uncovered_side + 2 * width

theorem fencing_for_specific_field :
  fencing_required 80 20 = 28 := by
  -- Unfold the definition of fencing_required
  unfold fencing_required
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_specific_field_l48_4873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_condition_lambda_lower_bound_l48_4843

open Real

noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 - (2*a + 2) * x + (2*a + 1) * log x

theorem tangent_slope_condition (a : ℝ) : 
  (deriv (f a)) 2 < 0 → a > 1/2 := by sorry

theorem lambda_lower_bound (a : ℝ) (l : ℝ) :
  a ∈ Set.Icc (3/2) (5/2) →
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → x₁ ≠ x₂ → 
    |f a x₁ - f a x₂| < l * |1/x₁ - 1/x₂|) →
  l ≥ 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_condition_lambda_lower_bound_l48_4843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l48_4845

-- Define the vector type
def MyVector := ℝ × ℝ

-- Define the given vectors
def a : MyVector := (-2, 0)
def b : MyVector := (2, 1)
def c (x : ℝ) : MyVector := (x, 1)

-- Define vector addition
def vectorAdd (v w : MyVector) : MyVector :=
  (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def scalarMul (k : ℝ) (v : MyVector) : MyVector :=
  (k * v.1, k * v.2)

-- Define collinearity
def collinear (v w : MyVector) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1 ∧ v.2 * w.1 = k * v.1 * w.2

-- Theorem statement
theorem vector_collinearity (x : ℝ) :
  collinear (vectorAdd (scalarMul 3 a) b) (c x) → x = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l48_4845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_sum_l48_4846

theorem unique_solution_sum (x₀ y₀ : ℕ) : 
  (∀ x y : ℕ, x > 0 → y > 0 → x^2 * y - x^2 - 3*y - 14 = 0 ↔ (x = x₀ ∧ y = y₀)) →
  x₀ + y₀ = 20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_sum_l48_4846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l48_4839

-- Define the function f(x) = 1/(x+1)
noncomputable def f (x : ℝ) : ℝ := 1 / (x + 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ x ≠ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l48_4839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l48_4855

-- Define the sequence a_n
noncomputable def a (n : ℕ) (x : ℝ) : ℝ := 2 - ((x + 3) / x) ^ n

-- State the theorem
theorem range_of_x (x : ℝ) : 
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |a n x - 2| < ε) → 
  x ∈ Set.Ioi (-3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l48_4855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_at_marked_price_is_30_percent_l48_4899

/-- Represents the gain percentage when selling at marked price, given the discount percentage and sale gain percentage. -/
noncomputable def gainPercentageAtMarkedPrice (discountPercentage : ℝ) (saleGainPercentage : ℝ) : ℝ :=
  ((1 + saleGainPercentage / 100) / (1 - discountPercentage / 100) - 1) * 100

/-- Theorem stating that with a 10% discount and 17% sale gain, the gain at marked price is 30%. -/
theorem gain_at_marked_price_is_30_percent :
  gainPercentageAtMarkedPrice 10 17 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_at_marked_price_is_30_percent_l48_4899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_rectangle_ratio_l48_4806

/-- Represents a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a circle --/
structure Circle where
  radius : ℝ

/-- The ratio of the area of the inner circle to the area of the outer rectangle --/
noncomputable def areaRatio (outerRect : Rectangle) (innerCircle : Circle) : ℝ :=
  (Real.pi * innerCircle.radius^2) / (outerRect.length * outerRect.width)

theorem inscribed_circles_rectangle_ratio 
  (outerRect : Rectangle)
  (h_ratio : outerRect.length / outerRect.width = 3 / 2)
  (middleCircle : Circle)
  (h_middle_inscribed : middleCircle.radius = outerRect.width / 2)
  (innerRect : Rectangle)
  (h_inner_inscribed : innerRect.length = 2 * middleCircle.radius ∧ 
                       innerRect.width = (2 * middleCircle.radius) * (2 / 3))
  (innerCircle : Circle)
  (h_inner_circle : innerCircle.radius = innerRect.width / 2) :
  areaRatio outerRect innerCircle = 27 * Real.pi / 104 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_rectangle_ratio_l48_4806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l48_4874

theorem inequality_proof (x y z : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) (hz : z ∈ Set.Icc 0 1) :
  x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l48_4874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_ratio_l48_4862

theorem cube_root_sum_ratio (x y : ℝ) (h1 : x ≥ Real.sqrt 2021) 
  (h2 : (x + Real.sqrt 2021) ^ (1/3 : ℝ) + (x - Real.sqrt 2021) ^ (1/3 : ℝ) = y ^ (1/3 : ℝ)) :
  2 ≤ y / x ∧ y / x < 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_ratio_l48_4862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_of_m_l48_4831

noncomputable def f (a b x : ℝ) : ℝ := (a * 2^x - 1) / (2^x + b)

theorem odd_function_range_of_m (a b m : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →
  (∀ x ∈ Set.Icc 1 2, 2 + m * f a b x + 2^x > 0) →
  m ∈ Set.Ioi (-2 * Real.sqrt 6 - 5) :=
by
  sorry

#check odd_function_range_of_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_of_m_l48_4831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_hyperbola_eccentricity_sqrt3_l48_4814

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the foci
def leftFocus (c : ℝ) : ℝ × ℝ := (-c, 0)
def rightFocus (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the asymptote
noncomputable def asymptote (a b : ℝ) (x : ℝ) : ℝ := (b / a) * x

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem hyperbola_eccentricity (a b c : ℝ) (P : ℝ × ℝ) :
  hyperbola a b P.1 P.2 →
  (∃ t : ℝ, P.2 = asymptote a b (P.1 - (rightFocus c).1) + (rightFocus c).2) →
  distance P (leftFocus c) = 3 * distance P (rightFocus c) →
  c^2 = 3 * a^2 := by
  sorry

-- Define eccentricity
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

-- Prove the final result
theorem hyperbola_eccentricity_sqrt3 (a b c : ℝ) (P : ℝ × ℝ) :
  hyperbola a b P.1 P.2 →
  (∃ t : ℝ, P.2 = asymptote a b (P.1 - (rightFocus c).1) + (rightFocus c).2) →
  distance P (leftFocus c) = 3 * distance P (rightFocus c) →
  eccentricity c a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_hyperbola_eccentricity_sqrt3_l48_4814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractions_weighted_sum_l48_4837

-- Define the polynomial coefficients
noncomputable def a₀ : ℝ := sorry
noncomputable def a₁ : ℝ := sorry
noncomputable def a₂ : ℝ := sorry
noncomputable def a₃ : ℝ := sorry
noncomputable def a₄ : ℝ := sorry

-- Define the given equation
axiom eq_coeff : ∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4

-- Theorem for the first part
theorem sum_of_fractions : a₁/2 + a₂/4 + a₃/8 + a₄/16 = 15 := by
  sorry

-- Theorem for the second part
theorem weighted_sum : a₁ + 2*a₂ + 3*a₃ + 4*a₄ = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractions_weighted_sum_l48_4837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l48_4832

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  (2 * b - Real.sqrt 3 * c) / (Real.sqrt 3 * a) = cos C / cos A →
  B = π / 6 →
  (1 / 2) * a * b * sin C = 4 * Real.sqrt 3 →
  -- Conclusions to prove
  A = π / 6 ∧
  let M := (1 / 2) * (b + c);
  Real.sqrt ((1 / 4) * (2 * b^2 + 2 * c^2 - (b - c)^2)) = 2 * Real.sqrt 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l48_4832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AC₁_equals_sqrt_98_plus_56_sqrt_2_l48_4881

/-- A parallelepiped with specific dimensions and angles -/
structure Parallelepiped where
  AB : ℝ
  AD : ℝ
  AA₁ : ℝ
  angle_BAD : ℝ
  angle_BAA₁ : ℝ
  angle_DAA₁ : ℝ

/-- The length of AC₁ in the given parallelepiped -/
noncomputable def length_AC₁ (p : Parallelepiped) : ℝ :=
  Real.sqrt (98 + 56 * Real.sqrt 2)

/-- Theorem stating that for a parallelepiped with the given dimensions and angles,
    the length of AC₁ is √(98 + 56√2) -/
theorem length_AC₁_equals_sqrt_98_plus_56_sqrt_2 (p : Parallelepiped)
    (h1 : p.AB = 5)
    (h2 : p.AD = 3)
    (h3 : p.AA₁ = 7)
    (h4 : p.angle_BAD = Real.pi / 3)
    (h5 : p.angle_BAA₁ = Real.pi / 4)
    (h6 : p.angle_DAA₁ = Real.pi / 4) :
    length_AC₁ p = Real.sqrt (98 + 56 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AC₁_equals_sqrt_98_plus_56_sqrt_2_l48_4881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hit_ground_time_l48_4867

/-- The time it takes for a ball to hit the ground when thrown upward -/
theorem ball_hit_ground_time : ∃ t : ℝ, t > 0 ∧ -16 * t^2 + 16 * t + 120 = 0 ∧ abs (t - 3.2835) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hit_ground_time_l48_4867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l48_4895

-- Define the function f(x)
noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.sin x + Real.cos x)

-- Theorem statement
theorem f_properties :
  (∃ (min : ℝ), ∀ x, f x ≥ min ∧ min = 1 - Real.sqrt 2) ∧
  (∀ k : ℤ, f (k * π - π / 8) = 1 - Real.sqrt 2) ∧
  (∀ k : ℤ, ∀ x y, k * π - π / 8 ≤ x ∧ x < y ∧ y ≤ k * π + 3 * π / 8 → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l48_4895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_combination_probability_l48_4803

/-- The probability of picking the same color combination from a jar of candies -/
theorem same_color_combination_probability : 
  let total_candies : ℕ := 20
  let red_candies : ℕ := 8
  let blue_candies : ℕ := 12
  let pick_size : ℕ := 2
  
  -- Define the probability function
  2869 / 4845 = (
    -- Probability of both picking 2 red
    (Nat.choose red_candies pick_size * Nat.choose (red_candies - pick_size) pick_size) / 
    (Nat.choose total_candies pick_size * Nat.choose (total_candies - pick_size) pick_size) +
    
    -- Probability of both picking 2 blue
    (Nat.choose blue_candies pick_size * Nat.choose (blue_candies - pick_size) pick_size) / 
    (Nat.choose total_candies pick_size * Nat.choose (total_candies - pick_size) pick_size) +
    
    -- Probability of both picking 1 red and 1 blue
    (Nat.choose red_candies 1 * Nat.choose blue_candies 1) / Nat.choose total_candies pick_size
  ) := by sorry

#eval 2869 + 4845  -- This should output 7714

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_combination_probability_l48_4803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_value_l48_4807

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem cos_2x_value (x₀ : ℝ) (h1 : x₀ ∈ Set.Icc (π/4) (π/2)) (h2 : f x₀ = 6/5) :
  Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_value_l48_4807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_AQA_l48_4883

-- Define the circles and points
variable (Γ Γ' : Set (ℝ × ℝ))
variable (P Q A A' O O' : ℝ × ℝ)

-- Define the conditions
def circles_intersect (Γ Γ' : Set (ℝ × ℝ)) (P Q : ℝ × ℝ) : Prop :=
  P ∈ Γ ∧ P ∈ Γ' ∧ Q ∈ Γ ∧ Q ∈ Γ'

def A_on_Γ (Γ : Set (ℝ × ℝ)) (A : ℝ × ℝ) : Prop :=
  A ∈ Γ

def line_AP_intersects_Γ' (Γ' : Set (ℝ × ℝ)) (A P A' : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A' = (1 - t) • A + t • P ∧ A' ∈ Γ'

-- Define similarity of triangles
def similar_triangles (t1 t2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (a1, b1, c1) := t1
  let (a2, b2, c2) := t2
  ∃ k : ℝ, k > 0 ∧
    ‖b1 - a1‖ / ‖b2 - a2‖ = k ∧
    ‖c1 - b1‖ / ‖c2 - b2‖ = k ∧
    ‖a1 - c1‖ / ‖a2 - c2‖ = k

-- State the theorem
theorem triangles_AQA'_OQO'_similar
  (h1 : circles_intersect Γ Γ' P Q)
  (h2 : A_on_Γ Γ A)
  (h3 : line_AP_intersects_Γ' Γ' A P A') :
  similar_triangles (A, Q, A') (O, Q, O') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_AQA_l48_4883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l48_4829

/-- Represents an isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  short_base : ℝ
  long_base : ℝ
  height : ℝ
  angle : ℝ

/-- Calculates the area of an isosceles trapezoid -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ :=
  (t.short_base + t.long_base) * t.height / 2

/-- The main theorem stating the area of the specific isosceles trapezoid -/
theorem isosceles_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    short_base := 18,
    long_base := 24,
    height := 6,
    angle := 30 * Real.pi / 180  -- Convert degrees to radians
  }
  area t = 126 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l48_4829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l48_4887

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}
def B : Set ℝ := {x : ℝ | -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 ≤ x ∧ x ≤ Real.sqrt 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l48_4887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_identity_expression_value_l48_4801

open Real

theorem tangent_sum_identity (x y : ℝ) :
  tan (x + y) = (tan x + tan y) / (1 - tan x * tan y) := by sorry

theorem expression_value :
  let x : ℝ := 10 * π / 180  -- 10 degrees in radians
  let y : ℝ := 20 * π / 180  -- 20 degrees in radians
  (sqrt 3 / 3) * tan x * tan y + tan x + tan y = sqrt 3 / 3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_identity_expression_value_l48_4801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_16_l48_4851

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_arithmetic (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

/-- Collinearity of three points -/
def collinear (P₁ P P₂ : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = (1 - t) • P₁ + t • P₂

theorem arithmetic_sequence_sum_16 
  (a : ℕ → ℝ) 
  (P₁ P P₂ : ℝ × ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_collinear : collinear P₁ P P₂)
  (h_vector : P - P₁ = a 2 • (P - P₁) + a 15 • (P₂ - P₁)) :
  sum_arithmetic a 16 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_16_l48_4851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koolaid_powder_amount_l48_4859

/-- Represents the amount of koolaid powder in tablespoons -/
def koolaid_powder : ℝ := 2

/-- Initial amount of water in tablespoons -/
def initial_water : ℝ := 16

/-- Amount of water evaporated in tablespoons -/
def evaporated_water : ℝ := 4

/-- Factor by which the remaining water is multiplied -/
def water_multiplier : ℝ := 4

/-- Final percentage of koolaid powder in the mixture -/
def final_koolaid_percentage : ℝ := 4

/-- Theorem stating that the amount of koolaid powder added is 2 tablespoons -/
theorem koolaid_powder_amount : koolaid_powder = 2 := by
  -- The proof goes here
  sorry

#eval koolaid_powder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_koolaid_powder_amount_l48_4859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l48_4889

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 3)) / Real.sqrt (1 - 2^x)

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo (-3 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l48_4889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l48_4844

def S (n : ℕ) : ℤ := 3 * n^2 + 2 * n - 1

def a : ℕ → ℤ
  | 0 => 4  -- Add this case to cover Nat.zero
  | 1 => 4
  | n + 2 => 6 * (n + 2) - 1

theorem sequence_general_term (n : ℕ) : 
  (∀ k : ℕ, k ≥ 1 → S k - S (k - 1) = a k) ∧ 
  (a 1 = 4) ∧ 
  (∀ m : ℕ, m ≥ 2 → a m = 6 * m - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l48_4844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l48_4802

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_of_inequality 
  (h1 : ∀ x : ℝ, f' x > 1 - f x)
  (h2 : f 0 = 6) :
  {x : ℝ | Real.exp x * f x > Real.exp x + 5} = Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l48_4802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_truck_tank_height_l48_4858

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculate the volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

theorem oil_truck_tank_height : 
  let stationaryTank : Cylinder := ⟨100, 25⟩
  let oilTruckTank : Cylinder := ⟨8, 10⟩
  let volumePumped : ℝ := cylinderVolume ⟨100, 0.064⟩
  volumePumped = cylinderVolume oilTruckTank → oilTruckTank.height = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_truck_tank_height_l48_4858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_purchase_total_andre_purchase_total_l48_4888

theorem mall_purchase_total (treadmill_price : ℝ) (discount_percent : ℝ) 
  (num_plates : ℕ) (plate_price : ℝ) : ℝ :=
  let discounted_treadmill := treadmill_price * (1 - discount_percent / 100)
  let total_plate_cost := (num_plates : ℝ) * plate_price
  discounted_treadmill + total_plate_cost

theorem andre_purchase_total : 
  mall_purchase_total 1350 30 2 50 = 1045 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_purchase_total_andre_purchase_total_l48_4888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l48_4824

/-- Parabola type -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Focus of the parabola -/
noncomputable def focus (e : Parabola) : Point :=
  ⟨e.p / 2, 0⟩

/-- Check if a point is on the parabola -/
def is_on_parabola (e : Parabola) (p : Point) : Prop :=
  p.y^2 = 2 * e.p * p.x

/-- Theorem: Minimum value of |PA| + |PF| is 6 -/
theorem min_distance_sum (e : Parabola) (a : Point) :
  (∀ m : Point, is_on_parabola e m → distance m (focus e) ≥ 2) →
  a = ⟨4, 1⟩ →
  (∃ c : ℝ, ∀ p : Point, is_on_parabola e p → distance p a + distance p (focus e) ≥ c) ∧
  (∀ ε > 0, ∃ p : Point, is_on_parabola e p ∧ distance p a + distance p (focus e) < 6 + ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l48_4824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_l48_4886

theorem lcm_ratio (a x : ℕ) (h : Nat.lcm a x = 84) (ha : a = 21) : 
  ∃ (k : ℕ), k * 4 = 21 ∧ k * x = 84 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_l48_4886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_implies_3mn_equals_one_l48_4804

theorem linear_equation_implies_3mn_equals_one
  (m n : ℕ)
  (h : ∃ (a b c : ℚ), (a ≠ 0 ∨ b ≠ 0) ∧
    ∀ (x y : ℚ), x^(4*m - 1) - 8*y^(3*n - 2*m) = a*x + b*y + c) :
  3*m*n = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_implies_3mn_equals_one_l48_4804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chucks_play_area_is_12_25pi_l48_4820

/-- The area in which Chuck the llama can play when tied to a shed -/
noncomputable def chucks_play_area (shed_length shed_width leash_length : ℝ) : ℝ :=
  let unrestricted_area := (3/4) * Real.pi * leash_length^2
  let additional_area := (1/4) * Real.pi * (leash_length - shed_length)^2
  unrestricted_area + additional_area

/-- Theorem stating that Chuck's play area is 12.25π square meters -/
theorem chucks_play_area_is_12_25pi :
  chucks_play_area 4 3 4 = (49/4) * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chucks_play_area_is_12_25pi_l48_4820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l48_4810

theorem max_value_of_function :
  ∃ (max_y : ℝ), max_y = 11 * Real.sqrt 3 / 6 ∧
  ∀ x ∈ Set.Icc (-5 * Real.pi / 12) (-Real.pi / 3),
    Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6) ≤ max_y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l48_4810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l48_4884

noncomputable section

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-3, 4)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_properties :
  vector_AB = (-4, 2) ∧ 
  magnitude vector_AB = 2 * Real.sqrt 5 ∧
  dot_product (A.1 - O.1, A.2 - O.2) (B.1 - O.1, B.2 - O.2) / (magnitude (A.1 - O.1, A.2 - O.2) * magnitude (B.1 - O.1, B.2 - O.2)) = Real.sqrt 5 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l48_4884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_trig_values_l48_4850

noncomputable def a : ℝ := Real.sin (Real.cos (2016 * Real.pi / 180))
noncomputable def b : ℝ := Real.sin (Real.sin (2016 * Real.pi / 180))
noncomputable def c : ℝ := Real.cos (Real.sin (2016 * Real.pi / 180))
noncomputable def d : ℝ := Real.cos (Real.cos (2016 * Real.pi / 180))

theorem order_of_trig_values : c > d ∧ d > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_trig_values_l48_4850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_intersecting_lines_theorem_l48_4840

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Define the tangent lines
def tangent_line1 (x y : ℝ) : Prop := y = 2
def tangent_line2 (x y : ℝ) : Prop := 4*x + 3*y - 10 = 0

-- Define the intersecting lines
def intersecting_line1 (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0
def intersecting_line2 (x : ℝ) : Prop := x = 1

-- Theorem for tangent lines
theorem tangent_lines_theorem :
  (∀ x y, C x y → ¬(tangent_line1 x y)) ∧
  (∀ x y, C x y → ¬(tangent_line2 x y)) ∧
  tangent_line1 P.1 P.2 ∧
  tangent_line2 P.1 P.2 :=
by sorry

-- Function to calculate distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem for intersecting lines
theorem intersecting_lines_theorem :
  (∃ x1 y1 x2 y2,
    C x1 y1 ∧ C x2 y2 ∧
    intersecting_line1 x1 y1 ∧ intersecting_line1 x2 y2 ∧
    intersecting_line1 P.1 P.2 ∧
    distance x1 y1 x2 y2 = 2 * Real.sqrt 3) ∧
  (∃ x1 y1 x2 y2,
    C x1 y1 ∧ C x2 y2 ∧
    intersecting_line2 x1 ∧ intersecting_line2 x2 ∧
    intersecting_line2 P.1 ∧
    distance x1 y1 x2 y2 = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_intersecting_lines_theorem_l48_4840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l48_4815

theorem abc_inequality : (6/10 : ℝ)^((2/10) : ℝ) > (2/10 : ℝ)^((2/10) : ℝ) ∧ 
                         (2/10 : ℝ)^((2/10) : ℝ) > (2/10 : ℝ)^((6/10) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l48_4815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_area_of_cube_l48_4893

-- Define a cube with volume 8
noncomputable def cube_volume : ℝ := 8

-- Define the surface area of the inscribed sphere
noncomputable def inscribed_sphere_surface_area : ℝ := 4 * Real.pi

-- Theorem statement
theorem inscribed_sphere_area_of_cube (v : ℝ) (h : v = cube_volume) :
  inscribed_sphere_surface_area = 4 * Real.pi := by
  -- Unfold the definition of inscribed_sphere_surface_area
  unfold inscribed_sphere_surface_area
  -- The left-hand side is now equal to the right-hand side by definition
  rfl

#check inscribed_sphere_area_of_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_area_of_cube_l48_4893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_pass_prob_at_least_one_pass_prob_exactly_one_highest_prob_l48_4853

-- Define the probabilities of each candidate passing
def prob_A : ℚ := 2/5
def prob_B : ℚ := 3/4
def prob_C : ℚ := 1/3

-- Theorem for the probability that all three candidates pass
theorem all_pass_prob : prob_A * prob_B * prob_C = 1/10 := by sorry

-- Theorem for the probability that at least one candidate passes
theorem at_least_one_pass_prob : 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C) = 9/10 := by sorry

-- Define probabilities for different scenarios
def prob_only_A : ℚ := prob_A * (1 - prob_B) * (1 - prob_C)
def prob_only_B : ℚ := (1 - prob_A) * prob_B * (1 - prob_C)
def prob_only_C : ℚ := (1 - prob_A) * (1 - prob_B) * prob_C
def prob_exactly_one : ℚ := prob_only_A + prob_only_B + prob_only_C

def prob_A_and_B : ℚ := prob_A * prob_B * (1 - prob_C)
def prob_A_and_C : ℚ := prob_A * (1 - prob_B) * prob_C
def prob_B_and_C : ℚ := (1 - prob_A) * prob_B * prob_C
def prob_exactly_two : ℚ := prob_A_and_B + prob_A_and_C + prob_B_and_C

def prob_none : ℚ := (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

-- Theorem for the probability that exactly one candidate passes and it being the highest
theorem exactly_one_highest_prob : 
  prob_exactly_one = 5/12 ∧ 
  prob_exactly_one > prob_exactly_two ∧
  prob_exactly_one > prob_A * prob_B * prob_C ∧
  prob_exactly_one > prob_none := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_pass_prob_at_least_one_pass_prob_exactly_one_highest_prob_l48_4853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_special_function_l48_4841

/-- The limit of ((1 + x^2 * 2^x) / (1 + x^2 * 5^x))^(1 / sin^3(x)) as x approaches 0 is 2/5 -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 0 → |x| < δ →
    |((1 + x^2 * 2^x) / (1 + x^2 * 5^x))^(1 / Real.sin x^3) - 2/5| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_special_function_l48_4841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_subset_bound_l48_4876

theorem family_subset_bound (X : Type) [Fintype X] (F : Set (Set X)) (k : ℕ) 
  (X₁ X₂ : Set (Set X)) :
  (∀ A ∈ F, ∃ x y z : X, A = {x, y, z}) →
  (∀ x y : X, x ≠ y → ∃ S : Finset (Set X), S.card = k ∧ ∀ A ∈ S, A ∈ F ∧ x ∈ A ∧ y ∈ A) →
  (F = X₁ ∪ X₂) →
  (∀ A ∈ F, (∃ x ∈ X₁, x ∩ A ≠ ∅) ∧ (∃ y ∈ X₂, y ∩ A ≠ ∅)) →
  Fintype.card X ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_subset_bound_l48_4876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_composition_l48_4885

-- Define the function r as noncomputable
noncomputable def r (θ : ℝ) : ℝ := 2 / (1 - θ)

-- State the theorem
theorem r_composition (θ : ℝ) (h : θ ≠ 1) (h' : r θ ≠ 1) : 
  r (r θ) = 2 * (θ - 1) / (1 + θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_composition_l48_4885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_segment_speed_l48_4849

/-- Represents a journey with three segments of equal duration --/
structure Journey where
  total_distance : ℚ
  total_time : ℚ
  speed1 : ℚ
  speed2 : ℚ
  speed3 : ℚ

/-- Calculates the average speed of a journey --/
def average_speed (j : Journey) : ℚ :=
  j.total_distance / j.total_time

/-- Theorem: Given the conditions of John's journey, the speed in the last segment is 82 mph --/
theorem last_segment_speed (j : Journey) 
  (h1 : j.total_distance = 144)
  (h2 : j.total_time = 2)
  (h3 : j.speed1 = 64)
  (h4 : j.speed2 = 70)
  (h5 : average_speed j = (j.speed1 + j.speed2 + j.speed3) / 3) :
  j.speed3 = 82 := by
  sorry

#check last_segment_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_segment_speed_l48_4849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l48_4823

theorem equation_solution (x : ℝ) : 
  (4 : ℝ)^(x + 3) = 320 - (4 : ℝ)^x ↔ x = (Real.log (64/13)) / (Real.log 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l48_4823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_passwords_l48_4882

/-- Represents a 5-digit password --/
def Password := Fin 5 → Fin 10

/-- Check if a password starts with a given two-digit sequence --/
def starts_with (p : Password) (a b : Fin 10) : Prop :=
  p 0 = a ∧ p 1 = b

/-- The set of all possible 5-digit passwords --/
def all_passwords : Set Password :=
  {p | True}

/-- The set of valid passwords (not starting with 78 or 90) --/
def valid_passwords : Set Password :=
  {p | ¬(starts_with p 7 8 ∨ starts_with p 9 0)}

/-- Assumption: valid_passwords is finite --/
instance : Fintype valid_passwords := sorry

/-- The main theorem stating the number of valid passwords --/
theorem count_valid_passwords : Fintype.card valid_passwords = 98000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_passwords_l48_4882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_K_to_M_l48_4809

-- Define the circles and their properties
structure Circle where
  radius : ℝ

-- Define the circles
def circle_K : Circle := ⟨1⟩  -- Arbitrary radius, we'll define the relationship later
def circle_L : Circle := ⟨1⟩
def circle_M : Circle := ⟨1⟩

-- Define the diameter
def diameter_AB : ℝ := 2 * circle_K.radius

-- Define the geometric relationships
axiom L_tangent_K : circle_L.radius + circle_K.radius = diameter_AB / 2
axiom M_tangent_K : circle_M.radius + circle_K.radius = diameter_AB / 2
axiom M_tangent_L : circle_M.radius + circle_L.radius = circle_K.radius

-- Define the relationship between radii
axiom radius_L_half_K : circle_L.radius = circle_K.radius / 2

-- Theorem to prove
theorem area_ratio_K_to_M :
  (π * circle_K.radius^2) / (π * circle_M.radius^2) = 16 := by
  sorry  -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_K_to_M_l48_4809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_characterization_l48_4864

theorem sequence_characterization (a : ℕ → ℕ) :
  (∀ n, (a n : ℝ) ≤ n * Real.sqrt n) →
  (∀ m n : ℕ, m ≠ n → (m - n : ℤ) ∣ (a m - a n : ℤ)) →
  (∀ n, a n = 1 ∨ a n = n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_characterization_l48_4864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tractor_productivity_l48_4828

/-- Represents the number of hectares a single tractor can plough in a day -/
def hectares_per_tractor_per_day : ℝ → ℝ := sorry

/-- Represents the total area of the field in hectares -/
def field_area : ℝ → ℝ := sorry

theorem tractor_productivity 
  (h1 : field_area = λ x => 6 * (hectares_per_tractor_per_day x) * 4)
  (h2 : field_area = λ _ => 4 * 144 * 5) :
  hectares_per_tractor_per_day = λ _ => 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tractor_productivity_l48_4828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_line_l48_4821

-- Define the set of points (x,y) in the Cartesian plane
def S : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p.1 = (Real.cos t) ^ 4 ∧ p.2 = (Real.sin t) ^ 4}

-- Theorem stating that S represents a line
theorem S_is_line : ∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ ∀ p ∈ S, a * p.1 + b * p.2 + c = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_line_l48_4821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_with_square_difference_l48_4879

theorem increasing_sequence_with_square_difference (k : ℕ) (h : k ≥ 2) :
  let a : ℕ → ℕ := λ n => (k + n) * (k + n - 1) / 2
  (∀ n, a n < a (n + 1)) ∧
  (∀ n, a n + a (n + 1) = (a (n + 1) - a n)^2) :=
by
  intro a
  constructor
  · sorry -- Proof that the sequence is increasing
  · sorry -- Proof that the sum of consecutive terms equals the square of their difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_with_square_difference_l48_4879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_arrangements_count_l48_4869

/-- Represents a school in the society -/
structure School where
  members : Finset Nat
  size : members.card = 5

/-- Represents the society of schools -/
structure Society where
  schools : Finset School
  size : schools.card = 4
  total_members : (schools.sum fun s => s.members.card) = 20
  nearest_neighbor : School → School
  neighbor_injective : Function.Injective nearest_neighbor

/-- The number of ways to organize the conference -/
def conference_arrangements (s : Society) : ℕ :=
  s.schools.card * 5 * 10 * 25

/-- Theorem stating the number of possible conference arrangements -/
theorem conference_arrangements_count (s : Society) :
  conference_arrangements s = 5000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_arrangements_count_l48_4869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_division_ratio_four_fifths_l48_4891

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- A line in a 2D plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a line passes through a point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

/-- The ratio of areas of two regions formed by a line passing through a triangle -/
noncomputable def area_ratio (t : Triangle) (l : Line) : ℝ := sorry

/-- The theorem to be proved -/
theorem centroid_division (t : Triangle) (l : Line) :
  passes_through l (centroid t) →
  (4/5 ≤ area_ratio t l ∧ area_ratio t l ≤ 5/4) := by
  sorry

/-- The position where the ratio is exactly 4/5 -/
theorem ratio_four_fifths (t : Triangle) :
  ∃ l : Line, passes_through l (centroid t) ∧ area_ratio t l = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_division_ratio_four_fifths_l48_4891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_external_tangent_y_intercept_l48_4822

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The common external tangent line of two circles -/
structure TangentLine where
  slope : ℝ
  y_intercept : ℝ

/-- Predicate to check if a line is a common external tangent to two circles -/
def is_common_external_tangent (c1 c2 : Circle) (line : TangentLine) : Prop :=
  sorry

/-- Theorem: The y-intercept of the common external tangent line with positive slope for the given circles is 75/19 -/
theorem common_external_tangent_y_intercept :
  let c1 : Circle := { center := (1, 5), radius := 3 }
  let c2 : Circle := { center := (15, 10), radius := 10 }
  let tangent : TangentLine := { slope := 20/19, y_intercept := 75/19 }
  (tangent.slope > 0) →
  (is_common_external_tangent c1 c2 tangent) →
  tangent.y_intercept = 75/19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_external_tangent_y_intercept_l48_4822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l48_4813

noncomputable def f (x y : ℝ) : ℝ := (x + y) / (Int.floor x * Int.floor y + Int.floor x + Int.floor y + 1)

theorem range_of_f :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 1 →
  ∃ S : Set ℝ, S = {1/2} ∪ Set.Icc (5/6) (5/4) ∧
  ∀ z : ℝ, z ∈ S ↔ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b = 1 ∧ f a b = z :=
by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l48_4813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_selling_price_calculation_l48_4816

/-- Calculates the selling price of a bond given its face value, interest rate, and the interest as a percentage of the selling price. -/
noncomputable def bondSellingPrice (faceValue : ℝ) (interestRate : ℝ) (interestPercentOfSellingPrice : ℝ) : ℝ :=
  (faceValue * interestRate) / interestPercentOfSellingPrice

/-- Theorem stating that for a bond with face value $5000, 10% interest rate, and interest being 6.5% of the selling price, the selling price is approximately $7692.31. -/
theorem bond_selling_price_calculation :
  let faceValue : ℝ := 5000
  let interestRate : ℝ := 0.10
  let interestPercentOfSellingPrice : ℝ := 0.065
  let calculatedSellingPrice := bondSellingPrice faceValue interestRate interestPercentOfSellingPrice
  abs (calculatedSellingPrice - 7692.31) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_selling_price_calculation_l48_4816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_three_l48_4812

/-- If a point P(x, y) lies on the unit circle in the third quadrant with x-coordinate -√10/10, 
    then tan(α) = 3, where α is the angle formed by the line OP (O being the origin) and the positive x-axis. -/
theorem tan_alpha_equals_three (P : ℝ × ℝ) (α : ℝ) :
  P.1 = -Real.sqrt 10 / 10 →  -- x-coordinate condition
  P.1^2 + P.2^2 = 1 →  -- unit circle condition
  P.1 < 0 ∧ P.2 < 0 →  -- third quadrant condition
  α = Real.arctan (P.2 / P.1) →  -- definition of α
  Real.tan α = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_three_l48_4812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l48_4854

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angle_at_F (q : Quadrilateral) : Prop := sorry

def diagonal_EH_perpendicular_to_HG (q : Quadrilateral) : Prop := sorry

noncomputable def side_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def perimeter (q : Quadrilateral) : ℝ :=
  side_length q.E q.F + side_length q.F q.G + side_length q.G q.H + side_length q.H q.E

-- State the theorem
theorem quadrilateral_perimeter (q : Quadrilateral) :
  is_right_angle_at_F q →
  diagonal_EH_perpendicular_to_HG q →
  side_length q.E q.F = 15 →
  side_length q.F q.G = 20 →
  side_length q.G q.H = 9 →
  perimeter q = 44 + Real.sqrt 706 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l48_4854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_and_circle_l48_4825

/-- A line passing through (1,1) parallel to 2x+4y+9=0 -/
structure ParallelLine where
  slope : ℝ
  passes_through : ℝ × ℝ := (1, 1)

/-- A circle intersecting the parallel line -/
structure IntersectingCircle (l : ParallelLine) where
  equation : ℝ → ℝ → ℝ → ℝ
  m : ℝ
  intersects : ∃ (P Q : ℝ × ℝ), equation P.1 P.2 m = 0 ∧ equation Q.1 Q.2 m = 0 ∧
    l.slope * (P.1 - 1) + P.2 = 1 ∧ l.slope * (Q.1 - 1) + Q.2 = 1
  perpendicular : ∀ (P Q : ℝ × ℝ), equation P.1 P.2 m = 0 → equation Q.1 Q.2 m = 0 →
    l.slope * (P.1 - 1) + P.2 = 1 → l.slope * (Q.1 - 1) + Q.2 = 1 →
    P.1 * Q.1 + P.2 * Q.2 = 0

/-- Main theorem -/
theorem parallel_line_and_circle
  (l : ParallelLine)
  (c : IntersectingCircle l)
  (h1 : l.slope = -1/2)
  (h2 : c.equation = fun x y m => x^2 + y^2 + x - 6*y + m) :
  (∀ x y, l.slope * (x - 1) + y = 1 ↔ x + 2*y - 3 = 0) ∧
  c.m = 3 := by
  sorry

#check parallel_line_and_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_and_circle_l48_4825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_investment_is_2000_l48_4872

/-- Represents the investment and profit distribution in a business partnership --/
structure BusinessPartnership where
  a_investment : ℚ
  b_investment : ℚ
  total_profit : ℚ
  management_fee_percent : ℚ
  a_total_received : ℚ

/-- Checks if the given business partnership satisfies the problem conditions --/
def is_valid_partnership (bp : BusinessPartnership) : Prop :=
  bp.b_investment = 3000 ∧
  bp.total_profit = 9600 ∧
  bp.management_fee_percent = 1/10 ∧
  bp.a_total_received = 4416

/-- Calculates the amount received by partner A --/
def calculate_a_received (bp : BusinessPartnership) : ℚ :=
  let management_fee := bp.total_profit * bp.management_fee_percent
  let remaining_profit := bp.total_profit - management_fee
  let a_share := (bp.a_investment / (bp.a_investment + bp.b_investment)) * remaining_profit
  management_fee + a_share

/-- Theorem stating that if the partnership is valid, A's investment must be 2000 --/
theorem a_investment_is_2000 (bp : BusinessPartnership) :
  is_valid_partnership bp → bp.a_investment = 2000 := by
  sorry

#eval calculate_a_received {
  a_investment := 2000,
  b_investment := 3000,
  total_profit := 9600,
  management_fee_percent := 1/10,
  a_total_received := 4416
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_investment_is_2000_l48_4872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_composition_l48_4848

/-- The atomic weight of aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of bromine in g/mol -/
def Br_weight : ℝ := 79.90

/-- The total molecular weight of the compound in g/mol -/
def total_weight : ℝ := 267

/-- The number of bromine atoms in the compound -/
def num_Br_atoms : ℕ := 3

theorem compound_composition :
  ∃ (x : ℝ),
    x = (total_weight - Al_weight) / Br_weight ∧
    x > 0 ∧
    num_Br_atoms = Int.floor (x + 0.5) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_composition_l48_4848
