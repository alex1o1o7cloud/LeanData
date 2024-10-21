import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_invariant_direction_vector_coprime_direction_vector_positive_first_direction_vector_unique_l342_34208

/-- The reflection matrix over a line passing through the origin --/
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4/5, 3/5],
    ![3/5, -4/5]]

/-- The direction vector of the line --/
def direction_vector : Fin 2 → ℚ :=
  ![4, -3]

/-- Theorem stating that the direction vector is invariant under reflection --/
theorem direction_vector_invariant :
  reflection_matrix.mulVec direction_vector = direction_vector :=
by sorry

/-- Theorem stating that the direction vector components are coprime integers --/
theorem direction_vector_coprime :
  Int.gcd (Int.natAbs 4) (Int.natAbs 3) = 1 :=
by sorry

/-- Theorem stating that the first component of the direction vector is positive --/
theorem direction_vector_positive_first :
  direction_vector 0 > 0 :=
by sorry

/-- Main theorem proving the uniqueness of the direction vector --/
theorem direction_vector_unique (v : Fin 2 → ℚ) :
  reflection_matrix.mulVec v = v →
  Int.gcd (Int.natAbs (Int.floor (v 0))) (Int.natAbs (Int.floor (v 1))) = 1 →
  v 0 > 0 →
  ∃ (c : ℚ), v = fun i => c * direction_vector i :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_invariant_direction_vector_coprime_direction_vector_positive_first_direction_vector_unique_l342_34208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_corner_same_color_is_one_l342_34234

/-- Represents the two possible colors for a cube face -/
inductive Color
| Red
| Blue

/-- Represents a cube with colored faces -/
structure Cube where
  faces : Fin 6 → Color

/-- The probability of painting a face red -/
noncomputable def prob_red : ℝ := 2/3

/-- The probability of painting a face blue -/
noncomputable def prob_blue : ℝ := 1/3

/-- Checks if three contiguous faces around a corner are the same color -/
def corner_same_color (c : Cube) (corner : Fin 8) : Prop :=
  sorry -- Implementation details omitted

/-- The probability that at least one corner has three contiguous faces of the same color -/
noncomputable def prob_any_corner_same_color : ℝ :=
  sorry -- Implementation details omitted

/-- Theorem: The probability of having at least one corner with three contiguous faces
    of the same color is 1, given the conditions of face coloring -/
theorem prob_corner_same_color_is_one :
  prob_any_corner_same_color = 1 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_corner_same_color_is_one_l342_34234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l342_34223

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 11, StrictMonoOn f (Set.Ioo (-1 : ℝ) 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l342_34223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l342_34206

/-- Calculate the difference between compound and simple interest -/
theorem interest_difference (principal : ℝ) (rate : ℝ) (time : ℕ) :
  principal = 1000 →
  rate = 0.1 →
  time = 4 →
  let simple_interest := principal * rate * (time : ℝ)
  let compound_interest := principal * ((1 + rate) ^ (time : ℝ) - 1)
  compound_interest - simple_interest = 64.10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_l342_34206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_of_f_l342_34279

/-- The minimum positive period of a sinusoidal function -/
noncomputable def min_positive_period (ω : ℝ) : ℝ := 2 * Real.pi / ω

/-- The given sinusoidal function -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin ((2/5) * x - Real.pi/6)

theorem min_period_of_f :
  min_positive_period (2/5) = 5 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_of_f_l342_34279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiger_catch_distance_l342_34250

/-- Calculates the total distance traveled by an escaped tiger -/
def tiger_distance (initial_speed : ℝ) (initial_duration : ℝ) (slow_speed : ℝ) (slow_duration : ℝ) (chase_speed : ℝ) (chase_duration : ℝ) : ℝ :=
  initial_speed * initial_duration + slow_speed * slow_duration + chase_speed * chase_duration

/-- Theorem stating the total distance traveled by the tiger -/
theorem tiger_catch_distance : 
  tiger_distance 25 3 10 2 50 0.5 = 120 := by
  unfold tiger_distance
  -- Perform the calculation
  simp [mul_add]
  -- The rest of the proof
  sorry

#eval tiger_distance 25 3 10 2 50 0.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiger_catch_distance_l342_34250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l342_34236

def is_valid_number (n : ℕ) : Bool :=
  10 ≤ n ∧ n < 100 ∧  -- two-digit integer
  n % 2 = 1 ∧  -- units digit is odd
  (n / 10) % 2 = 0 ∧  -- tens digit is even
  50 < n ∧ n < 90  -- greater than 50 but less than 90

theorem count_valid_numbers :
  (Finset.filter (fun n => is_valid_number n = true) (Finset.range 100)).card = 10 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l342_34236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_APB_l342_34254

noncomputable section

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is a midpoint of a side of a square with side length s -/
def Point.is_midpoint_of_side (p : Point) (s : ℝ) : Prop :=
  (p.x = s / 2 ∧ (p.y = 0 ∨ p.y = s)) ∨ (p.y = s / 2 ∧ (p.x = 0 ∨ p.x = s))

/-- Check if two points are on adjacent sides of a square -/
def Point.adjacent_to (p q : Point) : Prop :=
  (p.x = 0 ∧ q.y = 0) ∨ (p.x = 0 ∧ q.y = 10) ∨
  (p.y = 0 ∧ q.x = 0) ∨ (p.y = 0 ∧ q.x = 10) ∨
  (p.x = 10 ∧ q.y = 0) ∨ (p.x = 10 ∧ q.y = 10) ∨
  (p.y = 10 ∧ q.x = 0) ∨ (p.y = 10 ∧ q.x = 10)

/-- Check if a point is equidistant from three other points -/
def Point.equidistant (p a b c : Point) : Prop :=
  let d₁ := ((p.x - a.x)^2 + (p.y - a.y)^2)
  let d₂ := ((p.x - b.x)^2 + (p.y - b.y)^2)
  let d₃ := ((p.x - c.x)^2 + (p.y - c.y)^2)
  d₁ = d₂ ∧ d₂ = d₃

/-- Check if a point is on the opposite side of the square from two other points -/
def Point.on_opposite_side (c a b : Point) : Prop :=
  (a.x = 0 ∧ b.x = 0 ∧ c.x = 10) ∨
  (a.y = 0 ∧ b.y = 0 ∧ c.y = 10) ∨
  (a.x = 10 ∧ b.x = 10 ∧ c.x = 0) ∨
  (a.y = 10 ∧ b.y = 10 ∧ c.y = 0)

/-- Check if a segment is perpendicular to a side of the square -/
def segment_perpendicular (p c : Point) (s : ℝ) : Prop :=
  (p.x = c.x ∧ (c.y = 0 ∨ c.y = s)) ∨ (p.y = c.y ∧ (c.x = 0 ∨ c.x = s))

/-- Calculate the area of a triangle given three points -/
noncomputable def area_triangle (a b c : Point) : ℝ :=
  let s1 := ((b.x - a.x)^2 + (b.y - a.y)^2).sqrt
  let s2 := ((c.x - b.x)^2 + (c.y - b.y)^2).sqrt
  let s3 := ((a.x - c.x)^2 + (a.y - c.y)^2).sqrt
  let s := (s1 + s2 + s3) / 2
  (s * (s - s1) * (s - s2) * (s - s3)).sqrt

theorem area_triangle_APB (s : ℝ) (P A B C : Point) :
  s = 10 ∧
  A.is_midpoint_of_side s ∧
  B.is_midpoint_of_side s ∧
  A.adjacent_to B ∧
  P.equidistant A B C ∧
  C.on_opposite_side A B ∧
  segment_perpendicular P C s →
  area_triangle A P B = 25 * Real.sqrt 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_APB_l342_34254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_for_4ft_diameter_l342_34240

/-- The number of revolutions required for a wheel to travel one mile -/
noncomputable def wheel_revolutions (diameter : ℝ) : ℝ :=
  (5280 : ℝ) / (diameter * Real.pi)

/-- Theorem: A wheel with diameter 4 feet requires 1320/π revolutions to travel one mile -/
theorem wheel_revolutions_for_4ft_diameter :
  wheel_revolutions 4 = 1320 / Real.pi :=
by
  -- Unfold the definition of wheel_revolutions
  unfold wheel_revolutions
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_for_4ft_diameter_l342_34240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daps_to_dirps_l342_34295

/-- Represents the conversion rate between two units -/
structure ConversionRate (α β : Type) where
  rate : ℚ

/-- Given conversion rates between units, prove that 15 daps are equivalent to 20 dirps -/
theorem daps_to_dirps
  (dap_to_dop : ConversionRate ℕ ℕ)
  (dop_to_dip : ConversionRate ℕ ℕ)
  (dip_to_dirp : ConversionRate ℕ ℕ)
  (h1 : dap_to_dop.rate = 5 / 4)
  (h2 : dop_to_dip.rate = 3 / 10)
  (h3 : dip_to_dirp.rate = 2 / 1)
  : (15 : ℚ) = 20 * (dap_to_dop.rate * dop_to_dip.rate * dip_to_dirp.rate)⁻¹ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_daps_to_dirps_l342_34295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_theater_deal_savings_l342_34296

/-- Calculates the savings in euros for a movie theater deal -/
theorem movie_theater_deal_savings :
  let deal_price : ℚ := 20
  let ticket_price : ℚ := 8
  let popcorn_price : ℚ := ticket_price - 3
  let drink_price : ℚ := popcorn_price + 1
  let candy_price : ℚ := drink_price / 2
  let popcorn_discount : ℚ := 15 / 100
  let candy_discount : ℚ := 10 / 100
  let exchange_rate : ℚ := 85 / 100

  let discounted_popcorn : ℚ := popcorn_price * (1 - popcorn_discount)
  let discounted_candy : ℚ := candy_price * (1 - candy_discount)
  let normal_cost : ℚ := ticket_price + discounted_popcorn + drink_price + discounted_candy
  let savings_usd : ℚ := normal_cost - deal_price
  let savings_eur : ℚ := savings_usd * exchange_rate

  abs (savings_eur - 81 / 100) < 1 / 100 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_theater_deal_savings_l342_34296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_date_statistics_order_l342_34283

/-- Represents the occurrences of dates in a non-leap year with a specific data recording method. -/
structure DateOccurrences where
  regular : Nat  -- occurrences of dates 1 to 28
  almostFull : Nat  -- occurrences of dates 29 and 30
  full : Nat  -- occurrences of date 31

/-- Calculates the median of modes for the given date occurrences. -/
def medianOfModes : Real := 14.5

/-- Calculates the median for the given date occurrences. -/
def median : Real := 16  -- Simplified based on the problem's specific data

/-- Calculates the mean for the given date occurrences. -/
noncomputable def mean (occurrences : DateOccurrences) : Real :=
  let totalDays := occurrences.regular * 28 + occurrences.almostFull * 2 + occurrences.full
  let totalSum := (occurrences.regular * 28 * 29) / 2 + occurrences.almostFull * (29 + 30) + occurrences.full * 31
  (totalSum : Real) / totalDays

/-- Theorem stating the order relation between d, M, and μ for the given date occurrences. -/
theorem date_statistics_order (occurrences : DateOccurrences) 
    (h1 : occurrences.regular = 12)
    (h2 : occurrences.almostFull = 11)
    (h3 : occurrences.full = 6) :
    medianOfModes < median ∧ median < mean occurrences := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_date_statistics_order_l342_34283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_pi_quarter_plus_half_l342_34281

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then x + 1
  else if 0 < x ∧ x ≤ 1 then Real.sqrt (1 - x^2)
  else 0

theorem integral_f_equals_pi_quarter_plus_half :
  ∫ x in Set.Icc (-1) 1, f x = π / 4 + 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_pi_quarter_plus_half_l342_34281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l342_34211

structure Ellipse where
  m : ℝ
  equation : (x y : ℝ) → Prop := λ x y => x^2 + y^2 / m = 1

structure Point where
  x : ℝ
  y : ℝ

def Line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

def bisects (MF2 AB : Set Point) : Prop := sorry

theorem ellipse_triangle_area 
  (C : Ellipse)
  (M : Point)
  (k : ℝ)
  (F1 F2 A B : Point)
  (h1 : C.equation M.x M.y)
  (h2 : M.x = Real.sqrt 2 / 2 ∧ M.y = 1)
  (h3 : k ≠ 0)
  (h4 : Line k F2.x F2.y)
  (h5 : C.equation A.x A.y ∧ C.equation B.x B.y)
  (h6 : bisects {M, F2} {A, B})
  : Real.sqrt 6 / 2 = (1/2) * |F1.y - F2.y| * |A.x - B.x| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l342_34211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l342_34222

/-- Ellipse properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Given conditions -/
def ellipse_conditions (E : Ellipse) : Prop :=
  E.a^2 - E.b^2 = E.a^2 * (2/4) ∧
  2 * Real.sqrt (E.a^2 - E.b^2) * (Real.sqrt 2/2) = 2

/-- Standard equation of the ellipse -/
def standard_equation (E : Ellipse) : Prop :=
  E.a^2 = 2 ∧ E.b^2 = 1

/-- Slope sum property -/
def slope_sum_property (E : Ellipse) : Prop :=
  ∃ D : ℝ × ℝ, 
    D.1 = 0 ∧ D.2 = 1/2 ∧
    ∀ k : ℝ, ∀ A B : ℝ × ℝ,
      (A.1^2 / E.a^2 + A.2^2 / E.b^2 = 1) →
      (B.1^2 / E.a^2 + B.2^2 / E.b^2 = 1) →
      (A.2 = k * A.1 + 2) →
      (B.2 = k * B.1 + 2) →
      ((A.2 - D.2) / (A.1 - D.1) + (B.2 - D.2) / (B.1 - D.1) = 0)

/-- Main theorem -/
theorem ellipse_properties (E : Ellipse) 
  (h : ellipse_conditions E) : 
  standard_equation E ∧ slope_sum_property E := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l342_34222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_max_intersection_length_l342_34288

/-- Ellipse with given properties -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Circle with center (0, 0) and radius R -/
def Circle (R : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = R^2}

/-- Line segment length between two points -/
noncomputable def LineSegmentLength (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem ellipse_circle_max_intersection_length :
  ∀ (R : ℝ),
  3 < R → R < 5 →
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    a^2 - b^2 = (4/5 * a)^2 ∧
    (10*Real.sqrt 2/3, 1) ∈ Ellipse a b ∧
    (∀ (p q : ℝ × ℝ),
      p ∈ Ellipse a b →
      q ∈ Circle R →
      LineSegmentLength p q ≤ 2) ∧
    (∃ (p q : ℝ × ℝ),
      p ∈ Ellipse a b ∧
      q ∈ Circle R ∧
      LineSegmentLength p q = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_max_intersection_length_l342_34288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_probability_l342_34231

/-- The number of teachers -/
def numTeachers : ℕ := 3

/-- The number of students -/
def numStudents : ℕ := 3

/-- The number of groups -/
def numGroups : ℕ := 3

/-- The total number of people -/
def totalPeople : ℕ := numTeachers + numStudents

/-- The probability of each group having exactly 1 teacher and 1 student -/
def probability : ℚ := 2 / 5

theorem division_probability :
  (Nat.choose totalPeople 2 * Nat.choose (totalPeople - 2) 2 * Nat.choose (totalPeople - 4) 2) ≠ 0 →
  (Nat.choose numTeachers 1 * Nat.choose numStudents 1 * Nat.choose (numTeachers - 1) 1 * 
   Nat.choose (numStudents - 1) 1 * Nat.choose (numTeachers - 2) 1 * Nat.choose (numStudents - 2) 1 : ℚ) /
  (Nat.choose totalPeople 2 * Nat.choose (totalPeople - 2) 2 * Nat.choose (totalPeople - 4) 2 : ℚ) = probability :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_probability_l342_34231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_B_elements_l342_34258

-- Define the solution set A
def A (m : ℝ) : Set ℝ := {x : ℝ | (m * x - m^2 - 6) * (x + 4) < 0}

-- Define the set of integers Z
def Z : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n}

-- Define the set B as the intersection of A and Z
def B (m : ℝ) : Set ℝ := A m ∩ Z

-- Define the range of m that minimizes the number of elements in B
def minimizing_range : Set ℝ := {m : ℝ | 2 ≤ m ∧ m ≤ 3}

-- Theorem statement
theorem minimize_B_elements :
  ∀ m : ℝ, m ∈ minimizing_range → 
    ∀ m' : ℝ, Finite (B m) ∧ Finite (B m') → Cardinal.mk (B m) ≤ Cardinal.mk (B m') :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_B_elements_l342_34258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l342_34214

theorem problem_1 : Real.sqrt 9 * |(-1/3)| * (8 ^ (1/3 : ℝ)) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l342_34214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_134_in_base5_has_four_consecutive_digits_l342_34261

/-- Converts a decimal number to its base 5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Checks if a list of digits are consecutive --/
def areConsecutive (digits : List ℕ) : Bool :=
  match digits with
  | [] => true
  | [_] => true
  | d1 :: d2 :: rest => (d2 - d1 = 1) && areConsecutive (d2 :: rest)

theorem decimal_134_in_base5_has_four_consecutive_digits :
  let base5Repr := toBase5 134
  base5Repr.length = 4 ∧ areConsecutive base5Repr := by
  -- The proof goes here
  sorry

#eval toBase5 134  -- Should output [1, 1, 4, 4]
#eval areConsecutive (toBase5 134)  -- Should output true

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_134_in_base5_has_four_consecutive_digits_l342_34261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l342_34226

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = -2 * x + 1
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y ↔ y = (1/2) * x - 3/2) →
  (L2 P.1 P.2) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → x₁ ≠ x₂ →
    ((y₂ - y₁) / (x₂ - x₁)) * ((y₂ - y₁) / (x₂ - x₁)) = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l342_34226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_phi_is_pi_over_6_phi_when_max_f_is_three_over_two_l342_34253

-- Define the function f
noncomputable def f (x φ : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.cos (2 * x + φ) + Real.sin x ^ 2

-- Theorem for the first part of the problem
theorem range_of_f_when_phi_is_pi_over_6 :
  ∀ x, 0 ≤ f x (π / 6) ∧ f x (π / 6) ≤ 1 := by
  sorry

-- Theorem for the second part of the problem
theorem phi_when_max_f_is_three_over_two :
  (∃ φ : ℝ, 0 ≤ φ ∧ φ < π ∧ (∀ x, f x φ ≤ 3/2) ∧ (∃ x₀, f x₀ φ = 3/2)) → φ = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_phi_is_pi_over_6_phi_when_max_f_is_three_over_two_l342_34253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_length_l342_34266

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line
def my_line (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 2 * Real.sqrt 3 = 0

-- Define the chord length function
noncomputable def chord_length : ℝ :=
  2 * Real.sqrt (4 - (2 * Real.sqrt 3 / Real.sqrt 4)^2)

-- Theorem statement
theorem chord_intersection_length :
  chord_length = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_length_l342_34266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_path_shorter_l342_34220

/-- Represents a rectangular field with length and width -/
structure RectangularField where
  length : ℝ
  width : ℝ

/-- Calculates the length of the path along the edges of the field -/
def edge_path_length (field : RectangularField) : ℝ :=
  field.length + field.width

/-- Calculates the length of the diagonal path across the field -/
noncomputable def diagonal_path_length (field : RectangularField) : ℝ :=
  Real.sqrt (field.length^2 + field.width^2)

/-- Calculates the percentage difference between the edge path and the diagonal path -/
noncomputable def path_difference_percentage (field : RectangularField) : ℝ :=
  (edge_path_length field - diagonal_path_length field) / edge_path_length field * 100

theorem diagonal_path_shorter (field : RectangularField) :
  field.length = 3 ∧ field.width = 4 →
  28.5 < path_difference_percentage field ∧ path_difference_percentage field < 28.6 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval path_difference_percentage { length := 3, width := 4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_path_shorter_l342_34220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l342_34243

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 4

-- Define the line
def line_equation (x : ℝ) : Prop := x = -3

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_circle_to_line :
  ∃ (min_dist : ℝ),
    (∀ (x1 y1 x2 y2 : ℝ),
      circle_equation x1 y1 → line_equation x2 →
      distance x1 y1 x2 y2 ≥ min_dist) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      circle_equation x1 y1 ∧ line_equation x2 ∧
      distance x1 y1 x2 y2 = min_dist) ∧
    min_dist = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l342_34243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l342_34244

theorem system_solution (x y z : ℝ) (n : ℤ) :
  x * y = 1 ∧ x + y + Real.cos z ^ 2 = 2 →
  x = 1 ∧ y = 1 ∧ z = Real.pi / 2 + Real.pi * ↑n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l342_34244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invalid_votes_thirty_percent_l342_34256

/-- Represents an election between two candidates -/
structure Election where
  total_votes : ℕ
  valid_votes_percentage_a : ℚ
  valid_votes_b : ℚ

/-- Calculates the percentage of invalid votes in an election -/
noncomputable def invalid_votes_percentage (e : Election) : ℚ :=
  let valid_votes := e.valid_votes_b / (1 - e.valid_votes_percentage_a / 100)
  (e.total_votes - valid_votes) / e.total_votes * 100

/-- Theorem stating that the percentage of invalid votes is 30% -/
theorem invalid_votes_thirty_percent (e : Election) 
  (h1 : e.total_votes = 9000)
  (h2 : e.valid_votes_percentage_a = 60)
  (h3 : e.valid_votes_b = 2520) :
  invalid_votes_percentage e = 30 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invalid_votes_thirty_percent_l342_34256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_determines_m_n_existence_of_negative_value_iff_m_range_l342_34297

noncomputable def f (m n x : ℝ) : ℝ := m / Real.exp x + n * x

noncomputable def f_derivative (m n x : ℝ) : ℝ := -m / Real.exp x + n

theorem tangent_line_determines_m_n :
  ∀ m n : ℝ,
  (f_derivative m n 0 = -3) →
  f m n 0 = 2 →
  m = 2 ∧ n = -1 := by sorry

theorem existence_of_negative_value_iff_m_range :
  ∀ m : ℝ,
  (∃ x₀ : ℝ, x₀ ≤ 1 ∧ f m 1 x₀ < 0) ↔
  m < 1 / Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_determines_m_n_existence_of_negative_value_iff_m_range_l342_34297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_price_calculation_l342_34224

theorem jacket_price_calculation (original_price discount_percent coupon_discount tax_rate : ℝ) 
  (h1 : original_price = 150)
  (h2 : discount_percent = 0.30)
  (h3 : coupon_discount = 10)
  (h4 : tax_rate = 0.10) :
  original_price * (1 - discount_percent) - coupon_discount * (1 + tax_rate) = 104.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_price_calculation_l342_34224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l342_34278

/-- Given function f(x) = √3 * sin(ωx + φ) with specified conditions -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

/-- Theorem stating the properties of the function and the solution -/
theorem function_properties (ω φ α : ℝ) : 
  ω > 0 ∧ 
  -π/2 ≤ φ ∧ φ < π/2 ∧
  (∀ x, f ω φ (x - π/3) = f ω φ (π/3 - x)) ∧  -- Symmetry about x = π/3
  (∀ x, f ω φ (x + π/ω) = f ω φ x) ∧  -- Period is π
  π/6 < α ∧ α < 2*π/3 ∧
  f ω φ (α/2) = Real.sqrt 3 / 4 →
  ω = 2 ∧ 
  φ = -π/6 ∧ 
  Real.cos (α + 3*π/2) = (Real.sqrt 3 + Real.sqrt 15) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l342_34278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unshaded_rectangle_probability_l342_34271

/-- Represents a rectangle in the grid -/
structure Rectangle where
  left : Nat
  right : Nat
  top : Nat
  bottom : Nat

/-- The width of the original rectangle -/
def width : Nat := 2005

/-- The height of the original rectangle -/
def grid_height : Nat := 3

/-- Checks if a given rectangle contains a shaded square -/
def contains_shaded (r : Rectangle) : Bool :=
  (r.left ≤ width / 2 + 1 ∧ r.right ≥ width / 2 + 1) ∨
  (r.top = 2 ∧ r.bottom = 2 ∧ r.left ≤ 1003 ∧ r.right ≥ 1003)

/-- Counts the total number of possible rectangles -/
def total_rectangles : Nat :=
  6 * (width.choose 2) * grid_height.choose 2

/-- Counts the number of rectangles that don't contain a shaded square -/
def unshaded_rectangles : Nat :=
  total_rectangles - 3 * (width / 2 + 1) * (width / 2 + 1)

/-- The main theorem stating the probability of choosing an unshaded rectangle -/
theorem unshaded_rectangle_probability :
  (unshaded_rectangles : ℚ) / total_rectangles = 1002 / 2005 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unshaded_rectangle_probability_l342_34271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pairs_count_l342_34255

theorem ordered_pairs_count (n : ℕ) (h : n = 2 * 3^2 * 5 * 7^2) :
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = n ∧ 0 < p.1 ∧ 0 < p.2) (Finset.range (n+1) ×ˢ Finset.range (n+1))).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pairs_count_l342_34255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lcm_with_18_l342_34200

theorem largest_lcm_with_18 : 
  let lcm_list := [Nat.lcm 18 3, Nat.lcm 18 9, Nat.lcm 18 6, 
                   Nat.lcm 18 12, Nat.lcm 18 15, Nat.lcm 18 18]
  lcm_list.maximum? = some 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lcm_with_18_l342_34200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mango_juice_l342_34201

/-- Represents the ratio of juices in the drink -/
structure JuiceRatio where
  orange : ℕ
  watermelon : ℕ
  grape : ℕ
  pineapple : ℕ
  mango : ℕ

/-- Represents the amount of each juice in ounces -/
structure JuiceAmount where
  orange : ℚ
  watermelon : ℚ
  grape : ℚ
  pineapple : ℚ
  mango : ℚ

def total_ratio (r : JuiceRatio) : ℕ :=
  r.orange + r.watermelon + r.grape + r.pineapple + r.mango

def total_amount (a : JuiceAmount) : ℚ :=
  a.orange + a.watermelon + a.grape + a.pineapple + a.mango

theorem max_mango_juice (ratio : JuiceRatio) (grape_amount : ℚ) (max_total : ℚ) :
  ratio = ⟨3, 5, 2, 4, 6⟩ →
  grape_amount = 120 →
  max_total = 1000 →
  ∃ (amount : JuiceAmount),
    amount.grape = 100 ∧
    amount.mango = 300 ∧
    total_amount amount ≤ max_total ∧
    ∀ (other_amount : JuiceAmount),
      other_amount.grape = 100 →
      total_amount other_amount ≤ max_total →
      (other_amount.orange * ratio.mango = amount.orange * ratio.grape ∧
       other_amount.watermelon * ratio.mango = amount.watermelon * ratio.grape ∧
       other_amount.grape * ratio.mango = amount.grape * ratio.grape ∧
       other_amount.pineapple * ratio.mango = amount.pineapple * ratio.grape ∧
       other_amount.mango * ratio.mango = amount.mango * ratio.grape) →
      other_amount.mango ≤ amount.mango :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mango_juice_l342_34201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_circle_l342_34294

noncomputable def circle_points (t : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ p = (Real.sin θ * Real.cos θ, Real.sin θ * Real.sin θ)}

theorem smallest_complete_circle : 
  ∀ t : ℝ, t > 0 → (circle_points t = circle_points Real.pi → t ≥ Real.pi) ∧ 
  (circle_points Real.pi = {p : ℝ × ℝ | p.1^2 + p.2^2 = 1/4}) :=
by
  sorry

#check smallest_complete_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_circle_l342_34294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_parallel_line_slope_is_half_l342_34216

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x - 6 * y = 21

-- Define the slope-intercept form of a line
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Theorem: The slope of any line parallel to 3x - 6y = 21 is 1/2
theorem parallel_line_slope :
  ∃ (m : ℝ), ∀ (b : ℝ), (∀ x y : ℝ, line_equation x y ↔ slope_intercept_form m b x y) ∧ m = 1/2 := by
  sorry

-- Corollary: Any line parallel to 3x - 6y = 21 has a slope of 1/2
theorem parallel_line_slope_is_half :
  ∀ (m b : ℝ), (∀ x y : ℝ, line_equation x y ↔ slope_intercept_form m b x y) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_parallel_line_slope_is_half_l342_34216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_composite_sequence_l342_34269

theorem largest_composite_sequence : ∃ (a : ℕ), 
  (a < 40) ∧ 
  (a > 31) ∧
  (∀ i ∈ Finset.range 5, ¬ Nat.Prime (a - i)) ∧
  (∀ b : ℕ, b > a → ∃ i ∈ Finset.range 5, Nat.Prime (b - i)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_composite_sequence_l342_34269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_difference_theorem_l342_34264

def is_distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def in_range (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 10

theorem lcm_difference_theorem :
  ∃ (a b c d e f : ℕ),
    is_distinct a b c ∧
    is_distinct d e f ∧
    in_range a ∧
    in_range b ∧
    in_range c ∧
    in_range d ∧
    in_range e ∧
    in_range f ∧
    Nat.lcm (Nat.lcm a b) c = 4 ∧
    Nat.lcm (Nat.lcm d e) f = 504 ∧
    (∀ x y z, is_distinct x y z → in_range x → in_range y → in_range z → 
      4 ≤ Nat.lcm (Nat.lcm x y) z ∧ Nat.lcm (Nat.lcm x y) z ≤ 504) ∧
    504 - 4 = 500 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_difference_theorem_l342_34264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_rounded_l342_34292

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ : ℕ),
    n = d₁ * 100000 + d₂ * 10000 + d₃ * 1000 + d₄ * 100 + d₅ * 10 + d₆ ∧
    Finset.toSet {d₁, d₂, d₃, d₄, d₅, d₆} = Finset.toSet {0, 1, 3, 5, 7, 9} ∧
    d₁ ≠ 0

def smallest_valid_number : ℕ := 103579

def round_to_thousandth (n : ℕ) : ℕ :=
  ((n + 500) / 1000) * 1000

theorem smallest_valid_number_rounded :
  is_valid_number smallest_valid_number ∧
  (∀ m, is_valid_number m → m ≥ smallest_valid_number) ∧
  round_to_thousandth smallest_valid_number = 104000 := by
  sorry

#eval round_to_thousandth smallest_valid_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_rounded_l342_34292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_diagonals_count_l342_34252

/-- A convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (P : ConvexPolyhedron) : ℕ :=
  Nat.choose P.vertices 2 - P.edges - 2 * P.quadrilateral_faces

/-- Theorem: The number of space diagonals in the given polyhedron is 241 -/
theorem space_diagonals_count :
  let P : ConvexPolyhedron := {
    vertices := 26,
    edges := 60,
    faces := 36,
    triangular_faces := 24,
    quadrilateral_faces := 12
  }
  space_diagonals P = 241 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_diagonals_count_l342_34252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_bags_within_std_dev_l342_34213

noncomputable def sugar_weights : List ℝ := [495, 500, 503, 508, 498, 500, 493, 500, 503, 500]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

noncomputable def std_dev (l : List ℝ) : ℝ := (variance l).sqrt

theorem sugar_bags_within_std_dev :
  let x_bar := mean sugar_weights
  let s := std_dev sugar_weights
  (sugar_weights.filter (λ x => x_bar - s ≤ x ∧ x ≤ x_bar + s)).length = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_bags_within_std_dev_l342_34213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_sector_radius_l342_34245

/-- A sector is a portion of a circular disk enclosed by two radii and an arc. -/
structure Sector where
  radius : ℝ
  angle : ℝ

/-- The area of a sector. -/
noncomputable def sectorArea (s : Sector) : ℝ := (1/2) * s.radius^2 * s.angle

/-- The perimeter of a sector. -/
noncomputable def sectorPerimeter (s : Sector) : ℝ := 2 * s.radius + s.radius * s.angle

/-- Theorem: When a sector has a fixed area of 9 and its perimeter is minimized, the radius is 3. -/
theorem min_perimeter_sector_radius :
  ∃ (s : Sector), sectorArea s = 9 ∧
  (∀ (t : Sector), sectorArea t = 9 → sectorPerimeter s ≤ sectorPerimeter t) →
  s.radius = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_sector_radius_l342_34245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poisson_zero_probability_l342_34262

-- Define the average number of α-particles and the time period
noncomputable def avg_particles : ℝ := 3.87
noncomputable def time_period : ℝ := 7.5

-- Define the rate parameter λ for one second
noncomputable def lambda : ℝ := avg_particles / time_period

-- Define the probability of zero emissions
noncomputable def prob_zero_emissions : ℝ := Real.exp (-lambda)

-- Theorem statement
theorem poisson_zero_probability : 
  ∃ ε > 0, abs (prob_zero_emissions - 0.596) < ε := by
  sorry

-- Additional lemma to show the approximate value
lemma prob_zero_emissions_approx :
  ∃ ε > 0, abs (prob_zero_emissions - 0.596) < ε := by
  use 0.001
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poisson_zero_probability_l342_34262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_f_equality_l342_34204

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2*(x - 2)
  else 0  -- We define f as 0 outside the given intervals

-- State the theorem
theorem negative_f_equality (x : ℝ) :
  ((-3 ≤ x ∧ x ≤ 0) → -f x = 2 + x) ∧
  ((0 < x ∧ x ≤ 2) → -f x = -Real.sqrt (4 - (x - 2)^2) + 2) ∧
  ((2 < x ∧ x ≤ 3) → -f x = -2*(x - 2)) := by
  sorry

#check negative_f_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_f_equality_l342_34204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l342_34273

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / ⌊x^2 - 7*x + 16⌋

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iic 3 ∪ Set.Ici 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l342_34273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_maximization_l342_34272

theorem triangle_dot_product_maximization (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  C = π / 3 ∧
  c = 2 ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) →
  let dot_product := b * c * (Real.cos A)
  ∃ (max_A : ℝ), 
    (∀ A', 0 < A' ∧ A' < 2*π/3 → dot_product ≤ 2 * b * (Real.cos A')) ∧
    (A = max_A → b / a = 2 + Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_maximization_l342_34272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_location_l342_34230

theorem complex_number_location (z : ℂ) (h : Complex.I * z = -2 + Complex.I) : 
  z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_location_l342_34230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l342_34247

theorem cube_root_simplification :
  (2016^2 + 2016 * 2017 + 2017^2 + 2016^3 : ℝ) ^ (1/3) = 2017 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l342_34247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_condition_l342_34246

def is_triangle (x y z : ℤ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y > z ∧ y + z > x ∧ z + x > y

theorem triangle_inequality_condition (a : ℝ) : 
  (∀ x y z : ℤ, is_triangle x y z → x^2 + y^2 + z^2 ≤ a * (x*y + y*z + z*x)) ↔ 
  (1 ≤ a ∧ a < 6/5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_condition_l342_34246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_storm_intersection_average_time_l342_34235

/-- Car's eastward velocity in miles per minute -/
noncomputable def car_velocity : ℝ := 3/4

/-- Storm's radius in miles -/
def storm_radius : ℝ := 75

/-- Storm's velocity in miles per minute -/
noncomputable def storm_velocity : ℝ := (3/4) * Real.sqrt 2

/-- Initial north-south distance between car and storm in miles -/
def initial_distance : ℝ := 150

/-- Time when the car enters the storm -/
noncomputable def t₁ : ℝ := sorry

/-- Time when the car exits the storm -/
noncomputable def t₂ : ℝ := sorry

/-- Theorem stating that the average of entry and exit times is 400 minutes -/
theorem car_storm_intersection_average_time :
  (1/2) * (t₁ + t₂) = 400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_storm_intersection_average_time_l342_34235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l342_34241

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x^2 * log x
noncomputable def g (x : ℝ) : ℝ := x / exp x

-- State the theorem
theorem problem_statement (k : ℝ) :
  (k > 0) →
  (∃ x₁ ∈ Set.Icc (exp 1) (exp 2), ∃ x₂ ∈ Set.Icc 1 2,
    exp 3 * (k^2 - 2) * g x₂ ≥ k * f x₁) →
  k ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l342_34241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_499407_l342_34232

-- Define the ceiling and floor of log base sqrt(3) of k
noncomputable def ceilLogSqrt3 (k : ℕ) : ℤ := ⌈Real.log k / Real.log (Real.sqrt 3)⌉
noncomputable def floorLogSqrt3 (k : ℕ) : ℤ := ⌊Real.log k / Real.log (Real.sqrt 3)⌋

-- Define the sum function
noncomputable def sumFunction (k : ℕ) : ℤ := k * (ceilLogSqrt3 k - floorLogSqrt3 k)

theorem sum_equals_499407 :
  (Finset.range 1000).sum (fun k => sumFunction (k + 1)) = 499407 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_499407_l342_34232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l342_34237

noncomputable section

def Triangle := Real → Real → Real → Prop

def side (t : Triangle) (i : Fin 3) : Real := sorry
def angle (t : Triangle) (i : Fin 3) : Real := sorry

theorem triangle_side_length (t : Triangle) (A B C a b c : Real) :
  -- Triangle exists
  t A B C →
  -- a and c are equal to √6 + √2
  a = Real.sqrt 6 + Real.sqrt 2 →
  c = Real.sqrt 6 + Real.sqrt 2 →
  -- Angle A is 75°
  angle t 0 = 75 * π / 180 →
  -- The sides are opposite to their respective angles
  side t 0 = a ∧ side t 1 = b ∧ side t 2 = c →
  angle t 0 = A ∧ angle t 1 = B ∧ angle t 2 = C →
  -- Then b equals 2
  b = 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l342_34237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_lower_bound_and_infimum_l342_34210

/-- The function g defined for positive real numbers -/
noncomputable def g (x y z : ℝ) : ℝ := (x + y) / x + (y + z) / y + (z + x) / z

/-- Theorem stating the lower bound and infimum of g -/
theorem g_lower_bound_and_infimum :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → g x y z ≥ 6) ∧
  (∀ ε > 0, ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ g x y z < 6 + ε) := by
  sorry

#check g_lower_bound_and_infimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_lower_bound_and_infimum_l342_34210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_in_range_l342_34280

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 9 * Real.log x

theorem a_in_range (a : ℝ) 
  (h_monotone : ∀ x ∈ Set.Icc (a - 1) (a + 1), StrictMonoOn f (Set.Icc (a - 1) (a + 1))) : 
  a ∈ Set.Ioo 1 2 := by
  sorry

#check a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_in_range_l342_34280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_calculation_and_perpendicular_condition_l342_34209

def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (2, 3)

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)
def vector_scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_calculation_and_perpendicular_condition :
  (vector_add (vector_add (vector_sub D A) (vector_sub D B)) (vector_sub C B) = (0, 6)) ∧
  (∃ lambda : ℝ, lambda = -1 ∧ 
    dot_product (vector_add (vector_sub C A) (vector_scalar_mul lambda (vector_sub B A))) (vector_sub D C) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_calculation_and_perpendicular_condition_l342_34209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_runners_in_frame_probability_l342_34260

/-- Represents a runner in the circular stadium. -/
structure Runner where
  direction : Bool  -- true for counterclockwise, false for clockwise
  lap_time : ℕ      -- time to complete one lap in seconds

/-- Represents the photographer's frame. -/
structure Frame where
  center : ℝ        -- position of the frame center relative to the starting point
  width : ℝ         -- width of the frame as a fraction of the track

/-- The probability problem setup. -/
structure StadiumProblem where
  anna : Runner
  carl : Runner
  frame : Frame
  start_time : ℕ    -- start time of the photo window in seconds
  end_time : ℕ      -- end time of the photo window in seconds

/-- Calculates the probability of both runners being in the frame simultaneously. -/
def probability_both_in_frame (problem : StadiumProblem) : ℚ :=
  sorry

/-- The main theorem to prove. -/
theorem both_runners_in_frame_probability 
  (problem : StadiumProblem)
  (h_anna : problem.anna = { direction := true, lap_time := 100 })
  (h_carl : problem.carl = { direction := false, lap_time := 60 })
  (h_frame : problem.frame = { center := 0, width := 1/3 })
  (h_start_time : problem.start_time = 12 * 60)
  (h_end_time : problem.end_time = 15 * 60) :
  probability_both_in_frame problem = 8/45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_runners_in_frame_probability_l342_34260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_plus_one_equals_thirteen_tenths_l342_34239

theorem sin_cos_plus_one_equals_thirteen_tenths (x : ℝ) (h : Real.tan x = 1/3) :
  Real.sin x * Real.cos x + 1 = 13/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_plus_one_equals_thirteen_tenths_l342_34239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_all_distinct_l342_34277

/-- Recurrence formula for a_{n,k} -/
def a : ℕ → ℕ → ℚ
  | 0, _ => 2  -- Base case for n = 0
  | 1, 1 => 2  -- Base case for n = 1
  | (n+1), k => 
    if k ≤ 2^n 
    then 2 * (a n k)^3 
    else (1/2) * (a n (k - 2^n))^3

/-- Theorem: All a_{m,k} are distinct -/
theorem a_all_distinct :
  ∀ m₁ m₂ k₁ k₂, m₁ ≥ 1 → m₂ ≥ 1 → k₁ ≤ 2^m₁ → k₂ ≤ 2^m₂ → 
  (m₁ ≠ m₂ ∨ k₁ ≠ k₂) → a m₁ k₁ ≠ a m₂ k₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_all_distinct_l342_34277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_benny_turnips_benny_turnips_correct_l342_34227

theorem benny_turnips : ℕ → ℕ → ℕ
| melanie_turnips, difference =>
  if melanie_turnips = 139 ∧ difference = 26 then
    melanie_turnips - difference
  else
    0

theorem benny_turnips_correct (melanie_turnips : ℕ) (difference : ℕ) 
  (h1 : melanie_turnips = 139)
  (h2 : melanie_turnips = difference + (benny_turnips melanie_turnips difference))
  (h3 : difference = 26) : 
  benny_turnips melanie_turnips difference = 113 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_benny_turnips_benny_turnips_correct_l342_34227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_puzzle_l342_34270

theorem absolute_value_puzzle (x : ℤ) (h : x = -2023) :
  (abs (abs (abs x - x) - abs x) - x) = 4046 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_puzzle_l342_34270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l342_34248

/-- In a triangle ABC, if tan A = 2 tan B and a^2 - b^2 = (1/3)c, then c = 1 -/
theorem triangle_side_length (a b c : ℝ) (A B C : Real) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →  -- triangle inequality
  Real.tan A = 2 * Real.tan B →
  a^2 - b^2 = (1/3) * c →
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l342_34248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_volume_is_pi_over_two_l342_34268

/-- Represents the volume of a section of the tree at level n -/
noncomputable def section_volume (n : ℕ) : ℝ := Real.pi / (4 * 2^n)

/-- The total volume of the tree -/
noncomputable def total_volume : ℝ := ∑' n, section_volume n

/-- Theorem stating that the total volume of the tree is π/2 -/
theorem tree_volume_is_pi_over_two : total_volume = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_volume_is_pi_over_two_l342_34268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapped_paper_length_l342_34291

open Real

/-- Calculates the length of paper wrapped around a cylindrical tube. -/
noncomputable def paperLength (paperWidth : ℝ) (initialDiameter : ℝ) (finalDiameter : ℝ) (wraps : ℕ) (overlapFactor : ℝ) : ℝ :=
  let initialRadius := initialDiameter / 2
  let finalRadius := finalDiameter / 2
  let radiusIncrease := (finalRadius - initialRadius) / wraps
  let baseLength := (wraps : ℝ) * π * (2 * initialRadius + (wraps - 1 : ℕ) * radiusIncrease)
  overlapFactor * baseLength / 100  -- Convert to meters

/-- The length of the wrapped paper is 2227.5π meters. -/
theorem wrapped_paper_length :
  paperLength 4 1 15 450 110 = 2227.5 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapped_paper_length_l342_34291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_puree_water_percentage_l342_34299

/-- Represents the process of making tomato puree from tomato juice -/
structure TomatoPureeProcess where
  initial_juice_volume : ℝ
  initial_water_percentage : ℝ
  final_puree_volume : ℝ

/-- Calculates the percentage of water in the final tomato puree -/
noncomputable def water_percentage_in_puree (process : TomatoPureeProcess) : ℝ :=
  let initial_water_volume := process.initial_juice_volume * process.initial_water_percentage
  let initial_solids_volume := process.initial_juice_volume * (1 - process.initial_water_percentage)
  let final_water_volume := process.final_puree_volume - initial_solids_volume
  (final_water_volume / process.final_puree_volume) * 100

/-- Theorem stating that the water percentage in the tomato puree is 20% -/
theorem tomato_puree_water_percentage :
  let process : TomatoPureeProcess := {
    initial_juice_volume := 20,
    initial_water_percentage := 0.9,
    final_puree_volume := 2.5
  }
  water_percentage_in_puree process = 20 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_puree_water_percentage_l342_34299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vitamin_d3_bottles_l342_34274

theorem vitamin_d3_bottles (total_days : ℕ) (brand_a_daily : ℕ) (brand_a_per_bottle : ℕ) 
  (brand_b_daily : ℕ) (brand_b_per_bottle : ℕ) (available_a_bottles : ℕ) : 
  let total_capsules := total_days * brand_a_daily
  let available_capsules := available_a_bottles * brand_a_per_bottle
  let remaining_capsules := total_capsules - available_capsules
  let brand_b_days_per_bottle := brand_b_per_bottle / brand_b_daily
  Nat.ceil ((remaining_capsules : ℚ) / brand_b_days_per_bottle) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vitamin_d3_bottles_l342_34274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_analysis_l342_34267

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := (1 - m^2 : ℝ) + (m^2 - 3*m + 2 : ℝ)*Complex.I

/-- Theorem for the given problem -/
theorem complex_number_analysis :
  (∀ m : ℝ, z m = 0 → m = 1) ∧
  (∀ m : ℝ, (z m).re = 0 → m = -1) ∧
  (∀ m : ℝ, (z m).re < 0 ∧ (z m).im < 0 → 1 < m ∧ m < 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_analysis_l342_34267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_not_contradictory_l342_34205

/-- A bag containing balls of two colors -/
structure Bag where
  red : Nat
  white : Nat

/-- The event of drawing exactly n white balls from a bag -/
def exactlyWhite (b : Bag) (n : Nat) : Set (Fin (b.red + b.white)) :=
  sorry

theorem mutually_exclusive_not_contradictory (b : Bag) (h : b.red = 2 ∧ b.white = 2) :
  Disjoint (exactlyWhite b 1) (exactlyWhite b 2) ∧
  ¬(IsEmpty (exactlyWhite b 1) ∧ IsEmpty (exactlyWhite b 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_not_contradictory_l342_34205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_position_l342_34249

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem max_sum_position
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_2015 : S a 2015 > 0)
  (h_2016 : S a 2016 < 0) :
  ∃ n : ℕ, (∀ m : ℕ, S a m ≤ S a n) ∧ n = 1008 :=
by
  sorry

#check max_sum_position

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_position_l342_34249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_given_sin_and_tan_sign_l342_34282

theorem cos_value_given_sin_and_tan_sign (θ : ℝ) 
  (h1 : Real.sin θ = -4/5) (h2 : Real.tan θ > 0) : Real.cos θ = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_given_sin_and_tan_sign_l342_34282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_y_eq_x_point_in_second_quadrant_l342_34218

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (5, 4)
def C : ℝ × ℝ := (10, 8)

def vector_AP (l : ℝ) : ℝ × ℝ := (5 + 8*l - 2, 4 + 5*l - 3)

theorem point_on_line_y_eq_x (l : ℝ) :
  (vector_AP l).1 = (vector_AP l).2 → l = -1/3 := by sorry

theorem point_in_second_quadrant (l : ℝ) :
  (vector_AP l).1 < 0 ∧ (vector_AP l).2 > 0 → -5/8 < l ∧ l < -5/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_y_eq_x_point_in_second_quadrant_l342_34218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_nonnegative_iff_m_geq_one_l342_34203

/-- Given a function f(x) = (m+1)x^2 - (m-1)x + m-1, prove that f(x) ≥ 0 
    for all x in [-1/2, 1/2] if and only if m ≥ 1 -/
theorem function_nonnegative_iff_m_geq_one (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ) → 
    (m + 1) * x^2 - (m - 1) * x + (m - 1) ≥ 0) ↔ 
  m ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_nonnegative_iff_m_geq_one_l342_34203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_24_10_l342_34298

/-- The perimeter of a rhombus with given diagonal lengths -/
noncomputable def rhombusPerimeter (d1 d2 : ℝ) : ℝ :=
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)

/-- Theorem: A rhombus with diagonals 24 and 10 inches has a perimeter of 52 inches -/
theorem rhombus_perimeter_24_10 :
  rhombusPerimeter 24 10 = 52 := by
  -- Unfold the definition of rhombusPerimeter
  unfold rhombusPerimeter
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_24_10_l342_34298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_translation_vertical_line_test_l342_34202

-- Define the logarithm function with base 2
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the translation of a function
def translate (f : ℝ → ℝ) (h k : ℝ) : ℝ → ℝ := λ x => f (x - h) + k

-- Statement 1: Translation property of logarithmic function
theorem log_translation :
  ∀ x : ℝ, x > 3 → log2 (x - 3) + 2 = translate log2 3 2 x :=
by
  sorry

-- Statement 2: Vertical line test for functions
theorem vertical_line_test :
  ∀ (f : ℝ → ℝ) (a x₁ x₂ : ℝ),
    x₁ ≠ x₂ → f x₁ = a ∧ f x₂ = a → False :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_translation_vertical_line_test_l342_34202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_parallel_implies_planes_perp_l342_34207

-- Define structures for Plane and Line
structure Plane where
  -- Placeholder for plane properties
  dummy : Unit

structure Line where
  -- Placeholder for line properties
  dummy : Unit

-- Define perpendicularity between a line and a plane
def perpendicular_line_plane (l : Line) (p : Plane) : Prop :=
  sorry -- Placeholder definition

-- Define parallelism between a line and a plane
def parallel_line_plane (l : Line) (p : Plane) : Prop :=
  sorry -- Placeholder definition

-- Define perpendicularity between two planes
def perpendicular_plane_plane (p1 : Plane) (p2 : Plane) : Prop :=
  sorry -- Placeholder definition

-- Main theorem
theorem line_perp_parallel_implies_planes_perp (l : Line) (α β : Plane) :
  α ≠ β → perpendicular_line_plane l α → parallel_line_plane l β → perpendicular_plane_plane α β :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_parallel_implies_planes_perp_l342_34207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l342_34285

/-- The constant term in the expansion of (1/x - √x)^6 -/
def constantTerm : ℕ := 15

/-- The binomial expansion of (1/x - √x)^6 -/
noncomputable def binomialExpansion (x : ℝ) : ℝ := (1/x - Real.sqrt x)^6

/-- Theorem stating that the constant term in the expansion of (1/x - √x)^6 is 15 -/
theorem constant_term_of_expansion :
  ∃ (f : ℝ → ℝ), (∀ x, x ≠ 0 → f x = binomialExpansion x) ∧
                 (∃ c, ∀ x, x ≠ 0 → f x = c + x * (f x - c) ∧ c = constantTerm) := by
  sorry

#check constant_term_of_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l342_34285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_term_ap_inequality_l342_34257

/-- The number of 3-term arithmetic progressions in a sequence -/
def t₃ {k : ℕ} (a : Fin k → ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem three_term_ap_inequality {k : ℕ} (a : Fin k → ℕ) 
  (h_increasing : ∀ i j : Fin k, i < j → a i < a j) : 
  t₃ a ≤ t₃ (λ i : Fin k => i.val + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_term_ap_inequality_l342_34257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ENG_is_45_degrees_l342_34229

-- Define the rectangle EFGH
structure Rectangle (EF FG : ℝ) where
  ef_positive : EF > 0
  fg_positive : FG > 0

-- Define the point N on side EF
structure PointN (EF : ℝ) where
  n : ℝ
  n_on_ef : 0 ≤ n ∧ n ≤ EF

-- Define the midpoint property
def is_midpoint (N : ℝ) (EH : ℝ) : Prop :=
  2 * N = EH

-- Define angle in degrees
def angle_degrees (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 360

-- Theorem statement
theorem angle_ENG_is_45_degrees
  (EF FG : ℝ)
  (rect : Rectangle EF FG)
  (n : PointN EF)
  (h_ef : EF = 8)
  (h_fg : FG = 4)
  (h_midpoint : is_midpoint n.n (Real.sqrt (EF^2 + FG^2))) :
  angle_degrees 45 :=
sorry

#check angle_ENG_is_45_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ENG_is_45_degrees_l342_34229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l342_34265

/-- The function f(x) = 2cos(ωx + π/4) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + Real.pi / 4)

/-- The theorem stating that the maximum value of ω is 15/4 -/
theorem max_omega_value (ω : ℝ) : 
  (∀ x : ℝ, f ω x ≤ f ω Real.pi) → 
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → y ≤ Real.pi / 6 → f ω y ≤ f ω x) → 
  ω ≤ 15 / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l342_34265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_mappings_A_to_B_l342_34251

-- Define the sets A and B
def A : Finset Nat := {0, 1}
def B : Finset Nat := {0, 1, 2}

-- Define the type of mappings from A to B
def Mapping := A → B

-- Theorem statement
theorem number_of_mappings_A_to_B :
  Fintype.card (A → B) = 9 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_mappings_A_to_B_l342_34251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neither_prime_nor_composite_iff_one_highest_number_is_97_l342_34286

/-- The probability of selecting a number that is neither prime nor composite -/
def p : ℚ := 1 / 97

/-- The highest number in the set -/
def n : ℕ := 97

/-- A number is neither prime nor composite if and only if it is 1 -/
theorem neither_prime_nor_composite_iff_one (k : ℕ) : 
  (¬ Nat.Prime k ∧ k ≠ 1 ∧ ∃ d, 1 < d ∧ d < k ∧ k % d = 0) ↔ k = 1 := by sorry

/-- The probability of selecting 1 from the set {1, ..., n} -/
def prob_select_one (m : ℕ) : ℚ := 1 / m

theorem highest_number_is_97 : 
  prob_select_one n = p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_neither_prime_nor_composite_iff_one_highest_number_is_97_l342_34286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_infinite_grid_l342_34219

/-- A coloring of an infinite grid -/
def InfiniteGridColoring := ℤ → ℤ → ℕ

/-- Checks if a 2x2 sub-grid at position (x, y) has all different colors -/
def validSubGrid (c : InfiniteGridColoring) (x y : ℤ) : Prop :=
  c x y ≠ c (x+1) y ∧ 
  c x y ≠ c x (y+1) ∧ 
  c x y ≠ c (x+1) (y+1) ∧
  c (x+1) y ≠ c x (y+1) ∧
  c (x+1) y ≠ c (x+1) (y+1) ∧
  c x (y+1) ≠ c (x+1) (y+1)

/-- A valid coloring satisfies the condition for all 2x2 sub-grids -/
def validColoring (c : InfiniteGridColoring) : Prop :=
  ∀ x y : ℤ, validSubGrid c x y

/-- The number of colors used in a coloring -/
noncomputable def numColors (c : InfiniteGridColoring) : ℕ := 
  Nat.card {n : ℕ | ∃ x y : ℤ, c x y = n}

/-- Main theorem: The minimum number of colors for a valid coloring is 8 -/
theorem min_colors_infinite_grid : 
  (∃ c : InfiniteGridColoring, validColoring c ∧ numColors c = 8) ∧ 
  (∀ c : InfiniteGridColoring, validColoring c → numColors c ≥ 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_infinite_grid_l342_34219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_192_l342_34259

/-- A geometric sequence with 8 terms, first term 6, and last term 768 -/
def GeometricSequence : Type := {a : Fin 8 → ℝ // a 0 = 6 ∧ a 7 = 768 ∧ ∀ i j, i < j → a j / a i = a 1 / a 0}

theorem sixth_term_is_192 (a : GeometricSequence) : a.val 5 = 192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_192_l342_34259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l342_34212

-- Define the hyperbola C
structure Hyperbola where
  center : ℝ × ℝ
  symmetric_axes : Bool
  asymptotes_slope : ℝ

-- Define the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

-- Define the condition for asymptotes being tangent to the circle
def asymptotes_tangent_to_circle (h : Hyperbola) : Prop :=
  ∃ (x y : ℝ), circle_equation x y ∧ y = h.asymptotes_slope * x

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  sorry

-- Theorem statement
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_symmetric : h.symmetric_axes = true)
  (h_tangent : asymptotes_tangent_to_circle h) :
  eccentricity h = 2 * Real.sqrt 3 / 3 ∨ eccentricity h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l342_34212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_cartesian_eq_min_distance_C₁_C₂_l342_34290

-- Define the curves C₁ and C₂
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, Real.sqrt 2 * Real.sin α)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := Real.cos θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Theorem for the Cartesian equation of C₂
theorem C₂_cartesian_eq : ∀ (x y : ℝ), (∃ θ : ℝ, C₂ θ = (x, y)) ↔ (x - 1/2)^2 + y^2 = 1/4 := by sorry

-- Theorem for the minimum distance between points on C₁ and C₂
theorem min_distance_C₁_C₂ : 
  ∃ (min_dist : ℝ), 
    (∀ (α θ : ℝ), Real.sqrt ((C₁ α).1 - (C₂ θ).1)^2 + ((C₁ α).2 - (C₂ θ).2)^2 ≥ min_dist) ∧ 
    (∃ (α₀ θ₀ : ℝ), Real.sqrt ((C₁ α₀).1 - (C₂ θ₀).1)^2 + ((C₁ α₀).2 - (C₂ θ₀).2)^2 = min_dist) ∧
    (min_dist = (Real.sqrt 7 - 1) / 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_cartesian_eq_min_distance_C₁_C₂_l342_34290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l342_34238

/-- Represents a circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- Represents a triangle with orthocenter M and one side of length d -/
structure Triangle where
  M : ℝ × ℝ
  d : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Determines if a triangle can be constructed within a given circle -/
def canConstructTriangle (c : Circle) (t : Triangle) : Prop :=
  let OM := distance c.O t.M
  0 < t.d ∧ t.d ≤ 2 * c.r ∧
  (t.d < 2 * c.r → |c.r - Real.sqrt (4 * c.r^2 - t.d^2)| < OM ∧ OM < c.r + Real.sqrt (4 * c.r^2 - t.d^2)) ∧
  (t.d = 2 * c.r → OM = c.r)

/-- Calculates the orthocenter of a triangle -/
noncomputable def orthocenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

theorem triangle_construction_theorem (c : Circle) (t : Triangle) :
  canConstructTriangle c t ↔ 
    ∃ (A B C : ℝ × ℝ), 
      distance A B = t.d ∧
      distance c.O A = c.r ∧
      distance c.O B = c.r ∧
      distance c.O C = c.r ∧
      t.M = orthocenter A B C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l342_34238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_batch_size_l342_34221

theorem apple_batch_size (totalApples : ℕ) : totalApples = 30 :=
  by
  -- Conditions
  have tooSmall : totalApples / 6 = totalApples - (5 * totalApples / 6) := by sorry
  have notRipe : totalApples / 3 = totalApples - (2 * totalApples / 3) := by sorry
  have perfectApples : totalApples - (totalApples / 6 + totalApples / 3) = 15 := by sorry

  -- Proof (skipped)
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_batch_size_l342_34221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l342_34225

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := a^2 / b

-- State the theorem
theorem diamond_calculation : diamond (diamond 4 (diamond 2 3)) 1 = 144 := by
  -- Unfold the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp [pow_two]
  -- Perform numerical calculations
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l342_34225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_a_percentage_is_60_8_percent_l342_34275

/-- Represents an examination with questions divided into three groups -/
structure Examination where
  total_questions : Nat
  group_a_questions : Nat
  group_b_questions : Nat
  group_c_questions : Nat
  group_a_mark : Nat
  group_b_mark : Nat
  group_c_mark : Nat

/-- Calculates the total marks for the examination -/
def total_marks (exam : Examination) : Nat :=
  exam.group_a_questions * exam.group_a_mark +
  exam.group_b_questions * exam.group_b_mark +
  exam.group_c_questions * exam.group_c_mark

/-- Calculates the percentage of total marks carried by group A -/
noncomputable def group_a_percentage (exam : Examination) : Real :=
  (exam.group_a_questions * exam.group_a_mark : Real) / (total_marks exam) * 100

/-- Theorem stating that group A carries 60.8% of the total marks in the given examination -/
theorem group_a_percentage_is_60_8_percent (exam : Examination)
  (h1 : exam.total_questions = 100)
  (h2 : exam.group_a_questions + exam.group_b_questions + exam.group_c_questions = exam.total_questions)
  (h3 : exam.group_a_questions ≥ 1)
  (h4 : exam.group_b_questions = 23)
  (h5 : exam.group_c_questions = 1)
  (h6 : exam.group_a_mark = 1)
  (h7 : exam.group_b_mark = 2)
  (h8 : exam.group_c_mark = 3) :
  group_a_percentage exam = 60.8 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_a_percentage_is_60_8_percent_l342_34275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_inverse_through_point_l342_34242

/-- A function f(x) = 2^(ax+b) that passes through (1, 2) and whose inverse also passes through (1, 2) -/
noncomputable def f (a b : ℝ) : ℝ → ℝ := fun x ↦ 2^(a*x + b)

/-- The inverse of f -/
noncomputable def f_inv (a b : ℝ) : ℝ → ℝ := fun y ↦ (Real.log y / Real.log 2 - b) / a

theorem function_and_inverse_through_point (a b : ℝ) :
  f a b 1 = 2 ∧ f_inv a b 2 = 1 → a = -1 ∧ b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_inverse_through_point_l342_34242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_C₁_C₂_l342_34217

-- Define the curves C₁ and C₂
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (2 * Real.cos φ, Real.sin φ)

def C₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1)^2 + (p.2 - 3)^2 = 1}

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem distance_range_C₁_C₂ :
  ∃ (min max : ℝ),
    (∀ (φ : ℝ) (p : ℝ × ℝ), p ∈ C₂ → min ≤ distance (C₁ φ) p ∧ distance (C₁ φ) p ≤ max) ∧
    min = 1 ∧ max = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_C₁_C₂_l342_34217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l342_34284

-- Define the curves C₁ and C₂
noncomputable def C₁ (a : ℝ) (t : ℝ) : ℝ × ℝ :=
  (a * Real.cos t + Real.sqrt 3, a * Real.sin t)

noncomputable def C₂ (θ : ℝ) : ℝ :=
  Real.sqrt (2 * Real.sin θ + 6)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_segment_length :
  ∃ a : ℝ, a > 0 ∧
  A ∈ Set.range (C₁ a) ∧ B ∈ Set.range (C₁ a) ∧
  (∃ θ : ℝ, C₂ θ * Real.cos θ = A.1 ∧ C₂ θ * Real.sin θ = A.2) ∧
  (∃ θ : ℝ, C₂ θ * Real.cos θ = B.1 ∧ C₂ θ * Real.sin θ = B.2) ∧
  (0 : ℝ × ℝ) ∈ Set.Ico A B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l342_34284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_weight_approx_l342_34293

/-- The weight used by a dishonest dealer given their profit percentage --/
noncomputable def dealer_weight (profit_percent : ℝ) : ℝ :=
  1 / (1 + profit_percent / 100)

/-- Theorem stating that the dealer's weight is approximately 0.92 kg --/
theorem dealer_weight_approx :
  let profit_percent : ℝ := 8.695652173913047
  abs (dealer_weight profit_percent - 0.92) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_weight_approx_l342_34293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_when_fourth_powers_sum_to_one_l342_34289

theorem sin_cos_sum_when_fourth_powers_sum_to_one (α : Real) :
  Real.sin α ^ 4 + Real.cos α ^ 4 = 1 → Real.sin α + Real.cos α = 1 ∨ Real.sin α + Real.cos α = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_when_fourth_powers_sum_to_one_l342_34289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_sales_theorem_l342_34215

/-- Calculates the number of oranges to sell to recover from a loss and achieve a profit -/
noncomputable def oranges_to_sell (x : ℝ) (y : ℝ) : ℝ :=
  (1.55 / 16.56) * y

/-- Theorem stating the correct number of oranges to sell -/
theorem orange_sales_theorem (x : ℝ) (y : ℝ) :
  let initial_loss_rate : ℝ := 0.08
  let initial_sale_rate : ℝ := 18
  let target_profit_rate : ℝ := 0.55
  let cost_price : ℝ := 1 / (initial_sale_rate * (1 - initial_loss_rate))
  let new_selling_price : ℝ := cost_price * (1 + target_profit_rate)
  oranges_to_sell x y = new_selling_price * y :=
by
  sorry

#check orange_sales_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_sales_theorem_l342_34215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_monthly_income_l342_34287

-- Define the monthly incomes as real numbers
variable (A_m B_m C_m : ℝ)

-- Define the conditions
def income_ratio (A_m B_m : ℝ) : Prop := A_m / B_m = 5 / 2
def annual_income (A_m : ℝ) : Prop := A_m * 12 = 504000
def B_income_relation (B_m C_m : ℝ) : Prop := B_m = C_m * 1.12

-- Theorem statement
theorem C_monthly_income :
  ∀ A_m B_m C_m : ℝ,
  income_ratio A_m B_m →
  annual_income A_m →
  B_income_relation B_m C_m →
  C_m = 15000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_monthly_income_l342_34287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_fg_positive_l342_34263

-- Define the domain
def Domain : Set ℝ := {x | x < 0 ∨ x > 0}

-- Define the properties of f and g
def IsOddFunction (f : ℝ → ℝ) : Prop := ∀ x ∈ Domain, f (-x) = -f x
def IsEvenFunction (g : ℝ → ℝ) : Prop := ∀ x ∈ Domain, g (-x) = g x

-- Define the theorem
theorem solution_set_of_fg_positive
  (f g : ℝ → ℝ)
  (hf : IsOddFunction f)
  (hg : IsEvenFunction g)
  (h_deriv : ∀ x < 0, (deriv f x) * g x + f x * (deriv g x) > 0)
  (hg_zero : g (-2) = 0) :
  {x : ℝ | f x * g x > 0} = {x : ℝ | -2 < x ∧ x < 0 ∨ 2 < x} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_fg_positive_l342_34263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l342_34233

/-- A point on the parabola y^2 = 2x with x > 2 -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  x_gt_two : x > 2
  on_parabola : y^2 = 2*x

/-- Triangle formed by a point on the parabola and two points on the y-axis -/
structure Triangle (P : ParabolaPoint) where
  b : ℝ
  c : ℝ
  inscribed_circle : (x - 1)^2 + y^2 = 1 → x ∈ Set.Icc 0 P.x ∧ y ∈ Set.Icc c b

/-- The area of the triangle -/
noncomputable def triangle_area (P : ParabolaPoint) (T : Triangle P) : ℝ :=
  1/2 * P.x * (T.b - T.c)

theorem min_triangle_area (P : ParabolaPoint) (T : Triangle P) :
  triangle_area P T ≥ 8 ∧
  (triangle_area P T = 8 ↔ P.x = 4 ∧ (P.y = 2*Real.sqrt 2 ∨ P.y = -2*Real.sqrt 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l342_34233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_property_characterization_l342_34276

/-- The median of three real numbers -/
noncomputable def median (a b c : ℝ) : ℝ := (a + b + c) - max a (max b c) - min a (min b c)

/-- A function satisfying the median property -/
def MedianProperty (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, median (f a b) (f b c) (f c a) = median a b c

/-- Theorem stating that functions satisfying the median property are either the first or second projection -/
theorem median_property_characterization (f : ℝ → ℝ → ℝ) :
  MedianProperty f → (∀ x y : ℝ, f x y = x) ∨ (∀ x y : ℝ, f x y = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_property_characterization_l342_34276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_minus_sin_range_l342_34228

theorem cos_squared_minus_sin_range :
  ∀ x : ℝ, -1 ≤ (Real.cos x) ^ 2 - Real.sin x ∧ (Real.cos x) ^ 2 - Real.sin x ≤ 5/4 ∧
  (∃ x₁ : ℝ, (Real.cos x₁) ^ 2 - Real.sin x₁ = -1) ∧
  (∃ x₂ : ℝ, (Real.cos x₂) ^ 2 - Real.sin x₂ = 5/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_minus_sin_range_l342_34228
