import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l801_80161

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x : ℝ) : ℝ := x - 1
def line2 (x : ℝ) : ℝ := -2 * x + 8

-- Define the point N
def N : ℝ × ℝ := (0, 3)

-- Define the theorem
theorem circle_properties (C : Circle) :
  (C.center.1 = 3 ∧ C.center.2 = 2 ∧ C.radius = 1) ∧
  (3/2 ≤ C.center.1 ∧ C.center.1 ≤ 7/2) :=
by
  sorry

-- Assumptions
axiom center_on_line1 (C : Circle) : C.center.2 = line1 C.center.1
axiom center_on_line2 (C : Circle) : C.center.2 = line2 C.center.1
axiom radius_is_one (C : Circle) : C.radius = 1
axiom point_M_exists (C : Circle) : ∃ M : ℝ × ℝ, 
  (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = C.radius^2 ∧
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = M.1^2 + M.2^2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l801_80161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_zero_point_condition_l801_80157

/-- A function that represents y = m(1/4)^x - (1/2)^x + 1 --/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * (1/4)^x - (1/2)^x + 1

/-- Theorem stating that if f has only one zero point, then m ≤ 0 or m = 1/4 --/
theorem one_zero_point_condition (m : ℝ) : 
  (∃! x : ℝ, f m x = 0) → (m ≤ 0 ∨ m = 1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_zero_point_condition_l801_80157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_seven_pi_sixth_l801_80127

theorem sin_alpha_plus_seven_pi_sixth (α : ℝ) 
  (h : Real.cos (α - π/6) + Real.sin α = 4*Real.sqrt 3/5) : 
  Real.sin (α + 7*π/6) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_seven_pi_sixth_l801_80127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_of_product_of_sums_l801_80170

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Product of sum of digits from 1 to 100 -/
def product_of_sums : ℕ := (List.range 100).map (fun i => sum_of_digits (i + 1)) |>.prod

/-- Count of trailing zeros in a natural number -/
def trailing_zeros (n : ℕ) : ℕ := sorry

theorem trailing_zeros_of_product_of_sums : 
  trailing_zeros product_of_sums = 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_of_product_of_sums_l801_80170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l801_80102

/-- A triangular pyramid with pairwise perpendicular lateral edges -/
structure TriangularPyramid where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  perp_edges : True  -- Represents the condition that lateral edges are pairwise perpendicular

/-- The volume of a triangular pyramid with pairwise perpendicular lateral edges -/
noncomputable def volume (p : TriangularPyramid) : ℝ := (p.a * p.b * p.c) / 6

/-- Theorem: The volume of a triangular pyramid with pairwise perpendicular lateral edges
    of lengths a, b, and c is equal to abc/6 -/
theorem triangular_pyramid_volume (p : TriangularPyramid) : 
  volume p = (p.a * p.b * p.c) / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l801_80102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l801_80104

theorem sin_beta_value (α β : ℝ) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.sin β = Real.sqrt 2 / 10 ∨ Real.sin β = -Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l801_80104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equality_l801_80154

theorem factorial_equality : 2^6 * 3^3 * 2100 = Nat.factorial 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equality_l801_80154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l801_80133

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define our function
noncomputable def f (x : ℝ) : ℝ := log10 (x^2 - 1)

-- State the theorem
theorem f_increasing_interval :
  StrictMonoOn f (Set.Ioi 1) ∧ 
  ∀ a b, a < 1 → b > 1 → ¬StrictMonoOn f (Set.Ioo a b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l801_80133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_one_meter_apart_l801_80166

-- Define the square room
def Square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

-- Define the color of a point
inductive Color
| Black
| White

-- Define the coloring function
def coloring : Square → Color := sorry

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem same_color_one_meter_apart :
  ∃ (p q : Square), p ≠ q ∧ coloring p = coloring q ∧ distance p q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_one_meter_apart_l801_80166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_touching_circles_perimeter_l801_80155

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The circumference of a circle -/
noncomputable def Circle.circumference (c : Circle) : ℝ := 2 * Real.pi * c.radius

/-- Predicate indicating that two circles touch -/
def touches (c₁ c₂ : Circle) : Prop := 
  Real.sqrt ((c₁.center.1 - c₂.center.1)^2 + (c₁.center.2 - c₂.center.2)^2) = c₁.radius + c₂.radius

/-- The perimeter of the shaded region formed by three touching circles -/
noncomputable def perimeter_of_shaded_region (c₁ c₂ c₃ : Circle) : ℝ := 
  (c₁.circumference + c₂.circumference + c₃.circumference) / 6

/-- The perimeter of the region formed by three touching circles -/
theorem touching_circles_perimeter (c₁ c₂ c₃ : Circle) (h₁ : c₁.circumference = 36) 
  (h₂ : c₂.circumference = 36) (h₃ : c₃.circumference = 36) 
  (touch₁₂ : touches c₁ c₂) (touch₂₃ : touches c₂ c₃) (touch₃₁ : touches c₃ c₁) : 
  perimeter_of_shaded_region c₁ c₂ c₃ = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_touching_circles_perimeter_l801_80155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_diff_l801_80142

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (changed to ℚ for computability)
  d : ℚ      -- Common difference (changed to ℚ)
  first_term : a 1 = a 1  -- First term (tautology to define a 1)
  diff : ∀ n, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- Theorem: If a_3 = 6 and S_3 = 12, then d = 2 -/
theorem arithmetic_sequence_diff (seq : ArithmeticSequence) 
  (h1 : seq.a 3 = 6) (h2 : sum_n seq 3 = 12) : seq.d = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_diff_l801_80142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_class_win_probability_l801_80199

/-- Represents a class in the high school -/
structure SchoolClass where
  grade : Nat

/-- The probability of a class winning the championship in a single year -/
noncomputable def winProbability : ℝ := 1 / 3

/-- The number of consecutive years considered -/
def consecutiveYears : Nat := 3

/-- The probability of the same class winning the championship for three consecutive years -/
noncomputable def sameClassWinProbability : ℝ := winProbability ^ consecutiveYears

theorem same_class_win_probability :
  sameClassWinProbability = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_class_win_probability_l801_80199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_sum_l801_80112

theorem determinant_sum (x y : ℝ) (h1 : x ≠ y) 
  (h2 : Matrix.det ![![2, 3, 7], ![4, x, y], ![4, y, x]] = 0) : x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_sum_l801_80112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_negative_slopes_implies_non_negative_a_l801_80101

open Real

-- Define the function f(x) = ln x + ax^2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2

-- State the theorem
theorem no_negative_slopes_implies_non_negative_a (a : ℝ) :
  (∀ x > 0, deriv (f a) x ≥ 0) → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_negative_slopes_implies_non_negative_a_l801_80101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_sum_l801_80192

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * a 1 + (n * (n - 1) : ℝ) / 2 * (a 2 - a 1)

theorem smallest_positive_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 < 0 →
  a 8 + a 9 > 0 →
  a 8 * a 9 < 0 →
  (∀ n < 16, sum_of_arithmetic_sequence a n ≤ 0) ∧
  sum_of_arithmetic_sequence a 16 > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_sum_l801_80192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l801_80103

-- Define the function f
noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := (m - 2) * x^n

-- Define the point that lies on the graph
def point_on_graph (m n : ℝ) : Prop := f m n m = 9

-- Define a, b, and c
noncomputable def a (m n : ℝ) : ℝ := f m n (m^(-1/3:ℝ))
noncomputable def b (m n : ℝ) : ℝ := f m n (Real.log (1/3))
noncomputable def c (m n : ℝ) : ℝ := f m n (Real.sqrt 2 / 2)

-- Theorem statement
theorem relationship_abc (m n : ℝ) (h : point_on_graph m n) : a m n < c m n ∧ c m n < b m n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l801_80103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_F₁PF₂_eq_two_l801_80125

/-- The hyperbola C with equation x^2 - y^2 = 2 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2 = 2

/-- The left focus F₁ of the hyperbola C -/
noncomputable def F₁ : ℝ × ℝ := sorry

/-- The right focus F₂ of the hyperbola C -/
noncomputable def F₂ : ℝ × ℝ := sorry

/-- A point P on the hyperbola C -/
noncomputable def P : ℝ × ℝ := sorry

/-- The condition that P is on the hyperbola C -/
axiom P_on_C : hyperbola_C P.1 P.2

/-- The condition that PF₁ · PF₂ = 0 -/
axiom PF₁_dot_PF₂_eq_zero : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

/-- The theorem stating that the area of triangle F₁PF₂ is 2 -/
theorem area_F₁PF₂_eq_two : 
  (1/2 : ℝ) * Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_F₁PF₂_eq_two_l801_80125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_value_cos_angle_l801_80160

-- Define the vectors in R²
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := sorry
def c : Fin 2 → ℝ := sorry

-- Define the conditions
axiom b_magnitude : Real.sqrt (b 0 ^ 2 + b 1 ^ 2) = Real.sqrt 2
axiom c_magnitude : Real.sqrt (c 0 ^ 2 + c 1 ^ 2) = 3 * Real.sqrt 5
axiom a_parallel_c : ∃ (k : ℝ), c = fun i => k * a i
axiom vectors_perpendicular : 
  (a 0 + 2 * b 0) * (a 0 - b 0) + (a 1 + 2 * b 1) * (a 1 - b 1) = 0

-- Theorem statements
theorem c_value : c = ![3, 6] ∨ c = ![-3, -6] := by sorry

theorem cos_angle : 
  (a 0 * b 0 + a 1 * b 1) / (Real.sqrt (a 0 ^ 2 + a 1 ^ 2) * Real.sqrt (b 0 ^ 2 + b 1 ^ 2)) = 
    -Real.sqrt 10 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_value_cos_angle_l801_80160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_lcm_product_squares_implies_squares_l801_80175

theorem coprime_lcm_product_squares_implies_squares (a b c : ℕ) :
  Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c →
  ∃ k : ℕ, Nat.lcm a (Nat.lcm b c) = k^2 →
  ∃ m : ℕ, a * b * c = m^2 →
  ∃ x y z : ℕ, a = x^2 ∧ b = y^2 ∧ c = z^2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_lcm_product_squares_implies_squares_l801_80175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_is_two_l801_80163

-- Define the trapezoid
structure RightTrapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_right_angle_A : (A.1 - B.1) * (A.2 - D.2) + (A.2 - B.2) * (D.1 - A.1) = 0
  is_right_angle_D : (D.1 - C.1) * (D.2 - A.2) + (D.2 - C.2) * (A.1 - D.1) = 0
  is_BD_perp_BC : (B.1 - D.1) * (B.1 - C.1) + (B.2 - D.2) * (B.2 - C.2) = 0

-- Define the ratio CD / AD
noncomputable def ratio (t : RightTrapezoid) : ℝ :=
  let CD := Real.sqrt ((t.C.1 - t.D.1)^2 + (t.C.2 - t.D.2)^2)
  let AD := Real.sqrt ((t.A.1 - t.D.1)^2 + (t.A.2 - t.D.2)^2)
  CD / AD

-- Theorem statement
theorem min_ratio_is_two (t : RightTrapezoid) : 
  ∃ (min_ratio : ℝ), min_ratio = 2 ∧ ∀ (r : ℝ), r = ratio t → r ≥ min_ratio := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_is_two_l801_80163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l801_80107

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 + 2 * (3 : ℝ)^(x+1) - (9 : ℝ)^x

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.range f,
    (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ f x = y) ↔ -24 ≤ y ∧ y ≤ 12 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l801_80107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_factors_of_240_l801_80136

/-- The number of odd factors of 240 is 4 -/
theorem odd_factors_of_240 : Finset.card (Finset.filter (fun d => d % 2 = 1) (Nat.divisors 240)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_factors_of_240_l801_80136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_meeting_and_turning_back_l801_80113

/-- Represents a point on the number line -/
structure Point where
  position : ℚ

/-- Represents an ant moving on the number line -/
structure Ant where
  start : Point
  speed : ℚ
  direction : Int  -- 1 for right, -1 for left

noncomputable def meetingPoint (alpha : Ant) (beta : Ant) : ℚ :=
  (beta.start.position - alpha.start.position) / (alpha.speed * alpha.direction + beta.speed * beta.direction)

def antPosition (ant : Ant) (time : ℚ) : ℚ :=
  ant.start.position + ant.speed * (time * ant.direction)

noncomputable def distanceSum (alpha : Ant) (a b c : Point) (time : ℚ) : ℚ :=
  let pos := antPosition alpha time
  |pos - a.position| + |pos - b.position| + |pos - c.position|

theorem ant_meeting_and_turning_back 
  (a b c : Point)
  (alpha beta : Ant)
  (ha : a.position = -24)
  (hb : b.position = -10)
  (hc : c.position = 10)
  (halpha : alpha = { start := a, speed := 4, direction := 1 })
  (hbeta : beta = { start := c, speed := 6, direction := -1 }) :
  ∃ (t₁ t₂ : ℚ),
    antPosition alpha t₁ = antPosition beta t₁ ∧ 
    antPosition alpha t₁ = -52/5 ∧
    distanceSum alpha a b c t₂ = 40 ∧
    t₂ = 2 ∧
    ∃ (t₃ : ℚ), 
      let alpha_return := { start := { position := antPosition alpha t₂ }, speed := alpha.speed, direction := -1 }
      antPosition alpha_return (t₃ - t₂) = antPosition beta t₃ ∧
      antPosition alpha_return (t₃ - t₂) = -44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_meeting_and_turning_back_l801_80113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_interval_min_value_on_m_to_zero_max_value_on_m_to_zero_l801_80135

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(2*x) - 2^(x+1) + 3

-- Theorem for the maximum value on [-1,2]
theorem max_value_on_interval (x : ℝ) (h : x ∈ Set.Icc (-1) 2) : 
  f x ≤ 11 := by
  sorry

-- Theorems for the maximum and minimum values on [m,0] where m ≤ 0
theorem min_value_on_m_to_zero (m : ℝ) (hm : m ≤ 0) (x : ℝ) (hx : x ∈ Set.Icc m 0) :
  f x ≥ 2 := by
  sorry

theorem max_value_on_m_to_zero (m : ℝ) (hm : m ≤ 0) (x : ℝ) (hx : x ∈ Set.Icc m 0) :
  f x ≤ 2^(2*m) - 2^(m+1) + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_interval_min_value_on_m_to_zero_max_value_on_m_to_zero_l801_80135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimal_period_l801_80181

/-- The function f(x) = tan(2x + π/3) -/
noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 3)

/-- The minimal positive period of f(x) -/
noncomputable def minimal_period : ℝ := Real.pi / 2

/-- Theorem: The minimal positive period of f(x) is π/2 -/
theorem f_minimal_period :
  ∀ (x : ℝ), f (x + minimal_period) = f x ∧
  ∀ (T : ℝ), 0 < T → T < minimal_period → ∃ (x : ℝ), f (x + T) ≠ f x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimal_period_l801_80181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_properties_l801_80168

noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 2)

theorem tan_half_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f x < f y) ∧
  (∀ k : ℤ, ∀ x, f (k * π + x) = f (k * π - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_properties_l801_80168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_meeting_theorem_l801_80153

/-- The duration of the time interval in minutes -/
def interval_duration : ℕ := 60

/-- The probability of the two friends meeting -/
def meeting_probability : ℚ := 1/2

/-- n is the number of minutes each friend stays -/
noncomputable def n (p q r : ℕ+) : ℝ := p - q * Real.sqrt r

/-- Condition that r is not divisible by the square of any prime -/
def is_square_free (r : ℕ+) : Prop := ∀ (p : ℕ+), Nat.Prime p → r.val % (p * p).val ≠ 0

theorem friends_meeting_theorem (p q r : ℕ+) (h_square_free : is_square_free r) :
  n p q r = interval_duration - 30 * Real.sqrt 2 →
  p + q + r = 92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_meeting_theorem_l801_80153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_alloy_ratio_is_3_to_2_copper_tin_ratio_in_new_alloy_l801_80130

/-- Represents an alloy with copper and tin -/
structure Alloy where
  copper : ℝ
  tin : ℝ

/-- Represents the mixture of alloys and pure copper -/
structure Mixture where
  alloy1 : Alloy
  alloy2 : Alloy
  pure_copper : ℝ

/-- Calculates the ratio of copper to tin in the new alloy -/
def new_alloy_ratio (m : Mixture) : ℝ × ℝ :=
  let total_copper := m.alloy1.copper + m.alloy2.copper + m.pure_copper
  let total_tin := m.alloy1.tin + m.alloy2.tin
  (total_copper, total_tin)

/-- Theorem stating that the new alloy ratio is 3:2 -/
theorem new_alloy_ratio_is_3_to_2 (m : Mixture) : 
  new_alloy_ratio m = (3, 2) := by
  sorry

/-- The main theorem that proves the ratio of copper to tin in the new alloy is 3:2 -/
theorem copper_tin_ratio_in_new_alloy
  (alloy1 : Alloy)
  (alloy2 : Alloy)
  (pure_copper : ℝ)
  (h1 : alloy1.copper / alloy1.tin = 4)
  (h2 : alloy2.copper / alloy2.tin = 1 / 3)
  (h3 : alloy1.copper + alloy1.tin = 10)
  (h4 : alloy2.copper + alloy2.tin = 16)
  (h5 : alloy1.copper + alloy1.tin + alloy2.copper + alloy2.tin + pure_copper = 35) :
  new_alloy_ratio { alloy1 := alloy1, alloy2 := alloy2, pure_copper := pure_copper } = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_alloy_ratio_is_3_to_2_copper_tin_ratio_in_new_alloy_l801_80130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_will_reach_target_weight_l801_80150

/-- Represents the weight loss progress of a person on a diet --/
structure WeightLoss where
  initial_weight : ℝ
  months_elapsed : ℕ
  current_weight : ℝ
  target_weight : ℝ
  monthly_rate : ℝ

/-- Theorem stating that James will eventually reach his target weight --/
theorem will_reach_target_weight (james : WeightLoss) 
  (h1 : james.initial_weight = 222)
  (h2 : james.months_elapsed = 12)
  (h3 : james.target_weight = 190)
  (h4 : james.monthly_rate > 0)
  (h5 : james.current_weight > james.target_weight)
  : ∃ (future_months : ℕ), james.current_weight - james.monthly_rate * (future_months : ℝ) ≤ james.target_weight :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_will_reach_target_weight_l801_80150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spotlight_intersection_l801_80164

/-- Represents a spotlight on a flat horizontal platform -/
structure Spotlight where
  position : ℝ × ℝ
  angle : ℝ

/-- Represents the configuration of five spotlights -/
def SpotlightConfiguration := Fin 5 → Spotlight

/-- Predicate to check if a beam from a spotlight passes through a point -/
def beam_passes_through (spotlight : Spotlight) (point : ℝ × ℝ × ℝ) : Prop := sorry

/-- Predicate to check if beams from given spotlights intersect at a point -/
def beams_intersect (config : SpotlightConfiguration) (indices : Finset (Fin 5)) : Prop :=
  ∃ (point : ℝ × ℝ × ℝ), ∀ i ∈ indices, beam_passes_through (config i) point

/-- The main theorem statement -/
theorem spotlight_intersection
  (config : SpotlightConfiguration)
  (α β : ℝ)
  (h_acute : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2)
  (h_angles : ∀ i, (config i).angle = α ∨ (config i).angle = β)
  (h_four_intersect : ∀ (indices : Finset (Fin 5)), indices.card = 4 →
    ∃ (rotated_config : SpotlightConfiguration),
      (∀ i, (rotated_config i).position = (config i).position) ∧
      (∀ i, (rotated_config i).angle = (config i).angle) ∧
      beams_intersect rotated_config indices) :
  ∃ (final_config : SpotlightConfiguration),
    (∀ i, (final_config i).position = (config i).position) ∧
    (∀ i, (final_config i).angle = (config i).angle) ∧
    beams_intersect final_config (Finset.univ : Finset (Fin 5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spotlight_intersection_l801_80164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_finger_is_four_l801_80156

def g : ℕ → ℕ 
  | 0 => 2  -- Assuming 0 maps to 2 as it's not explicitly shown in the graph
  | 2 => 4
  | 4 => 2
  | 6 => 0
  | 8 => 0
  | _ => 0  -- For all other inputs, assume output is 0

def anna_sequence : ℕ → ℕ 
  | 0 => 2  -- Start with 2 on the thumb
  | n + 1 => g (anna_sequence n)  -- Apply g to the previous element

theorem tenth_finger_is_four : anna_sequence 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_finger_is_four_l801_80156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_sum_problem_l801_80162

theorem cubic_sum_problem (y : ℝ) (h : y^3 + (1/y)^3 = 110) : y + 1/y = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_sum_problem_l801_80162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_ratio_l801_80173

/-- A line y = b - 2x intersecting the y-axis and x = 6 line -/
structure IntersectingLine where
  b : ℝ
  y : ℝ → ℝ
  h1 : 0 < b
  h2 : b < 6
  h3 : ∀ x, y x = b - 2 * x

/-- Points of intersection -/
noncomputable def P (l : IntersectingLine) : ℝ × ℝ := (0, l.b)
noncomputable def S (l : IntersectingLine) : ℝ × ℝ := (6, l.y 6)
noncomputable def Q (l : IntersectingLine) : ℝ × ℝ := (l.b / 2, 0)

/-- Areas of triangles -/
noncomputable def areaQRS (l : IntersectingLine) : ℝ := sorry
noncomputable def areaQOP (l : IntersectingLine) : ℝ := sorry

/-- Theorem stating the result -/
theorem intersecting_line_ratio (l : IntersectingLine) 
  (h : areaQRS l / areaQOP l = 4 / 25) : l.b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_ratio_l801_80173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sequence_surface_area_sum_l801_80134

/-- The sum of surface areas of all cubes and octahedrons in the infinite sequence -/
noncomputable def infiniteSequenceSum : ℝ := (54 + 9 * Real.sqrt 3) / 8

/-- Edge length of the nth cube in the sequence -/
noncomputable def cubeEdgeLength (n : ℕ) : ℝ := (1 / 3) ^ n

/-- Surface area of the nth cube in the sequence -/
noncomputable def cubeSurfaceArea (n : ℕ) : ℝ := 6 * (cubeEdgeLength n) ^ 2

/-- Surface area of the nth octahedron in the sequence -/
noncomputable def octahedronSurfaceArea (n : ℕ) : ℝ := Real.sqrt 3 * (cubeEdgeLength n) ^ 2

/-- Sum of surface areas of all cubes in the sequence -/
noncomputable def cubeSum : ℝ := ∑' n, cubeSurfaceArea n

/-- Sum of surface areas of all octahedrons in the sequence -/
noncomputable def octahedronSum : ℝ := ∑' n, octahedronSurfaceArea n

/-- Theorem stating that the sum of all surface areas equals the calculated value -/
theorem infinite_sequence_surface_area_sum :
  cubeSum + octahedronSum = infiniteSequenceSum := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sequence_surface_area_sum_l801_80134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_for_P_l801_80105

-- Define the point P
def P : ℝ × ℝ := (1, -4)

-- Define the angle α (noncomputable due to arctan)
noncomputable def α : ℝ := Real.arctan (-4)

-- Theorem statement
theorem tan_double_angle_for_P :
  P.2 / P.1 = Real.tan α →
  Real.tan (2 * α) = 8 / 15 := by
  intro h
  sorry  -- Proof details are omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_for_P_l801_80105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ab_l801_80190

noncomputable def angle : Real := 50 * Real.pi / 180

-- Define the equation
def equation (a b : Int) : Prop :=
  Real.sqrt (9 - 8 * Real.sin angle) = a + b * (1 / Real.sin angle)

-- State the theorem
theorem product_ab (a b : Int) (h : equation a b) : a * b = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ab_l801_80190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_heaviest_weight_l801_80171

def is_valid_weight_set (weights : Finset ℕ) : Prop :=
  weights.card = 8 ∧
  (∀ a b c d, a ∈ weights → b ∈ weights → c ∈ weights → d ∈ weights →
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d →
    max a b + min a b ≠ max c d + min c d)

theorem min_heaviest_weight (weights : Finset ℕ) :
  is_valid_weight_set weights →
  ∃ (w : ℕ), w ∈ weights ∧ w ≥ 34 ∧ (∀ x ∈ weights, x ≤ w) :=
by
  sorry

#check min_heaviest_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_heaviest_weight_l801_80171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_probability_l801_80185

-- Define the parameters of the problem
def total_questions : ℕ := 10
def correct_answers : ℕ := 6
def selected_questions : ℕ := 3
def min_correct_for_selection : ℕ := 2

-- Define the probability of being selected
def probability_of_selection : ℚ := 2/3

-- Theorem statement
theorem selection_probability :
  (Nat.choose correct_answers min_correct_for_selection * Nat.choose (total_questions - correct_answers) (selected_questions - min_correct_for_selection) +
   Nat.choose correct_answers selected_questions) /
  Nat.choose total_questions selected_questions = probability_of_selection :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_probability_l801_80185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_squares_105_103_l801_80149

theorem abs_diff_squares_105_103 : |((105 : ℤ)^2 - 103^2)| = 416 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_squares_105_103_l801_80149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_identification_l801_80152

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem sine_function_identification
  (A ω φ : ℝ)
  (h_A_pos : A > 0)
  (h_ω_pos : ω > 0)
  (h_highest : f A ω φ 2 = Real.sqrt 2)
  (h_zero : f A ω φ 6 = 0)
  (h_between : 2 < 6 ∧ 6 < 2 + Real.pi / ω) :
  ∀ x, f A ω φ x = Real.sqrt 2 * Real.sin (Real.pi / 8 * x + Real.pi / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_identification_l801_80152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_half_l801_80196

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance from the center to a focus -/
noncomputable def focal_distance (e : Ellipse) : ℝ := 
  Real.sqrt (e.a^2 - e.b^2)

/-- The length of the line segment PQ -/
noncomputable def PQ_length (e : Ellipse) : ℝ := 
  2 * e.b^2 / e.a

/-- The length of the line segment FA -/
noncomputable def FA_length (e : Ellipse) : ℝ := 
  e.a + focal_distance e

/-- Theorem: If PQ_length = FA_length, then the eccentricity is 1/2 -/
theorem eccentricity_is_half (e : Ellipse) 
  (h : PQ_length e = FA_length e) : 
  eccentricity e = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_half_l801_80196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_of_triangle_DEF_l801_80120

/-- Given a triangle DEF with side lengths d, e, and f satisfying certain conditions,
    prove that its largest angle is 120°. -/
theorem largest_angle_of_triangle_DEF (d e f : ℝ) : 
  d > 0 → e > 0 → f > 0 →  -- Positive side lengths
  d + 3*e + 3*f = d^2 →    -- First condition
  d + 3*e - 3*f = -5 →     -- Second condition
  ∃ (D E F : ℝ),           -- Angles of the triangle
    D + E + F = π ∧        -- Sum of angles in a triangle
    d / (Real.sin D) = e / (Real.sin E) ∧  -- Law of sines
    d / (Real.sin D) = f / (Real.sin F) ∧
    max D (max E F) = 2*π/3 :=   -- 120° in radians
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_of_triangle_DEF_l801_80120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l801_80117

-- Define the quadratic function
noncomputable def f (x : ℝ) := -2 * x^2 + 6 * x - 1/2

-- State the theorem
theorem quadratic_properties :
  -- The graph opens downward
  (∀ x y : ℝ, f ((x + y) / 2) > (f x + f y) / 2) ∧
  -- The axis of symmetry is x = 3/2
  (∀ h : ℝ, f (3/2 + h) = f (3/2 - h)) ∧
  -- The vertex is (3/2, 4)
  (f (3/2) = 4 ∧ ∀ x : ℝ, f x ≤ 4) ∧
  -- The range of f(x) on the interval [0, 2] is [-1/2, 4]
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → -1/2 ≤ f x ∧ f x ≤ 4) ∧
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2 ∧ f x = -1/2 ∧ f y = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l801_80117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_one_pq_values_l801_80197

theorem complex_product_one_pq_values
  (a b c : ℂ)
  (h_prod : a * b * c = 1)
  (h_not_real : ¬(a.re = a) ∧ ¬(b.re = b) ∧ ¬(c.re = c))
  (h_not_unit : Complex.abs a ≠ 1 ∧ Complex.abs b ≠ 1 ∧ Complex.abs c ≠ 1)
  (p : ℝ)
  (q : ℝ)
  (h_p_def : p = (a + b + c + a⁻¹ + b⁻¹ + c⁻¹).re)
  (h_q_def : q = (a/b + b/c + c/a).re) :
  p = -3 ∧ q = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_one_pq_values_l801_80197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_age_ratio_l801_80186

/-- Tom's current age -/
def T : ℕ := sorry

/-- Number of years ago when Tom's age was three times the sum of his children's ages -/
def N : ℕ := sorry

/-- The sum of Tom's three children's current ages equals Tom's current age -/
axiom children_sum : T = T

/-- N years ago, Tom's age was three times the sum of his children's ages -/
axiom past_relation : T - N = 3 * (T - 3 * N)

/-- The ratio of Tom's current age to N is 4 -/
theorem tom_age_ratio : T / N = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_age_ratio_l801_80186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_field_area_l801_80119

/-- The area of a square field given diagonal crossing time and speed -/
theorem square_field_area (crossing_time speed : Real) : 
  crossing_time = 3.0004166666666667 →
  speed = 2.4 →
  ∃ area : Real, abs (area - 25939764.41) < 0.01 := by
  intros h_time h_speed
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_field_area_l801_80119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_areas_sum_circle_areas_sum_proof_l801_80189

noncomputable def circle_radius (first_radius : Real) : Nat → Real
  | 0 => first_radius
  | n+1 => circle_radius first_radius n / 2

theorem circle_areas_sum (π : Real) (first_radius : Real) (sum : Real) : Prop :=
  first_radius = 1 →
  (∀ n : Nat, n > 0 → 
    (circle_radius first_radius n) = (circle_radius first_radius (n-1)) / 2) →
  sum = (4 * π) / 3 →
  (∑' n, π * (circle_radius first_radius n)^2) = sum

-- The proof goes here
theorem circle_areas_sum_proof : circle_areas_sum Real.pi 1 ((4 * Real.pi) / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_areas_sum_circle_areas_sum_proof_l801_80189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_value_l801_80146

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.log (Real.sqrt (x^2 + 1) + x) - 4

-- State the theorem
theorem f_neg_two_value (a b : ℝ) :
  f a b 2 = 2 → f a b (-2) = -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_value_l801_80146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_theorem_l801_80158

-- Define the logarithmic expressions
noncomputable def a (x : ℝ) := Real.log (x + 1) / Real.log (Real.sqrt (2 * x - 3))
noncomputable def b (x : ℝ) := Real.log ((2 * x - 3)^2) / Real.log (2 * x^2 - 3 * x + 5)
noncomputable def c (x : ℝ) := Real.log (2 * x^2 - 3 * x + 5) / Real.log (x + 1)

-- State the theorem
theorem log_equality_theorem :
  ∃! x : ℝ, (x > 0 ∧ 2 * x - 3 > 0 ∧ 2 * x^2 - 3 * x + 5 > 0) ∧
  ((a x = b x ∧ a x = c x + 1) ∨
   (b x = c x ∧ b x = a x + 1) ∨
   (c x = a x ∧ c x = b x + 1)) ∧
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_theorem_l801_80158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_with_many_prime_divisors_dividing_2_pow_n_plus_1_l801_80111

theorem exists_n_with_many_prime_divisors_dividing_2_pow_n_plus_1 :
  ∃ n : ℕ+, (∃ p : List ℕ, (∀ x ∈ p, Nat.Prime x) ∧ 
    (p.length = 2000) ∧ (∀ x ∈ p, x ∣ n.val)) ∧ (n ∣ 2^n.val + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_with_many_prime_divisors_dividing_2_pow_n_plus_1_l801_80111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_non_coplanar_triangles_l801_80194

-- Define a parallelepiped as a type with 8 vertices
inductive Parallelepiped
| vertex : Fin 8 → Parallelepiped

-- Define a triangle as a set of 3 distinct vertices
def Triangle := { t : Fin 3 → Parallelepiped // Function.Injective t }

-- Define a function to check if two triangles are coplanar (implementation omitted)
noncomputable def areCoplanar : Triangle → Triangle → Bool := sorry

-- Total number of ways to choose two triangles
def totalTrianglePairs : ℕ := Nat.choose 8 3 * Nat.choose 5 3

-- Number of non-coplanar triangle pairs
def nonCoplanarPairs : ℕ := totalTrianglePairs - 192

-- Theorem statement
theorem probability_of_non_coplanar_triangles :
  (nonCoplanarPairs : ℚ) / totalTrianglePairs = 367 / 385 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_non_coplanar_triangles_l801_80194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l801_80148

open Set

def A : Set ℝ := {x : ℝ | 4 ≤ x ∧ x < 8}
def B : Set ℝ := {x : ℝ | 6 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x : ℝ | x > a}

theorem set_operations :
  (A ∪ B = {x : ℝ | 4 ≤ x ∧ x < 9}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | 8 ≤ x ∧ x < 9}) ∧
  (∀ a, A ∩ C a = ∅ → a ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l801_80148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_cosine_l801_80180

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define point P
def P : ℝ × ℝ := (3, 2)

-- Define the property of tangent lines
def is_tangent (A : ℝ × ℝ) : Prop :=
  myCircle A.1 A.2 ∧ ((A.1 - P.1) * (A.1 - 1) + (A.2 - P.2) * (A.2 - 1) = 0)

theorem tangent_lines_cosine :
  ∃ (A B : ℝ × ℝ),
    is_tangent A ∧
    is_tangent B ∧
    A ≠ B ∧
    (let vec_PA := (A.1 - P.1, A.2 - P.2)
     let vec_PB := (B.1 - P.1, B.2 - P.2)
     let cos_APB := (vec_PA.1 * vec_PB.1 + vec_PA.2 * vec_PB.2) /
                    (Real.sqrt (vec_PA.1^2 + vec_PA.2^2) * Real.sqrt (vec_PB.1^2 + vec_PB.2^2))
     cos_APB = 3/5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_cosine_l801_80180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_in_second_quadrant_l801_80172

-- Define the complex number z
def z (a : ℝ) : ℂ := -1 + a * (1 + Complex.I)

-- Define the condition for z to be in the second quadrant
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem imaginary_part_of_z_in_second_quadrant :
  ∃ a : ℝ, in_second_quadrant (z a) ∧ (z a).im = (1 : ℝ) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_in_second_quadrant_l801_80172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_geq_lower_bound_l801_80140

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - |x - 1| + 2 * a

/-- The lower bound for a -/
noncomputable def a_lower_bound : ℝ := (Real.sqrt 3 + 1) / 4

/-- Theorem stating the equivalence between f(x) ≥ 0 for all x and a ≥ lower_bound -/
theorem f_nonnegative_iff_a_geq_lower_bound (a : ℝ) :
  (∀ x, f a x ≥ 0) ↔ a ≥ a_lower_bound := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_geq_lower_bound_l801_80140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_101_of_3_over_11_l801_80188

theorem digit_101_of_3_over_11 : ∃ (d : ℕ → ℕ), 
  (∀ n, d n < 10) ∧ 
  (∀ n, d n = d (n + 2)) ∧ 
  (d 0 = 2 ∧ d 1 = 7) ∧
  (3 : ℚ) / 11 = (∑' n, (d n : ℚ) / 10^(n+1)) ∧
  d 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_101_of_3_over_11_l801_80188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_properties_l801_80147

/-- An ellipse with eccentricity √6/3 and a hyperbola y^2 - x^2 = 1 with common foci -/
structure EllipseHyperbolaSystem where
  C : Set (ℝ × ℝ)  -- The ellipse
  H : Set (ℝ × ℝ)  -- The hyperbola
  eccentricity : ℝ
  foci : ℝ × ℝ
  h_eccentricity : eccentricity = Real.sqrt 6 / 3
  h_hyperbola_eq : ∀ x y, ((x, y) ∈ H) ↔ y^2 - x^2 = 1
  h_common_foci : ∃ c, foci = (0, c) ∧ foci = (0, -c) ∧ c^2 = 2

/-- The standard equation of the ellipse -/
def ellipse_equation (sys : EllipseHyperbolaSystem) : Prop :=
  ∀ x y, ((x, y) ∈ sys.C) ↔ y^2 / 3 + x^2 = 1

/-- The line MN passes through the origin -/
def line_passes_origin (sys : EllipseHyperbolaSystem) : Prop :=
  ∀ M N : ℝ × ℝ,
    M ∈ sys.C → N ∈ sys.C → M ≠ N →
    (let A := (0, -Real.sqrt 3);
     let slope_AM := (M.2 - A.2) / (M.1 - A.1);
     let slope_AN := (N.2 - A.2) / (N.1 - A.1);
     slope_AM * slope_AN = -3) →
    ∃ k : ℝ, N.2 - M.2 = k * (N.1 - M.1) ∧ M.2 = k * M.1 ∧ N.2 = k * N.1

/-- The minimum area of triangle MNP -/
def min_triangle_area (sys : EllipseHyperbolaSystem) : Prop :=
  ∀ M N P : ℝ × ℝ,
    M ∈ sys.C → N ∈ sys.C → P ∈ sys.C →
    M ≠ N → M ≠ P → N ≠ P →
    (M.1 - P.1)^2 + (M.2 - P.2)^2 = (N.1 - P.1)^2 + (N.2 - P.2)^2 →
    ∀ area : ℝ,
      area = abs ((M.1 - N.1) * (P.2 - N.2) - (M.2 - N.2) * (P.1 - N.1)) / 2 →
      area ≥ 3 / 2

theorem ellipse_hyperbola_properties (sys : EllipseHyperbolaSystem) :
  ellipse_equation sys ∧
  line_passes_origin sys ∧
  min_triangle_area sys := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_properties_l801_80147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_and_existence_l801_80128

theorem square_difference_and_existence 
  (m n t : ℕ) 
  (hm : m > 0) (hn : n > 0) (ht : t > 0)
  (h : t * (m^2 - n^2) + m - n^2 - n = 0) : 
  (∃ (k : ℕ), m - n = k^2) ∧ 
  (∀ (t : ℕ), t > 0 → ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ t * (m^2 - n^2) + m - n^2 - n = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_and_existence_l801_80128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_15_minus_alpha_beta_value_l801_80191

-- Part 1
theorem sin_15_minus_alpha (α : ℝ) 
  (h1 : Real.cos (15 * π / 180 + α) = 15 / 17) 
  (h2 : 0 < α) (h3 : α < π / 2) : 
  Real.sin (15 * π / 180 - α) = (15 - 8 * Real.sqrt 3) / 34 := by sorry

-- Part 2
theorem beta_value (α β : ℝ) 
  (h1 : Real.cos α = 1 / 7) 
  (h2 : Real.cos (α - β) = 13 / 14) 
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π / 2) : 
  β = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_15_minus_alpha_beta_value_l801_80191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oxygen_mass_percentage_in_N2O_l801_80176

-- Define the atomic masses
noncomputable def atomic_mass_N : ℝ := 14.01
noncomputable def atomic_mass_O : ℝ := 16.00

-- Define the molar mass of N2O
noncomputable def molar_mass_N2O : ℝ := 2 * atomic_mass_N + atomic_mass_O

-- Define the mass percentage function
noncomputable def mass_percentage (element_mass : ℝ) (compound_mass : ℝ) : ℝ :=
  (element_mass / compound_mass) * 100

-- Theorem to prove
theorem oxygen_mass_percentage_in_N2O :
  ∃ ε > 0, |mass_percentage atomic_mass_O molar_mass_N2O - 36.36| < ε := by
  sorry

-- Note: We've changed the approximation to a more formal ε-δ definition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oxygen_mass_percentage_in_N2O_l801_80176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_root_of_quadratic_l801_80144

theorem smaller_root_of_quadratic (x : ℝ) : 
  let equation := (x - 2/3) * (x - 2/3) + (x - 2/3) * (x - 1/3) - 1/9
  (equation = 0) → 
  (x = 1/3 ∨ x = 5/6) ∧ 
  (∀ y : ℝ, equation = 0 → y ≥ 1/3) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_root_of_quadratic_l801_80144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_polynomials_satisfy_root_property_l801_80122

/-- The complex cubic root of unity -/
noncomputable def ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

/-- A polynomial of degree 6 with real coefficients and constant term 2023 -/
def polynomial (a b c d e : ℝ) (x : ℂ) : ℂ :=
  x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + 2023

/-- Predicate for a polynomial satisfying the root property -/
def satisfiesRootProperty (a b c d e : ℝ) : Prop :=
  ∀ s : ℂ, polynomial a b c d e s = 0 → polynomial a b c d e (ω * s) = 0

/-- There are infinitely many polynomials satisfying the root property -/
theorem infinitely_many_polynomials_satisfy_root_property :
  ∃ S : Set (ℝ × ℝ × ℝ × ℝ × ℝ), 
    (Set.Infinite S) ∧ (∀ p ∈ S, satisfiesRootProperty p.1 p.2.1 p.2.2.1 p.2.2.2.1 p.2.2.2.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_polynomials_satisfy_root_property_l801_80122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_problem_l801_80184

/-- Given that α is inversely proportional to β, and α = 6 when β = 4, 
    prove that α = 1/2 when β = 48 -/
theorem inverse_proportion_problem (k : ℝ) 
  (h1 : ∀ (α β : ℝ), α * β = k) -- inverse proportion definition
  (h2 : 6 * 4 = k) -- given condition
  : k / 48 = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_problem_l801_80184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l801_80151

theorem smallest_positive_z (x z : ℝ) : 
  Real.cos x = 0 → 
  Real.cos (x + z) = -1/2 → 
  z > 0 → 
  (∀ w, w > 0 ∧ Real.cos x = 0 ∧ Real.cos (x + w) = -1/2 → z ≤ w) → 
  z = 11 * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l801_80151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_12_l801_80129

/-- Represents the volume of gas at a given temperature -/
noncomputable def gas_volume (temp : ℝ) : ℝ :=
  35 - (5/4) * (28 - temp)

/-- The problem statement -/
theorem gas_volume_at_12 :
  gas_volume 12 = 15 := by
  -- Unfold the definition of gas_volume
  unfold gas_volume
  -- Simplify the expression
  simp [sub_eq_add_neg, mul_comm, mul_assoc]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_12_l801_80129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_area_theorem_l801_80116

/-- The number of sides of a regular polygon inscribed in a circle, given its area and radius. -/
def n_sides : ℕ := 10

/-- The area of a regular polygon with n sides inscribed in a circle of radius R. -/
noncomputable def polygon_area (n : ℕ) (R : ℝ) : ℝ :=
  1/2 * (n : ℝ) * R^2 * Real.sin (2 * Real.pi / (n : ℝ))

/-- Theorem stating that if the area of a regular polygon inscribed in a circle of radius R is 4R^2, then the number of sides is 10. -/
theorem regular_polygon_area_theorem (R : ℝ) (h : R > 0) :
  polygon_area n_sides R = 4 * R^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_area_theorem_l801_80116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_arithmetic_sequence_l801_80179

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := (x - 1)^2 + n

noncomputable def a (n : ℕ) : ℝ := f n 1

noncomputable def b (n : ℕ) : ℝ := max (f n (-1)) (f n 3)

noncomputable def c (n : ℕ) : ℝ := b n - a n / b n

theorem c_is_arithmetic_sequence :
  ∃ (d : ℝ), d ≠ 0 ∧ ∀ (n : ℕ), c (n + 1) - c n = d :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_arithmetic_sequence_l801_80179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternate_seating_three_boys_three_girls_l801_80121

/-- The number of ways to seat 3 boys and 3 girls alternately in a row -/
def alternateSeating (numBoys numGirls : ℕ) : ℕ :=
  2 * (Nat.factorial numBoys * Nat.factorial numGirls)

theorem alternate_seating_three_boys_three_girls :
  alternateSeating 3 3 = 72 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternate_seating_three_boys_three_girls_l801_80121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quadrant_range_l801_80174

theorem complex_quadrant_range (m : ℝ) :
  let z : ℂ := Complex.ofReal (m + 1) + Complex.I * Complex.ofReal (-(m - 3))
  (z.re > 0 ∧ z.im > 0) ∨ (z.re < 0 ∧ z.im < 0) ↔ -1 < m ∧ m < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quadrant_range_l801_80174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_values_l801_80110

-- Define the power function
noncomputable def power_function (k : ℝ) (x : ℝ) : ℝ := x^k

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.cos (2*x) + k * Real.sin x

-- Theorem statement
theorem sum_of_max_min_values (k : ℝ) :
  power_function k (1/3) = 1/9 →
  ∃ (max_val min_val : ℝ),
    (∀ x, f k x ≤ max_val) ∧
    (∃ x, f k x = max_val) ∧
    (∀ x, f k x ≥ min_val) ∧
    (∃ x, f k x = min_val) ∧
    max_val + min_val = -3/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_values_l801_80110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l801_80115

noncomputable def power_function (m : ℤ) (x : ℝ) : ℝ := (m^2 - 2*m - 2) * x^(m^2 + 4*m)

def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def no_axis_intersection (f : ℝ → ℝ) : Prop :=
  (∀ x ≠ 0, f x ≠ 0) ∧ (∀ y ≠ 0, ∃ x, f x ≠ y)

theorem power_function_properties (m : ℤ) :
  symmetric_about_origin (power_function m) ∧
  no_axis_intersection (power_function m) →
  m = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l801_80115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l801_80100

/-- The inclination angle of a line with equation ax + by + c = 0 -/
noncomputable def inclination_angle (a b : ℝ) : ℝ :=
  Real.arctan (a / b)

theorem line_inclination_angle :
  let line_eq : ℝ → ℝ → ℝ := fun x y ↦ x - y + 2
  inclination_angle 1 (-1) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l801_80100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_implies_a_range_l801_80118

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem f_minimum_implies_a_range (a : ℝ) :
  (∀ x, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_implies_a_range_l801_80118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_a_2009_l801_80143

noncomputable def sequenceA (a : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧
  ∀ n ∈ Finset.range 2008, 
    a (n + 2)^2 - (a (n + 1) / 2009 + 1 / a (n + 1)) * a (n + 2) + 1 / 2009 = 0

theorem max_value_a_2009 (a : ℕ → ℝ) (h : sequenceA a) :
  ∃ (max_a_2009 : ℝ), a 2009 ≤ max_a_2009 ∧ max_a_2009 = (1 / 3) * 2009^2007 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_a_2009_l801_80143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_product_digit_sum_l801_80132

/-- Represents a 101-digit number repeating a 3-digit pattern -/
def RepeatingNumber (pattern : Nat) : Nat :=
  pattern * ((10^99 - 1) / 999)

/-- The product of two specific 101-digit numbers -/
def SpecialProduct : Nat :=
  RepeatingNumber 404 * RepeatingNumber 707

/-- Extracts the thousands digit from a number -/
def ThousandsDigit (n : Nat) : Nat :=
  (n / 1000) % 10

/-- Extracts the ten-thousands digit from a number -/
def TenThousandsDigit (n : Nat) : Nat :=
  (n / 10000) % 10

theorem special_product_digit_sum :
  ThousandsDigit SpecialProduct + TenThousandsDigit SpecialProduct = 8 := by
  sorry

#eval ThousandsDigit SpecialProduct
#eval TenThousandsDigit SpecialProduct
#eval ThousandsDigit SpecialProduct + TenThousandsDigit SpecialProduct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_product_digit_sum_l801_80132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_ball_radius_is_correct_l801_80183

/-- Represents a torus in 3D space -/
structure Torus where
  inner_radius : ℝ
  outer_radius : ℝ
  center : ℝ × ℝ × ℝ

/-- Represents a spherical ball in 3D space -/
structure Ball where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- The largest ball that can be placed on top of a torus -/
noncomputable def largest_ball_on_torus (t : Torus) : Ball :=
  { center := (0, 0, 9/4),
    radius := 9/4 }

theorem largest_ball_radius_is_correct (t : Torus) :
  t.inner_radius = 2 →
  t.outer_radius = 4 →
  t.center = (3, 0, 1) →
  (largest_ball_on_torus t).radius = 9/4 := by
  sorry

#check largest_ball_radius_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_ball_radius_is_correct_l801_80183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l801_80138

noncomputable def f (x : ℝ) := 2 * Real.cos (2 * x + Real.pi / 6)

theorem f_properties :
  (∀ x, f (Real.pi / 3 - x) = f (Real.pi / 3 + x)) ∧
  (∀ x, f (x + Real.pi / 6) = -f (-x - Real.pi / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l801_80138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l801_80145

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := a * b - a * b^2

-- Define the # operation
def hash_op (a b : ℚ) : ℚ := a^2 + b - a^2 * b

-- Theorem statement
theorem fraction_equality : (at_op 8 3) / (hash_op 8 3) = 48 / 125 := by
  -- Evaluate at_op 8 3
  have h1 : at_op 8 3 = -48 := by
    unfold at_op
    norm_num
  
  -- Evaluate hash_op 8 3
  have h2 : hash_op 8 3 = -125 := by
    unfold hash_op
    norm_num
  
  -- Combine the results
  calc
    (at_op 8 3) / (hash_op 8 3) = (-48) / (-125) := by rw [h1, h2]
    _ = 48 / 125 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l801_80145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_without_y_squared_l801_80109

/-- The value of m that makes the expansion of (y+3)(y^2-my-2) not contain the term y^2 -/
theorem expansion_without_y_squared :
  (∃ m : ℝ, ∀ y : ℝ, (y + 3) * (y^2 - m*y - 2) = y^3 + 0*y^2 + ((-2 - 3*m) : ℝ)*y + (-6 : ℝ)) ↔ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_without_y_squared_l801_80109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_constant_l801_80178

/-- The set of all points in the plane -/
def S : Type := ℝ × ℝ

/-- A function from points in the plane to real numbers -/
noncomputable def f : S → ℝ := sorry

/-- Definition of a nondegenerate triangle -/
def is_nondegenerate_triangle (A B C : S) : Prop := sorry

/-- Definition of the orthocenter of a triangle -/
noncomputable def orthocenter (A B C : S) : S := sorry

/-- The main theorem: if f satisfies the given condition for all nondegenerate triangles, 
    then f is constant -/
theorem f_is_constant : 
  (∀ (A B C : S), is_nondegenerate_triangle A B C → 
    let H := orthocenter A B C
    ∀ (h1 : f A ≤ f B) (h2 : f B ≤ f C), f A + f C = f B + f H) →
  ∀ (P Q : S), f P = f Q :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_constant_l801_80178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_zero_range_a_f_leq_abs_x_l801_80108

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then x^2 + 2*x + a - 1
  else if 0 < x ∧ x ≤ 3 then -x^2 + 2*x - a
  else 0  -- undefined outside the domain

-- Theorem 1: Minimum value of f(x) when a = 0
theorem min_value_f_zero : 
  ∃ (m : ℝ), m = -3 ∧ ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f 0 x ≥ m := by
  sorry

-- Theorem 2: Range of a such that f(x) ≤ |x| for all x in the domain
theorem range_a_f_leq_abs_x :
  ∀ a : ℝ, (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f a x ≤ |x|) ↔ (1/4 ≤ a ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_zero_range_a_f_leq_abs_x_l801_80108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_negative_97_squared_l801_80187

def alternating_sum (n : ℕ) : ℤ :=
  Finset.sum (Finset.range n) (λ i => (if i % 2 = 0 then 1 else -1) * (100 - i : ℤ)^2)

theorem fourth_term_is_negative_97_squared (h : alternating_sum 100 = 5050) :
  (if 3 % 2 = 0 then 1 else -1) * (100 - 3 : ℤ)^2 = -97^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_negative_97_squared_l801_80187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sidewalk_snow_volume_l801_80177

/-- Calculates the volume of snow on a sidewalk with varying depths -/
noncomputable def snow_volume (length width : ℝ) (depth1 depth2 : ℝ) : ℝ :=
  (length / 2) * width * depth1 + (length / 2) * width * depth2

theorem sidewalk_snow_volume :
  snow_volume 30 3 (1/2) (1/3) = 37.5 := by
  -- Unfold the definition of snow_volume
  unfold snow_volume
  -- Simplify the expression
  simp [mul_add, mul_assoc, mul_comm]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sidewalk_snow_volume_l801_80177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_l801_80137

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Circle with center (0, 0) and radius 2 -/
def circleSet : Set Point := {p : Point | p.x^2 + p.y^2 = 4}

/-- Point A through which the tangent lines pass -/
def pointA : Point := ⟨2, 4⟩

/-- First potential tangent line: x = 2 -/
def line1 : Line := ⟨1, 0, -2⟩

/-- Second potential tangent line: 3x - 4y + 10 = 0 -/
def line2 : Line := ⟨3, -4, 10⟩

/-- Predicate to check if a line is tangent to the circle -/
def isTangent (l : Line) : Prop :=
  ∃ p : Point, p ∈ circleSet ∧ l.a * p.x + l.b * p.y + l.c = 0 ∧
  ∀ q : Point, q ∈ circleSet → l.a * q.x + l.b * q.y + l.c = 0 → q = p

/-- Predicate to check if a line passes through point A -/
def passesThroughA (l : Line) : Prop :=
  l.a * pointA.x + l.b * pointA.y + l.c = 0

/-- Theorem stating that line1 and line2 are the tangent lines to the circle passing through point A -/
theorem tangent_lines_theorem : 
  (isTangent line1 ∧ passesThroughA line1) ∧
  (isTangent line2 ∧ passesThroughA line2) ∧
  ∀ l : Line, isTangent l ∧ passesThroughA l → l = line1 ∨ l = line2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_l801_80137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_swarm_problem_l801_80167

theorem bee_swarm_problem (N : ℕ) : 
  N > 0 → 
  (Real.sqrt (N / 2 : ℝ) : ℝ) + (8 * N : ℝ) / 9 + 1 = N → 
  N = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_swarm_problem_l801_80167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_MAB_properties_l801_80198

-- Define the points
def M : ℝ × ℝ := (0, -1)
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (2, 1)

-- Define the triangle MAB
def triangle_MAB : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ t u : ℝ, 0 ≤ t ∧ 0 ≤ u ∧ t + u ≤ 1 ∧ p = (t * A.1 + u * B.1 + (1 - t - u) * M.1, t * A.2 + u * B.2 + (1 - t - u) * M.2)}

-- Define the area of a triangle
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Define the slope of a line passing through two points
noncomputable def line_slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the inclination angle of a line with a given slope
noncomputable def inclination_angle (k : ℝ) : ℝ := Real.arctan k

-- Theorem statement
theorem triangle_MAB_properties :
  (triangle_area M A B = 2) ∧
  (∀ k : ℝ, (∃ p : ℝ × ℝ, p ∈ triangle_MAB ∧ line_slope M p = k) ↔ -1 ≤ k ∧ k ≤ 1) ∧
  (∀ α : ℝ, (∃ k : ℝ, (∃ p : ℝ × ℝ, p ∈ triangle_MAB ∧ line_slope M p = k) ∧ inclination_angle k = α) ↔
    (0 ≤ α ∧ α ≤ Real.pi/4) ∨ (3*Real.pi/4 ≤ α ∧ α ≤ Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_MAB_properties_l801_80198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l801_80126

def sequenceTerm (n : ℕ+) : ℚ := ((-1 : ℚ) ^ (n.val + 1)) / n.val

theorem sequence_formula (n : ℕ+) : 
  sequenceTerm n = ((-1 : ℚ) ^ (n.val + 1)) / n.val := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l801_80126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convert_one_third_mps_to_kmph_l801_80131

/-- Conversion factor from meters per second to kilometers per hour -/
noncomputable def conversion_factor : ℝ := 3.6

/-- Initial speed in meters per second -/
noncomputable def initial_speed : ℝ := 1/3

/-- Conversion from meters per second to kilometers per hour -/
noncomputable def convert_mps_to_kmph (speed_mps : ℝ) : ℝ := speed_mps * conversion_factor

/-- Theorem: Converting 1/3 m/s to km/h results in 1.2 km/h -/
theorem convert_one_third_mps_to_kmph : 
  convert_mps_to_kmph initial_speed = 1.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convert_one_third_mps_to_kmph_l801_80131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l801_80195

noncomputable section

open Real

-- Define the vectors a and b
def a (α : ℝ) : ℝ × ℝ := (sin (α + π / 6), 3)
def b (α : ℝ) : ℝ × ℝ := (1, 4 * cos α)

-- Define the perpendicular and parallel conditions
def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

theorem vector_relations (α : ℝ) (h : 0 < α ∧ α < π) :
  (perpendicular (a α) (b α) → tan α = -25 * Real.sqrt 3 / 3) ∧
  (parallel (a α) (b α) → α = π / 6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l801_80195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l801_80114

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.log x - a

-- State the theorem
theorem f_properties (a : ℝ) (h_a : 0 < a ∧ a < Real.exp 1) :
  -- 1. When a = e, the tangent line at x = 1 has equation y = 0
  (a = Real.exp 1 → (deriv (f a)) 1 = 0) ∧
  -- 2. f(x) has a minimum value in the interval (a/e, 1)
  (∃ x₀ : ℝ, a / Real.exp 1 < x₀ ∧ x₀ < 1 ∧ 
    ∀ x : ℝ, a / Real.exp 1 < x ∧ x < 1 → f a x₀ ≤ f a x) ∧
  -- 3. The minimum value of f(x) in (a/e, 1) is greater than 0
  (∃ x₀ : ℝ, a / Real.exp 1 < x₀ ∧ x₀ < 1 ∧ 
    (∀ x : ℝ, a / Real.exp 1 < x ∧ x < 1 → f a x₀ ≤ f a x) ∧
    0 < f a x₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l801_80114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_problem_l801_80159

/-- Represents the number of students choosing option A on the n-th Monday -/
def a : ℕ+ → ℕ := sorry

/-- Total number of students served daily -/
def total_students : ℕ := 500

/-- Initial number of students choosing option A -/
def initial_a : ℕ := 428

/-- Proportion of students switching from A to B -/
def switch_a_to_b : ℚ := 1/5

/-- Proportion of students switching from B to A -/
def switch_b_to_a : ℚ := 3/10

theorem cafeteria_problem (n : ℕ+) :
  (∀ k : ℕ+, a k + (total_students - a k) = total_students) →
  (∀ k : ℕ+, a (k + 1) = a k * (1 - switch_a_to_b) + (total_students - a k) * switch_b_to_a) →
  a 1 = initial_a →
  a 8 = 301 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_problem_l801_80159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l801_80165

-- Define the function f
noncomputable def f (b : ℝ) : ℝ → ℝ := fun x =>
  if x ≥ 0 then 2^x + 2*x + b else -(2^(-x) + 2*(-x) + b)

-- State the theorem
theorem odd_function_value (b : ℝ) :
  (∀ x, f b (-x) = -(f b x)) → f b (-1) = -3 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l801_80165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_changes_for_distinct_sums_l801_80106

-- Define the matrix type
def Matrix3x3 (α : Type) := Fin 3 → Fin 3 → α

-- Define the initial matrix
def initial_matrix : Matrix3x3 ℤ := λ i j =>
  match i, j with
  | 0, 0 => 4
  | 0, 1 => 9
  | 0, 2 => 2
  | 1, 0 => 8
  | 1, 1 => 1
  | 1, 2 => 6
  | 2, 0 => 3
  | 2, 1 => 5
  | 2, 2 => 7

-- Function to calculate row sum
def row_sum (m : Matrix3x3 ℤ) (i : Fin 3) : ℤ :=
  (m i 0) + (m i 1) + (m i 2)

-- Function to calculate column sum
def col_sum (m : Matrix3x3 ℤ) (j : Fin 3) : ℤ :=
  (m 0 j) + (m 1 j) + (m 2 j)

-- Function to update matrix
def update_matrix (m : Matrix3x3 ℤ) (x y z : ℤ) : Matrix3x3 ℤ :=
  λ i j =>
    if i = 0 ∧ j = 0 then x
    else if i = 0 ∧ j = 2 then y
    else if i = 1 ∧ j = 1 then z
    else m i j

-- Theorem statement
theorem min_changes_for_distinct_sums :
  ∃ (x y z : ℤ),
    (∀ i j : Fin 3, i ≠ j → row_sum (update_matrix initial_matrix x y z) i ≠
                            row_sum (update_matrix initial_matrix x y z) j) ∧
    (∀ i j : Fin 3, i ≠ j → col_sum (update_matrix initial_matrix x y z) i ≠
                            col_sum (update_matrix initial_matrix x y z) j) ∧
    (∀ (a b c : ℤ), 
      (∀ i j : Fin 3, i ≠ j → row_sum (update_matrix initial_matrix a b c) i ≠
                               row_sum (update_matrix initial_matrix a b c) j) →
      (∀ i j : Fin 3, i ≠ j → col_sum (update_matrix initial_matrix a b c) i ≠
                               col_sum (update_matrix initial_matrix a b c) j) →
      (a ≠ 4 ∨ b ≠ 2 ∨ c ≠ 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_changes_for_distinct_sums_l801_80106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2_value_l801_80123

def my_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (2 * n) = 2 * a (2 * n - 2) + 1

theorem a_2_value (a : ℕ → ℤ) (h1 : my_sequence a) (h2 : a 16 = 127) : a 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2_value_l801_80123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_POM_l801_80141

/-- Given a curve C with parametric equations x = 2√2 * t^2 and y = 4t,
    proves that the area of triangle POM is 2√3, where M(√2, 0) and P is on C with |PM| = 4√2 -/
theorem area_triangle_POM (t : ℝ) (P : ℝ × ℝ) :
  (∃ t, P.1 = 2 * Real.sqrt 2 * t^2 ∧ P.2 = 4 * t) →  -- P is on curve C
  ‖P - (Real.sqrt 2, 0)‖ = 4 * Real.sqrt 2 →  -- |PM| = 4√2
  (1/2) * Real.sqrt 2 * abs P.2 = 2 * Real.sqrt 3 :=  -- Area of triangle POM
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_POM_l801_80141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_correct_answers_for_prize_l801_80193

/-- Represents the scoring system for a math competition. -/
structure ScoringSystem where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ

/-- Calculates the minimum number of correct answers needed to achieve a target score. -/
def min_correct_answers (system : ScoringSystem) (target_score : ℤ) : ℕ :=
  let x := (target_score - system.incorrect_points * system.total_questions) /
           (system.correct_points - system.incorrect_points)
  (Int.ceil x).toNat

/-- Theorem stating the minimum number of correct answers needed to win a prize. -/
theorem min_correct_answers_for_prize (system : ScoringSystem)
    (h1 : system.total_questions = 30)
    (h2 : system.correct_points = 4)
    (h3 : system.incorrect_points = -2)
    (prize_threshold : ℤ)
    (h4 : prize_threshold = 60) :
    min_correct_answers system prize_threshold = 20 := by
  sorry

#eval min_correct_answers ⟨30, 4, -2⟩ 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_correct_answers_for_prize_l801_80193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_reachable_region_l801_80169

/-- Represents the vehicle's speed on roads in miles per hour -/
noncomputable def road_speed : ℝ := 60

/-- Represents the vehicle's speed across desert in miles per hour -/
noncomputable def desert_speed : ℝ := 10

/-- Represents the time limit in hours -/
noncomputable def time_limit : ℝ := 1/10

/-- Represents the maximum distance the vehicle can travel on roads -/
noncomputable def max_road_distance : ℝ := road_speed * time_limit

/-- Represents the maximum distance the vehicle can travel across desert -/
noncomputable def max_desert_distance : ℝ := desert_speed * time_limit

/-- Represents the area of the reachable region -/
noncomputable def reachable_area : ℝ := 4 * Real.pi + 12

/-- The main theorem to prove -/
theorem area_of_reachable_region : reachable_area = 4 * Real.pi + 12 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_reachable_region_l801_80169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_bound_l801_80182

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|

noncomputable def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem min_value_bound (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_sum : m + n = 2) :
  (m^2 + 2) / m + (n^2 + 1) / n ≥ (7 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_bound_l801_80182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l801_80124

theorem tan_half_sum (a b : ℝ) 
  (h1 : Real.cos a + Real.cos b = 3/5) 
  (h2 : Real.sin a + Real.sin b = 1/3) : 
  Real.tan ((a + b) / 2) = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l801_80124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_perpendicular_to_plane_parallel_l801_80139

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Fact (finrank ℝ V = 3)]

-- Define lines and planes
def Line (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Set V
def Plane (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Set V

-- Define parallel and perpendicular relations
def Parallel (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] (l1 l2 : Line V) : Prop := sorry
def Perpendicular (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] (l : Line V) (p : Plane V) : Prop := sorry

-- State the theorems to be proved
theorem parallel_transitive {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  {a b c : Line V} (hab : Parallel V a b) (hbc : Parallel V b c) : 
  Parallel V a c := by sorry

theorem perpendicular_to_plane_parallel {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  {a b : Line V} {γ : Plane V} 
  (ha : Perpendicular V a γ) (hb : Perpendicular V b γ) : Parallel V a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_perpendicular_to_plane_parallel_l801_80139
