import Mathlib

namespace NUMINAMATH_CALUDE_common_tangents_O₁_O₂_l510_51002

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Number of common tangents between two circles -/
def num_common_tangents (c1 c2 : Circle) : ℕ := sorry

/-- Circle O₁: x² + y² - 2x = 0 -/
def O₁ : Circle :=
  { equation := λ x y => x^2 + y^2 - 2*x = 0 }

/-- Circle O₂: x² + y² - 4x = 0 -/
def O₂ : Circle :=
  { equation := λ x y => x^2 + y^2 - 4*x = 0 }

theorem common_tangents_O₁_O₂ :
  num_common_tangents O₁ O₂ = 1 := by sorry

end NUMINAMATH_CALUDE_common_tangents_O₁_O₂_l510_51002


namespace NUMINAMATH_CALUDE_hyperbola_equation_l510_51098

/-- Given a hyperbola with one focus at (5,0) and asymptotes y = ± 4/3 x, 
    its equation is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (F : ℝ × ℝ) (slope : ℝ) :
  F = (5, 0) →
  slope = 4/3 →
  ∀ (x y : ℝ), (x^2 / 9 - y^2 / 16 = 1) ↔ 
    (∃ (a b c : ℝ), 
      a^2 + b^2 = c^2 ∧
      c = 5 ∧
      b / a = slope ∧
      x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l510_51098


namespace NUMINAMATH_CALUDE_third_group_men_l510_51004

/-- The work rate of a man -/
def man_rate : ℝ := sorry

/-- The work rate of a woman -/
def woman_rate : ℝ := sorry

/-- The number of men in the third group -/
def x : ℕ := sorry

/-- The work rate of 3 men and 8 women equals the work rate of 6 men and 2 women -/
axiom work_rate_equality : 3 * man_rate + 8 * woman_rate = 6 * man_rate + 2 * woman_rate

/-- The work rate of x men and 2 women is 0.7142857142857143 times the work rate of 3 men and 8 women -/
axiom work_rate_fraction : 
  x * man_rate + 2 * woman_rate = 0.7142857142857143 * (3 * man_rate + 8 * woman_rate)

/-- The number of men in the third group is 4 -/
theorem third_group_men : x = 4 := by sorry

end NUMINAMATH_CALUDE_third_group_men_l510_51004


namespace NUMINAMATH_CALUDE_line_perp_plane_condition_l510_51011

-- Define the types for lines and planes
variable (L P : Type) [NormedAddCommGroup L] [NormedSpace ℝ L] [NormedAddCommGroup P] [NormedSpace ℝ P]

-- Define the perpendicular relation
variable (perpendicular : L → L → Prop)
variable (perpendicular_plane : L → P → Prop)

-- Define the subset relation
variable (subset : L → P → Prop)

-- Theorem statement
theorem line_perp_plane_condition (l m : L) (α : P) 
  (h_subset : subset m α) :
  (∀ l m α, perpendicular_plane l α → perpendicular l m) ∧ 
  (∃ l m α, perpendicular l m ∧ ¬perpendicular_plane l α) :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_condition_l510_51011


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l510_51066

theorem arithmetic_expression_equality : (4 + 6 * 3) - (2 * 3) + 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l510_51066


namespace NUMINAMATH_CALUDE_two_digit_square_with_square_digit_product_l510_51005

/-- A function that returns true if a number is a perfect square --/
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that returns the product of digits of a two-digit number --/
def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- The main theorem to be proved --/
theorem two_digit_square_with_square_digit_product : 
  ∃! n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ is_square n ∧ is_square (digit_product n) :=
sorry

end NUMINAMATH_CALUDE_two_digit_square_with_square_digit_product_l510_51005


namespace NUMINAMATH_CALUDE_symmedian_point_is_centroid_of_projections_l510_51050

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Projects a point onto a line segment -/
def projectOntoSegment (P : Point) (A : Point) (B : Point) : Point :=
  sorry

/-- Calculates the centroid of a triangle -/
def centroid (T : Triangle) : Point :=
  sorry

/-- Determines if a point is inside a triangle -/
def isInside (P : Point) (T : Triangle) : Prop :=
  sorry

/-- Calculates the Symmedian Point of a triangle -/
def symmedianPoint (T : Triangle) : Point :=
  sorry

/-- Main theorem: The Symmedian Point is the unique point inside the triangle
    that is the centroid of its projections -/
theorem symmedian_point_is_centroid_of_projections (T : Triangle) :
  let S := symmedianPoint T
  isInside S T ∧
  ∀ P, isInside P T →
    (S = P ↔
      let X := projectOntoSegment P T.B T.C
      let Y := projectOntoSegment P T.C T.A
      let Z := projectOntoSegment P T.A T.B
      P = centroid ⟨X, Y, Z⟩) :=
  sorry

end NUMINAMATH_CALUDE_symmedian_point_is_centroid_of_projections_l510_51050


namespace NUMINAMATH_CALUDE_rhombus_area_from_intersecting_strips_l510_51013

/-- The area of a rhombus formed by two intersecting strips -/
theorem rhombus_area_from_intersecting_strips (α : ℝ) (h_α : 0 < α ∧ α < π) :
  let strip_width : ℝ := 1
  let rhombus_side : ℝ := strip_width / Real.sin α
  let rhombus_area : ℝ := rhombus_side * strip_width
  rhombus_area = 1 / Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_from_intersecting_strips_l510_51013


namespace NUMINAMATH_CALUDE_type_b_first_is_better_l510_51056

/-- Represents the score for a correct answer to a question type -/
def score (questionType : Bool) : ℝ :=
  if questionType then 80 else 20

/-- Represents the probability of correctly answering a question type -/
def probability (questionType : Bool) : ℝ :=
  if questionType then 0.6 else 0.8

/-- Calculates the expected score when choosing a specific question type first -/
def expectedScore (firstQuestionType : Bool) : ℝ :=
  let p1 := probability firstQuestionType
  let p2 := probability (!firstQuestionType)
  let s1 := score firstQuestionType
  let s2 := score (!firstQuestionType)
  p1 * s1 + p1 * p2 * s2

/-- Theorem stating that choosing type B questions first yields a higher expected score -/
theorem type_b_first_is_better :
  expectedScore true > expectedScore false :=
sorry

end NUMINAMATH_CALUDE_type_b_first_is_better_l510_51056


namespace NUMINAMATH_CALUDE_unique_number_exists_l510_51082

-- Define the properties of x
def is_reciprocal_not_less_than_1 (x : ℝ) : Prop := 1 / x ≥ 1
def does_not_contain_6 (x : ℕ) : Prop := ¬ (∃ d : ℕ, d < 10 ∧ d = 6 ∧ ∃ k : ℕ, x = 10 * k + d)
def cube_less_than_221 (x : ℝ) : Prop := x^3 < 221
def is_even (x : ℕ) : Prop := ∃ k : ℕ, x = 2 * k
def is_prime (x : ℕ) : Prop := Nat.Prime x
def is_multiple_of_5 (x : ℕ) : Prop := ∃ k : ℕ, x = 5 * k
def is_irrational (x : ℝ) : Prop := ¬ (∃ p q : ℤ, q ≠ 0 ∧ x = p / q)
def is_less_than_6 (x : ℝ) : Prop := x < 6
def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k^2
def is_greater_than_20 (x : ℝ) : Prop := x > 20
def log_base_10_at_least_2 (x : ℝ) : Prop := Real.log x / Real.log 10 ≥ 2
def is_not_less_than_10 (x : ℝ) : Prop := x ≥ 10

-- Define the theorem
theorem unique_number_exists : ∃! x : ℕ, 
  (is_reciprocal_not_less_than_1 x ∨ does_not_contain_6 x ∨ cube_less_than_221 x) ∧
  (¬is_reciprocal_not_less_than_1 x ∨ ¬does_not_contain_6 x ∨ ¬cube_less_than_221 x) ∧
  (is_even x ∨ is_prime x ∨ is_multiple_of_5 x) ∧
  (¬is_even x ∨ ¬is_prime x ∨ ¬is_multiple_of_5 x) ∧
  (is_irrational x ∨ is_less_than_6 x ∨ is_perfect_square x) ∧
  (¬is_irrational x ∨ ¬is_less_than_6 x ∨ ¬is_perfect_square x) ∧
  (is_greater_than_20 x ∨ log_base_10_at_least_2 x ∨ is_not_less_than_10 x) ∧
  (¬is_greater_than_20 x ∨ ¬log_base_10_at_least_2 x ∨ ¬is_not_less_than_10 x) :=
by sorry


end NUMINAMATH_CALUDE_unique_number_exists_l510_51082


namespace NUMINAMATH_CALUDE_train_speed_calculation_l510_51022

-- Define the given parameters
def train_length : ℝ := 140
def bridge_length : ℝ := 235
def crossing_time : ℝ := 30

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_calculation :
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  let speed_kmhr := speed_ms * conversion_factor
  speed_kmhr = 45 := by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l510_51022


namespace NUMINAMATH_CALUDE_complex_number_intersection_l510_51044

theorem complex_number_intersection (M N : Set ℂ) (i : ℂ) (z : ℂ) : 
  M = {1, 2, z*i} → 
  N = {3, 4} → 
  M ∩ N = {4} → 
  i^2 = -1 →
  z = -4*i := by sorry

end NUMINAMATH_CALUDE_complex_number_intersection_l510_51044


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l510_51009

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 12,
    prove that the fourth term is 6. -/
theorem arithmetic_sequence_fourth_term
  (a : ℝ)  -- Third term of the sequence
  (d : ℝ)  -- Common difference of the sequence
  (h : a + (a + 2*d) = 12)  -- Sum of third and fifth terms is 12
  : a + d = 6 :=  -- Fourth term is 6
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l510_51009


namespace NUMINAMATH_CALUDE_trig_invariant_poly_characterization_l510_51058

/-- A real polynomial that satisfies P(cos x) = P(sin x) for all real x -/
def TrigInvariantPoly (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, P (Real.cos x) = P (Real.sin x)

/-- The main theorem stating the existence of Q for a trig-invariant polynomial P -/
theorem trig_invariant_poly_characterization
  (P : ℝ → ℝ) (hP : TrigInvariantPoly P) :
  ∃ Q : ℝ → ℝ, ∀ X : ℝ, P X = Q (X^4 - X^2) := by
  sorry

end NUMINAMATH_CALUDE_trig_invariant_poly_characterization_l510_51058


namespace NUMINAMATH_CALUDE_definite_integral_sin_plus_one_l510_51030

theorem definite_integral_sin_plus_one (f : ℝ → ℝ) (h : ∀ x, f x = 1 + Real.sin x) :
  ∫ x in (0)..(Real.pi / 2), f x = Real.pi / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_sin_plus_one_l510_51030


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l510_51028

theorem opposite_of_negative_fraction (m : ℚ) : 
  m = -(-(-(1 / 3))) → m = -(1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l510_51028


namespace NUMINAMATH_CALUDE_crow_probability_l510_51078

theorem crow_probability (a b c d : ℕ) : 
  a + b = 50 →  -- Total crows on birch
  c + d = 50 →  -- Total crows on oak
  b ≥ a →       -- Black crows ≥ White crows on birch
  d ≥ c - 1 →   -- Black crows ≥ White crows - 1 on oak
  (b * (d + 1) + a * (c + 1)) / (50 * 51 : ℚ) > (b * c + a * d) / (50 * 51 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_crow_probability_l510_51078


namespace NUMINAMATH_CALUDE_product_remainder_divisible_by_eight_l510_51054

theorem product_remainder_divisible_by_eight :
  (1502 * 1786 * 1822 * 2026) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_divisible_by_eight_l510_51054


namespace NUMINAMATH_CALUDE_table_tennis_matches_l510_51095

theorem table_tennis_matches (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 ∧ (n * (n - 1)) / 2 ≠ 10 := by
  sorry

#check table_tennis_matches

end NUMINAMATH_CALUDE_table_tennis_matches_l510_51095


namespace NUMINAMATH_CALUDE_inequality_proof_l510_51080

theorem inequality_proof (a b : ℝ) (n : ℕ+) 
  (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  (a + b)^(n : ℝ) - a^(n : ℝ) - b^(n : ℝ) ≥ 2^(2*(n : ℝ)) - 2^((n : ℝ) + 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l510_51080


namespace NUMINAMATH_CALUDE_line_segment_param_sum_squares_l510_51096

/-- Given a line segment from (1,2) to (6,9) parameterized by x = pt + q and y = rt + s,
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1,2), prove that p^2 + q^2 + r^2 + s^2 = 79 -/
theorem line_segment_param_sum_squares :
  ∀ (p q r s : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    p * t + q = 1 + 5 * t ∧ 
    r * t + s = 2 + 7 * t) →
  p^2 + q^2 + r^2 + s^2 = 79 := by
sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_squares_l510_51096


namespace NUMINAMATH_CALUDE_g_f_three_equals_one_l510_51060

-- Define the domain of x
inductive Domain : Type
| one : Domain
| two : Domain
| three : Domain
| four : Domain

-- Define function f
def f : Domain → Domain
| Domain.one => Domain.three
| Domain.two => Domain.four
| Domain.three => Domain.two
| Domain.four => Domain.one

-- Define function g
def g : Domain → ℕ
| Domain.one => 2
| Domain.two => 1
| Domain.three => 6
| Domain.four => 8

-- Theorem to prove
theorem g_f_three_equals_one : g (f Domain.three) = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_f_three_equals_one_l510_51060


namespace NUMINAMATH_CALUDE_inequality_proof_l510_51029

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  Real.sqrt ((x^3 + y + 1) * (y^3 + x + 1)) ≥ x^2 + y^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l510_51029


namespace NUMINAMATH_CALUDE_inequality_proof_l510_51026

theorem inequality_proof (n : ℕ) (h : n > 1) : (4^n : ℚ) / (n + 1) < (2*n).factorial / (n.factorial ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l510_51026


namespace NUMINAMATH_CALUDE_rotation180_maps_points_and_is_isometry_l510_51061

-- Define the points
def A : ℝ × ℝ := (-2, 1)
def A' : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (-1, 4)
def B' : ℝ × ℝ := (1, -4)

-- Define the rotation function
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement
theorem rotation180_maps_points_and_is_isometry :
  (rotate180 A = A') ∧ 
  (rotate180 B = B') ∧ 
  (∀ p q : ℝ × ℝ, dist p q = dist (rotate180 p) (rotate180 q)) := by
  sorry


end NUMINAMATH_CALUDE_rotation180_maps_points_and_is_isometry_l510_51061


namespace NUMINAMATH_CALUDE_expand_and_simplify_l510_51048

theorem expand_and_simplify (a : ℝ) : (2*a - 3)^2 + (2*a + 3)*(2*a - 3) = 8*a^2 - 12*a := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l510_51048


namespace NUMINAMATH_CALUDE_y_derivative_l510_51069

open Real

noncomputable def y (x : ℝ) : ℝ :=
  Real.sqrt (9 * x^2 - 12 * x + 5) * arctan (3 * x - 2) - log (3 * x - 2 + Real.sqrt (9 * x^2 - 12 * x + 5))

theorem y_derivative (x : ℝ) :
  deriv y x = ((9 * x - 6) * arctan (3 * x - 2)) / Real.sqrt (9 * x^2 - 12 * x + 5) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l510_51069


namespace NUMINAMATH_CALUDE_max_take_home_pay_l510_51071

/-- The income that maximizes take-home pay given a specific tax rate and fee structure -/
theorem max_take_home_pay :
  let tax_rate (x : ℝ) := 2 * x / 100
  let admin_fee := 500
  let take_home_pay (x : ℝ) := 1000 * x - (tax_rate x * 1000 * x) - admin_fee
  ∃ (x : ℝ), ∀ (y : ℝ), take_home_pay x ≥ take_home_pay y ∧ x = 25 := by
sorry

end NUMINAMATH_CALUDE_max_take_home_pay_l510_51071


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l510_51062

theorem complex_absolute_value_product : Complex.abs (3 - 2*Complex.I) * Complex.abs (3 + 2*Complex.I) = 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l510_51062


namespace NUMINAMATH_CALUDE_elena_savings_theorem_l510_51099

/-- The amount Elena saves when buying binders with a discount and rebate -/
def elenaSavings (numBinders : ℕ) (pricePerBinder : ℚ) (discountRate : ℚ) (rebateThreshold : ℚ) (rebateAmount : ℚ) : ℚ :=
  let originalCost := numBinders * pricePerBinder
  let discountedPrice := originalCost * (1 - discountRate)
  let finalPrice := if originalCost > rebateThreshold then discountedPrice - rebateAmount else discountedPrice
  originalCost - finalPrice

/-- Theorem stating that Elena saves $10.25 under the given conditions -/
theorem elena_savings_theorem :
  elenaSavings 7 3 (25 / 100) 20 5 = (41 / 4) := by
  sorry

end NUMINAMATH_CALUDE_elena_savings_theorem_l510_51099


namespace NUMINAMATH_CALUDE_real_part_of_z_l510_51034

theorem real_part_of_z (z : ℂ) (h : z - Complex.abs z = -8 + 12*I) : 
  Complex.re z = 5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l510_51034


namespace NUMINAMATH_CALUDE_ellipse_properties_l510_51077

/-- Definition of the ellipse M -/
def ellipse_M (x y : ℝ) (a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1 ∧ a > 0

/-- One focus of the ellipse is at (-1, 0) -/
def focus_F : ℝ × ℝ := (-1, 0)

/-- A line l passing through F -/
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

/-- Theorem stating the main results -/
theorem ellipse_properties :
  ∃ (a : ℝ),
    -- 1. The equation of the ellipse
    (∀ x y : ℝ, ellipse_M x y a ↔ x^2 / 4 + y^2 / 3 = 1) ∧
    -- 2. Length of CD when l has a 45° angle
    (∃ C D : ℝ × ℝ,
      C.1 ≠ D.1 ∧
      ellipse_M C.1 C.2 a ∧
      ellipse_M D.1 D.2 a ∧
      C.2 = line_l 1 C.1 ∧
      D.2 = line_l 1 D.1 ∧
      Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 24 / 7) ∧
    -- 3. Maximum value of |S₁ - S₂|
    (∃ S_max : ℝ,
      S_max = Real.sqrt 3 ∧
      ∀ k : ℝ,
        ∃ C D : ℝ × ℝ,
          C.1 ≠ D.1 ∧
          ellipse_M C.1 C.2 a ∧
          ellipse_M D.1 D.2 a ∧
          C.2 = line_l k C.1 ∧
          D.2 = line_l k D.1 ∧
          |C.2 - D.2| ≤ S_max) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l510_51077


namespace NUMINAMATH_CALUDE_congruence_sufficient_not_necessary_for_similarity_l510_51046

-- Define triangles
variable (T1 T2 : Type)

-- Define congruence and similarity relations
variable (congruent : T1 → T2 → Prop)
variable (similar : T1 → T2 → Prop)

-- Theorem: Triangle congruence is sufficient but not necessary for similarity
theorem congruence_sufficient_not_necessary_for_similarity :
  (∀ t1 : T1, ∀ t2 : T2, congruent t1 t2 → similar t1 t2) ∧
  ¬(∀ t1 : T1, ∀ t2 : T2, similar t1 t2 → congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_congruence_sufficient_not_necessary_for_similarity_l510_51046


namespace NUMINAMATH_CALUDE_prime_even_intersection_l510_51018

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def P : Set ℕ := {n : ℕ | isPrime n}
def Q : Set ℕ := {n : ℕ | isEven n}

theorem prime_even_intersection : P ∩ Q = {2} := by sorry

end NUMINAMATH_CALUDE_prime_even_intersection_l510_51018


namespace NUMINAMATH_CALUDE_figurine_arrangement_l510_51072

/-- The number of ways to arrange n uniquely sized figurines in a line -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n uniquely sized figurines in a line,
    with two specific figurines at opposite ends -/
def arrangementsWithEndsFixed (n : ℕ) : ℕ := 2 * arrangements (n - 2)

theorem figurine_arrangement :
  arrangementsWithEndsFixed 9 = 10080 := by
  sorry

end NUMINAMATH_CALUDE_figurine_arrangement_l510_51072


namespace NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l510_51036

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (a + 2, a - 3)

-- Define the property of being in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem range_of_a_in_fourth_quadrant :
  ∀ a : ℝ, in_fourth_quadrant (P a) ↔ -2 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l510_51036


namespace NUMINAMATH_CALUDE_line_passes_through_point_three_common_tangents_implies_a_8_l510_51086

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- Define the line l
def line_l (m x y : ℝ) : Prop := (m + 1) * x + 2 * y - 1 + m = 0

-- Define the second circle
def circle_2 (x y a : ℝ) : Prop := x^2 + y^2 - 2*x + 8*y + a = 0

-- Theorem 1: Line l always passes through the fixed point (-1, 1)
theorem line_passes_through_point :
  ∀ m : ℝ, line_l m (-1) 1 :=
sorry

-- Theorem 2: If circle C and circle_2 have exactly three common tangents, then a = 8
theorem three_common_tangents_implies_a_8 :
  (∃! (t1 t2 t3 : ℝ × ℝ), 
    (∀ x y, circle_C x y → (x - t1.1)^2 + (y - t1.2)^2 = 0 ∨ 
                           (x - t2.1)^2 + (y - t2.2)^2 = 0 ∨ 
                           (x - t3.1)^2 + (y - t3.2)^2 = 0) ∧
    (∀ x y, circle_2 x y a → (x - t1.1)^2 + (y - t1.2)^2 = 0 ∨ 
                              (x - t2.1)^2 + (y - t2.2)^2 = 0 ∨ 
                              (x - t3.1)^2 + (y - t3.2)^2 = 0)) →
  a = 8 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_point_three_common_tangents_implies_a_8_l510_51086


namespace NUMINAMATH_CALUDE_power_equality_implies_q_eight_l510_51016

theorem power_equality_implies_q_eight : 16^4 = 4^q → q = 8 := by sorry

end NUMINAMATH_CALUDE_power_equality_implies_q_eight_l510_51016


namespace NUMINAMATH_CALUDE_divisibility_problem_l510_51042

theorem divisibility_problem (n : ℕ) (h : n = 856) :
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 8) = 24 * k₁ ∧ (n + 8) = 32 * k₂ ∧ (n + 8) = 36 * k₃ ∧ (n + 8) = 3 * k₄) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l510_51042


namespace NUMINAMATH_CALUDE_graph_transformation_l510_51090

-- Define the original function f
variable (f : ℝ → ℝ)

-- Define the symmetry operation with respect to x = 1
def symmetry_x1 (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (2 - x)

-- Define the left shift operation
def shift_left (g : ℝ → ℝ) (units : ℝ) : ℝ → ℝ := λ x => g (x + units)

-- Theorem statement
theorem graph_transformation (f : ℝ → ℝ) :
  shift_left (symmetry_x1 f) 2 = λ x => f (1 - x) := by sorry

end NUMINAMATH_CALUDE_graph_transformation_l510_51090


namespace NUMINAMATH_CALUDE_intersection_sum_mod20_l510_51088

/-- The sum of x-coordinates of intersection points of two modular functions -/
theorem intersection_sum_mod20 : ∃ (x₁ x₂ : ℕ),
  (x₁ < 20 ∧ x₂ < 20) ∧
  (∀ (y : ℕ), (7 * x₁ + 3) % 20 = y % 20 ↔ (13 * x₁ + 17) % 20 = y % 20) ∧
  (∀ (y : ℕ), (7 * x₂ + 3) % 20 = y % 20 ↔ (13 * x₂ + 17) % 20 = y % 20) ∧
  x₁ + x₂ = 12 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_mod20_l510_51088


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l510_51052

/-- The maximum distance from any point on the circle ρ = 8sinθ to the line θ = π/3 is 6 -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 8 * p.2}
  let line := {p : ℝ × ℝ | p.2 = Real.sqrt 3 * p.1}
  ∃ (d : ℝ),
    d = 6 ∧
    ∀ p ∈ circle, ∀ q ∈ line,
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ d :=
by sorry


end NUMINAMATH_CALUDE_max_distance_circle_to_line_l510_51052


namespace NUMINAMATH_CALUDE_dave_initial_apps_l510_51037

/-- The number of apps Dave had on his phone initially -/
def initial_apps : ℕ := sorry

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := 18

/-- The number of apps remaining after deletion -/
def remaining_apps : ℕ := 5

/-- Theorem stating the initial number of apps -/
theorem dave_initial_apps : initial_apps = 23 := by
  sorry

end NUMINAMATH_CALUDE_dave_initial_apps_l510_51037


namespace NUMINAMATH_CALUDE_percentage_problem_l510_51032

theorem percentage_problem : ∃ P : ℝ, P = (0.25 * 16 + 2) ∧ P = 6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l510_51032


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l510_51027

/-- A right triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  ab : ℝ
  bc : ℝ
  ac : ℝ
  right_triangle : ab^2 + bc^2 = ac^2
  ab_eq : ab = 6
  bc_eq : bc = 8
  ac_eq : ac = 10

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_hypotenuse : side_length ≤ t.ac
  on_ab : side_length ≤ t.ab
  on_bc : side_length ≤ t.bc

/-- The theorem stating that the side length of the inscribed square is 120/37 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 120 / 37 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l510_51027


namespace NUMINAMATH_CALUDE_julia_total_kids_l510_51040

/-- The total number of kids Julia played or interacted with during the week -/
def total_kids : ℕ :=
  let monday_tag := 7
  let tuesday_tag := 13
  let thursday_tag := 18
  let wednesday_cards := 20
  let wednesday_hide_seek := 11
  let wednesday_puzzle := 9
  let friday_board_game := 15
  let friday_drawing := 12
  monday_tag + tuesday_tag + thursday_tag + wednesday_cards + wednesday_hide_seek + wednesday_puzzle + friday_board_game + friday_drawing

theorem julia_total_kids : total_kids = 105 := by
  sorry

end NUMINAMATH_CALUDE_julia_total_kids_l510_51040


namespace NUMINAMATH_CALUDE_decimal_shift_difference_l510_51055

theorem decimal_shift_difference (x : ℝ) : 10 * x - x / 10 = 23.76 → x = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_decimal_shift_difference_l510_51055


namespace NUMINAMATH_CALUDE_german_enrollment_l510_51089

theorem german_enrollment (total_students : ℕ) (both_subjects : ℕ) (only_english : ℕ) 
  (h1 : total_students = 32)
  (h2 : both_subjects = 12)
  (h3 : only_english = 10)
  (h4 : total_students = both_subjects + only_english + (total_students - (both_subjects + only_english))) :
  total_students - (both_subjects + only_english) + both_subjects = 22 := by
  sorry

#check german_enrollment

end NUMINAMATH_CALUDE_german_enrollment_l510_51089


namespace NUMINAMATH_CALUDE_product_of_roots_l510_51064

theorem product_of_roots : Real.sqrt 4 ^ (1/3) * Real.sqrt 8 ^ (1/4) = 2 * Real.sqrt 32 ^ (1/12) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l510_51064


namespace NUMINAMATH_CALUDE_six_digit_number_theorem_l510_51035

/-- A six-digit number represented as a list of its digits -/
def SixDigitNumber := List Nat

/-- Checks if a list represents a valid six-digit number -/
def isValidSixDigitNumber (n : SixDigitNumber) : Prop :=
  n.length = 6 ∧ ∀ d ∈ n.toFinset, 0 ≤ d ∧ d ≤ 9

/-- Converts a six-digit number to its integer value -/
def toInt (n : SixDigitNumber) : ℕ :=
  n.foldl (fun acc d => acc * 10 + d) 0

/-- Left-shifts the digits of a six-digit number -/
def leftShift (n : SixDigitNumber) : SixDigitNumber :=
  match n with
  | [a, b, c, d, e, f] => [f, a, b, c, d, e]
  | _ => []

/-- The condition that needs to be satisfied -/
def satisfiesCondition (n : SixDigitNumber) : Prop :=
  isValidSixDigitNumber n ∧
  toInt (leftShift n) = n.head! * toInt n

theorem six_digit_number_theorem :
  ∀ n : SixDigitNumber,
    satisfiesCondition n →
    (n = [1, 1, 1, 1, 1, 1] ∨ n = [1, 0, 2, 5, 6, 4]) :=
sorry

end NUMINAMATH_CALUDE_six_digit_number_theorem_l510_51035


namespace NUMINAMATH_CALUDE_existence_of_subset_with_property_P_l510_51092

-- Define the property P for a subset A and a natural number m
def property_P (A : Set ℕ) (m : ℕ) : Prop :=
  ∀ k : ℕ, ∃ a : ℕ → ℕ, 
    (∀ i, i < k → a i ∈ A) ∧
    (∀ i, i < k - 1 → 1 ≤ a (i + 1) - a i ∧ a (i + 1) - a i ≤ m)

-- Main theorem
theorem existence_of_subset_with_property_P 
  (r : ℕ) (partition : Fin r → Set ℕ) 
  (partition_properties : 
    (∀ i j, i ≠ j → partition i ∩ partition j = ∅) ∧ 
    (⋃ i, partition i) = Set.univ) :
  ∃ (i : Fin r) (m : ℕ), property_P (partition i) m :=
sorry

end NUMINAMATH_CALUDE_existence_of_subset_with_property_P_l510_51092


namespace NUMINAMATH_CALUDE_fermat_like_theorem_l510_51021

theorem fermat_like_theorem (x y z n : ℕ) (h : n ≥ z) : x^n + y^n ≠ z^n := by
  sorry

end NUMINAMATH_CALUDE_fermat_like_theorem_l510_51021


namespace NUMINAMATH_CALUDE_division_remainder_l510_51084

theorem division_remainder : ∃ q : ℕ, 1234567 = 257 * q + 123 ∧ 123 < 257 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l510_51084


namespace NUMINAMATH_CALUDE_work_efficiency_l510_51075

theorem work_efficiency (sakshi_days tanya_days : ℝ) : 
  tanya_days = 16 →
  sakshi_days / 1.25 = tanya_days →
  sakshi_days = 20 := by
sorry

end NUMINAMATH_CALUDE_work_efficiency_l510_51075


namespace NUMINAMATH_CALUDE_largest_number_l510_51000

theorem largest_number : 
  let numbers : List ℝ := [0.935, 0.9401, 0.9349, 0.9041, 0.9400]
  ∀ x ∈ numbers, x ≤ 0.9401 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l510_51000


namespace NUMINAMATH_CALUDE_complex_sum_equals_one_l510_51093

theorem complex_sum_equals_one (w : ℂ) (h : w = Complex.exp (Complex.I * (6 * Real.pi / 11))) :
  w / (1 + w^2) + w^3 / (1 + w^6) + w^4 / (1 + w^8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_one_l510_51093


namespace NUMINAMATH_CALUDE_set1_equivalence_set2_equivalence_set3_equivalence_set4_equivalence_l510_51070

-- Define the sets of points for each condition
def set1 : Set (ℝ × ℝ) := {p | p.1 ≥ -2}
def set2 : Set (ℝ × ℝ) := {p | -2 < p.1 ∧ p.1 < 2}
def set3 : Set (ℝ × ℝ) := {p | |p.1| < 2}
def set4 : Set (ℝ × ℝ) := {p | |p.1| ≥ 2}

-- State the theorems to be proved
theorem set1_equivalence : set1 = {p : ℝ × ℝ | p.1 ≥ -2} := by sorry

theorem set2_equivalence : set2 = {p : ℝ × ℝ | -2 < p.1 ∧ p.1 < 2} := by sorry

theorem set3_equivalence : set3 = {p : ℝ × ℝ | -2 < p.1 ∧ p.1 < 2} := by sorry

theorem set4_equivalence : set4 = {p : ℝ × ℝ | p.1 ≤ -2 ∨ p.1 ≥ 2} := by sorry

end NUMINAMATH_CALUDE_set1_equivalence_set2_equivalence_set3_equivalence_set4_equivalence_l510_51070


namespace NUMINAMATH_CALUDE_larger_bill_value_l510_51059

/-- Proves that the value of the larger denomination bill is $10 given the problem conditions --/
theorem larger_bill_value (total_bills : ℕ) (total_value : ℕ) (five_dollar_bills : ℕ) (larger_bills : ℕ) :
  total_bills = 5 + larger_bills →
  total_bills = 12 →
  five_dollar_bills = 4 →
  larger_bills = 8 →
  total_value = 100 →
  total_value = 5 * five_dollar_bills + larger_bills * 10 :=
by sorry

end NUMINAMATH_CALUDE_larger_bill_value_l510_51059


namespace NUMINAMATH_CALUDE_binomial_26_6_l510_51094

theorem binomial_26_6 (h1 : Nat.choose 24 5 = 42504) (h2 : Nat.choose 24 6 = 134596) :
  Nat.choose 26 6 = 230230 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l510_51094


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l510_51083

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, (∀ x : ℝ, x^2 + 2*x + 1 - a^2 < 0 → -1 + a < x ∧ x < -1 - a) → a < 1) ∧
  (∃ a : ℝ, a < 1 ∧ ¬(∀ x : ℝ, x^2 + 2*x + 1 - a^2 < 0 → -1 + a < x ∧ x < -1 - a)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l510_51083


namespace NUMINAMATH_CALUDE_multiple_right_triangles_exist_l510_51010

/-- A right triangle with a given hypotenuse length and one non-right angle -/
structure RightTriangle where
  hypotenuse : ℝ
  angle : ℝ
  hypotenuse_positive : 0 < hypotenuse
  angle_range : 0 < angle ∧ angle < π / 2

/-- Theorem stating that multiple right triangles can have the same hypotenuse and non-right angle -/
theorem multiple_right_triangles_exist (h : ℝ) (θ : ℝ) 
  (h_pos : 0 < h) (θ_range : 0 < θ ∧ θ < π / 2) :
  ∃ (t1 t2 : RightTriangle), t1 ≠ t2 ∧ 
    t1.hypotenuse = h ∧ t1.angle = θ ∧
    t2.hypotenuse = h ∧ t2.angle = θ :=
sorry

end NUMINAMATH_CALUDE_multiple_right_triangles_exist_l510_51010


namespace NUMINAMATH_CALUDE_edward_final_earnings_l510_51041

/-- Edward's lawn mowing business earnings and expenses --/
def edward_business (spring_earnings summer_earnings supplies_cost : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supplies_cost

/-- Theorem: Edward's final earnings --/
theorem edward_final_earnings :
  edward_business 2 27 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_edward_final_earnings_l510_51041


namespace NUMINAMATH_CALUDE_point_on_transformed_graph_l510_51065

/-- Given a function f : ℝ → ℝ such that f(3) = -2,
    prove that (1, 0) satisfies the equation 3y = 2f(3x) + 4 -/
theorem point_on_transformed_graph (f : ℝ → ℝ) (h : f 3 = -2) :
  let g : ℝ → ℝ := λ x => (2 * f (3 * x) + 4) / 3
  g 1 = 0 := by sorry

end NUMINAMATH_CALUDE_point_on_transformed_graph_l510_51065


namespace NUMINAMATH_CALUDE_problem_statement_l510_51006

theorem problem_statement (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a + b + c = 0) : 
  c * b^2 ≤ a * b^2 ∧ a * b > a * c := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l510_51006


namespace NUMINAMATH_CALUDE_M_necessary_not_sufficient_for_N_l510_51053

def M : Set ℝ := {x | |x + 1| < 4}
def N : Set ℝ := {x | x / (x - 3) < 0}

theorem M_necessary_not_sufficient_for_N :
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧ (∃ b : ℝ, b ∈ M ∧ b ∉ N) := by
  sorry

end NUMINAMATH_CALUDE_M_necessary_not_sufficient_for_N_l510_51053


namespace NUMINAMATH_CALUDE_binary_multiplication_1111_111_l510_51097

theorem binary_multiplication_1111_111 :
  (0b1111 : Nat) * 0b111 = 0b1001111 := by sorry

end NUMINAMATH_CALUDE_binary_multiplication_1111_111_l510_51097


namespace NUMINAMATH_CALUDE_z_power_sum_l510_51023

theorem z_power_sum (z : ℂ) (h : z = (Real.sqrt 2) / (1 - Complex.I)) : 
  z^100 + z^50 + 1 = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_z_power_sum_l510_51023


namespace NUMINAMATH_CALUDE_trig_equation_solution_l510_51039

theorem trig_equation_solution (x : Real) : 
  (Real.sin (π/4 + 5*x) * Real.cos (π/4 + 2*x) = Real.sin (π/4 + x) * Real.sin (π/4 - 6*x)) ↔ 
  (∃ n : Int, x = n * π/4) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l510_51039


namespace NUMINAMATH_CALUDE_ratio_unchanged_l510_51063

/-- Represents the number of animals in the zoo -/
structure ZooPopulation where
  cheetahs : ℕ
  pandas : ℕ

/-- The zoo population 5 years ago -/
def initial_population : ZooPopulation := sorry

/-- The current zoo population -/
def current_population : ZooPopulation :=
  { cheetahs := initial_population.cheetahs + 2,
    pandas := initial_population.pandas + 6 }

/-- The ratio of cheetahs to pandas -/
def cheetah_panda_ratio (pop : ZooPopulation) : ℚ :=
  pop.cheetahs / pop.pandas

theorem ratio_unchanged :
  cheetah_panda_ratio initial_population = cheetah_panda_ratio current_population :=
by sorry

end NUMINAMATH_CALUDE_ratio_unchanged_l510_51063


namespace NUMINAMATH_CALUDE_max_sum_with_constraint_max_sum_achievable_l510_51014

theorem max_sum_with_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 :=
by sorry

theorem max_sum_achievable :
  ∃ (x y z : ℚ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  16 * x * y * z = (x + y)^2 * (x + z)^2 ∧
  x + y + z = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_constraint_max_sum_achievable_l510_51014


namespace NUMINAMATH_CALUDE_find_divisor_l510_51031

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) (divisor : Nat) :
  dividend = divisor * quotient + remainder →
  dividend = 109 →
  quotient = 9 →
  remainder = 1 →
  divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l510_51031


namespace NUMINAMATH_CALUDE_complex_fraction_problem_l510_51057

theorem complex_fraction_problem (x y : ℂ) 
  (h1 : (x^2 + y^2) / (x + y) = 4)
  (h2 : (x^4 + y^4) / (x^3 + y^3) = 2)
  (h3 : x + y ≠ 0)
  (h4 : x^3 + y^3 ≠ 0)
  (h5 : x^5 + y^5 ≠ 0) :
  (x^6 + y^6) / (x^5 + y^5) = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_problem_l510_51057


namespace NUMINAMATH_CALUDE_divisible_by_six_ones_digits_l510_51081

theorem divisible_by_six_ones_digits : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, n < 10 ∧ ∃ m : ℕ, 6 ∣ (10 * m + n)) ∧ S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_six_ones_digits_l510_51081


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l510_51049

/-- Represents the speed of a boat in still water (in kmph) -/
def boat_speed : ℝ := 48

/-- Represents the speed of the stream (in kmph) -/
def stream_speed : ℝ := 16

/-- Represents the time ratio of upstream to downstream travel -/
def time_ratio : ℝ := 2

theorem stream_speed_calculation :
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = time_ratio :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l510_51049


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l510_51045

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℤ)
  (h_arith : ArithmeticSequence a)
  (h_diff : a 4 - a 2 = -2)
  (h_seventh : a 7 = -3) :
  a 9 = -5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l510_51045


namespace NUMINAMATH_CALUDE_min_value_of_3x_plus_4y_min_value_is_five_l510_51038

theorem min_value_of_3x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + 3 * b = 5 * a * b → 3 * x + 4 * y ≤ 3 * a + 4 * b :=
by sorry

theorem min_value_is_five (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_3x_plus_4y_min_value_is_five_l510_51038


namespace NUMINAMATH_CALUDE_negation_of_existence_l510_51043

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x, x^2 - x + 1 = 0) ↔ (∀ x, x^2 - x + 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l510_51043


namespace NUMINAMATH_CALUDE_probability_three_defective_l510_51007

/-- Represents the probability of selecting a defective smartphone from a category. -/
structure CategoryProbability where
  total : ℕ
  defective : ℕ
  probability : ℚ
  valid : probability = defective / total

/-- Represents the data for the smartphone shipment. -/
structure ShipmentData where
  premium : CategoryProbability
  standard : CategoryProbability
  basic : CategoryProbability

/-- The probability of selecting three defective smartphones, one from each category. -/
def probabilityAllDefective (data : ShipmentData) : ℚ :=
  data.premium.probability * data.standard.probability * data.basic.probability

/-- The given shipment data. -/
def givenShipment : ShipmentData := {
  premium := { total := 120, defective := 26, probability := 26 / 120, valid := by norm_num }
  standard := { total := 160, defective := 68, probability := 68 / 160, valid := by norm_num }
  basic := { total := 60, defective := 30, probability := 30 / 60, valid := by norm_num }
}

/-- Theorem stating that the probability of selecting three defective smartphones
    is equal to 221 / 4800 for the given shipment data. -/
theorem probability_three_defective :
  probabilityAllDefective givenShipment = 221 / 4800 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_defective_l510_51007


namespace NUMINAMATH_CALUDE_new_mean_after_combining_l510_51067

theorem new_mean_after_combining (n1 n2 : ℕ) (mean1 mean2 additional : ℚ) :
  let sum1 : ℚ := n1 * mean1
  let sum2 : ℚ := n2 * mean2
  let total_sum : ℚ := sum1 + sum2 + additional
  let total_count : ℕ := n1 + n2 + 1
  (total_sum / total_count : ℚ) = (n1 * mean1 + n2 * mean2 + additional) / (n1 + n2 + 1) :=
by
  sorry

-- Example usage with the given problem values
example : 
  let n1 : ℕ := 7
  let n2 : ℕ := 9
  let mean1 : ℚ := 15
  let mean2 : ℚ := 28
  let additional : ℚ := 100
  (n1 * mean1 + n2 * mean2 + additional) / (n1 + n2 + 1) = 457 / 17 :=
by
  sorry

end NUMINAMATH_CALUDE_new_mean_after_combining_l510_51067


namespace NUMINAMATH_CALUDE_multiply_98_98_l510_51073

theorem multiply_98_98 : 98 * 98 = 9604 := by
  sorry

end NUMINAMATH_CALUDE_multiply_98_98_l510_51073


namespace NUMINAMATH_CALUDE_smaller_angle_is_55_degrees_l510_51008

/-- A parallelogram with specific angle properties -/
structure SpecialParallelogram where
  /-- The measure of the smaller angle in degrees -/
  smaller_angle : ℝ
  /-- The measure of the larger angle in degrees -/
  larger_angle : ℝ
  /-- The length of the parallelogram -/
  length : ℝ
  /-- The width of the parallelogram -/
  width : ℝ
  /-- The larger angle exceeds the smaller angle by 70 degrees -/
  angle_difference : larger_angle = smaller_angle + 70
  /-- Consecutive angles in a parallelogram are supplementary -/
  supplementary : smaller_angle + larger_angle = 180
  /-- The length is three times the width -/
  length_width_ratio : length = 3 * width

/-- Theorem: In a parallelogram where one angle exceeds the other by 70 degrees,
    the measure of the smaller angle is 55 degrees -/
theorem smaller_angle_is_55_degrees (p : SpecialParallelogram) : p.smaller_angle = 55 := by
  sorry

end NUMINAMATH_CALUDE_smaller_angle_is_55_degrees_l510_51008


namespace NUMINAMATH_CALUDE_total_fruits_three_days_l510_51068

/-- Represents the number of fruits eaten by a dog on a given day -/
def fruits_eaten (initial : ℕ) (day : ℕ) : ℕ :=
  initial * 2^(day - 1)

/-- Represents the total fruits eaten by all dogs over a period of days -/
def total_fruits (bonnies_day1 : ℕ) (days : ℕ) : ℕ :=
  let blueberries_day1 := (3 * bonnies_day1) / 4
  let apples_day1 := 3 * blueberries_day1
  let cherries_day1 := 5 * apples_day1
  (Finset.sum (Finset.range days) (λ d => fruits_eaten bonnies_day1 (d + 1))) +
  (Finset.sum (Finset.range days) (λ d => fruits_eaten blueberries_day1 (d + 1))) +
  (Finset.sum (Finset.range days) (λ d => fruits_eaten apples_day1 (d + 1))) +
  (Finset.sum (Finset.range days) (λ d => fruits_eaten cherries_day1 (d + 1)))

theorem total_fruits_three_days :
  total_fruits 60 3 = 6405 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_three_days_l510_51068


namespace NUMINAMATH_CALUDE_restaurant_spirits_profit_l510_51076

/-- Calculates the profit made by a restaurant on a bottle of spirits -/
theorem restaurant_spirits_profit
  (bottle_cost : ℝ)
  (servings_per_bottle : ℕ)
  (price_per_serving : ℝ)
  (h1 : bottle_cost = 30)
  (h2 : servings_per_bottle = 16)
  (h3 : price_per_serving = 8) :
  servings_per_bottle * price_per_serving - bottle_cost = 98 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_spirits_profit_l510_51076


namespace NUMINAMATH_CALUDE_no_simultaneous_doughnut_and_syrup_l510_51087

theorem no_simultaneous_doughnut_and_syrup :
  ¬∃ (x : ℝ), (x^2 - 9*x + 13 < 0) ∧ (x^2 + x - 5 < 0) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_doughnut_and_syrup_l510_51087


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l510_51020

/-- Given a quadratic function f(x) = ax^2 + bx + a satisfying certain conditions,
    prove its expression and range. -/
theorem quadratic_function_properties (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + a
  (∀ x, f (x + 7/4) = f (7/4 - x)) →
  (∃! x, f x = 7 * x + a) →
  (f = λ x ↦ -2 * x^2 + 7 * x - 2) ∧
  (Set.range f = Set.Iic (33/8)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l510_51020


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l510_51015

noncomputable def f (x : ℝ) : ℝ := -1 / (x + 2) + (1 / 2) * Real.log (x^2 + 4) + (1 / 2) * Real.arctan (x / 2)

theorem indefinite_integral_proof (x : ℝ) (h : x ≠ -2) : 
  deriv f x = (x^3 + 6*x^2 + 8*x + 8) / ((x + 2)^2 * (x^2 + 4)) :=
by sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l510_51015


namespace NUMINAMATH_CALUDE_smallest_number_l510_51001

theorem smallest_number (a b c d : ℝ) :
  a = 1 ∧ b = 0 ∧ c = -Real.sqrt 3 ∧ d = -Real.sqrt 2 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l510_51001


namespace NUMINAMATH_CALUDE_sum_of_composition_equals_negative_ten_l510_51085

def p (x : ℝ) : ℝ := abs x - 2

def q (x : ℝ) : ℝ := -abs x

def evaluation_points : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_composition_equals_negative_ten :
  (evaluation_points.map (λ x => q (p x))).sum = -10 := by sorry

end NUMINAMATH_CALUDE_sum_of_composition_equals_negative_ten_l510_51085


namespace NUMINAMATH_CALUDE_infinitely_many_mtrp_numbers_l510_51012

/-- Sum of digits in decimal representation of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Definition of MTRP-number -/
def is_mtrp_number (m n : ℕ) : Prop :=
  n > 0 ∧ n % m = 1 ∧ sum_of_digits (n^2) ≥ sum_of_digits n

theorem infinitely_many_mtrp_numbers (m : ℕ) :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_mtrp_number m n := by sorry

end NUMINAMATH_CALUDE_infinitely_many_mtrp_numbers_l510_51012


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l510_51033

theorem binomial_square_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 8 * x + 16 = (r * x + s)^2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l510_51033


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l510_51079

theorem gcd_of_three_numbers : Nat.gcd 84 (Nat.gcd 294 315) = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l510_51079


namespace NUMINAMATH_CALUDE_equation_has_real_root_l510_51091

theorem equation_has_real_root (a b c : ℝ) : 
  ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l510_51091


namespace NUMINAMATH_CALUDE_consecutive_binomial_ratio_l510_51047

theorem consecutive_binomial_ratio (n k : ℕ) : 
  n > k → 
  (n.choose k : ℚ) / (n.choose (k + 1)) = 1 / 3 →
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2)) = 1 / 2 →
  n + k = 13 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_binomial_ratio_l510_51047


namespace NUMINAMATH_CALUDE_vector_parallel_implies_x_equals_two_l510_51019

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem vector_parallel_implies_x_equals_two :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (2, x)
  parallel (a.1 + b.1, a.2 + b.2) (4 * b.1 - 2 * a.1, 4 * b.2 - 2 * a.2) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_x_equals_two_l510_51019


namespace NUMINAMATH_CALUDE_seattle_seahawks_field_goals_l510_51025

theorem seattle_seahawks_field_goals :
  ∀ (total_score touchdown_score field_goal_score touchdown_count : ℕ),
    total_score = 37 →
    touchdown_score = 7 →
    field_goal_score = 3 →
    touchdown_count = 4 →
    ∃ (field_goal_count : ℕ),
      total_score = touchdown_count * touchdown_score + field_goal_count * field_goal_score ∧
      field_goal_count = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_seattle_seahawks_field_goals_l510_51025


namespace NUMINAMATH_CALUDE_max_acute_angles_convex_polygon_l510_51024

-- Define a convex polygon
structure ConvexPolygon where
  n : ℕ  -- number of sides
  convex : Bool  -- property of being convex

-- Define the theorem
theorem max_acute_angles_convex_polygon (p : ConvexPolygon) : 
  p.convex = true →  -- the polygon is convex
  (∃ (sum_exterior_angles : ℝ), sum_exterior_angles = 360) →  -- sum of exterior angles is 360°
  (∀ (i : ℕ) (interior_angle exterior_angle : ℝ), 
    i < p.n → interior_angle + exterior_angle = 180) →  -- interior and exterior angles are supplementary
  (∃ (max_acute : ℕ), max_acute = 3 ∧ 
    ∀ (acute_count : ℕ), acute_count ≤ max_acute) :=
by sorry

end NUMINAMATH_CALUDE_max_acute_angles_convex_polygon_l510_51024


namespace NUMINAMATH_CALUDE_water_in_final_mixture_l510_51074

/-- Given a mixture where x liters of 10% acid solution is added to 5 liters of pure acid,
    resulting in a final mixture that is 40% water, prove that the amount of water
    in the final mixture is 3.6 liters. -/
theorem water_in_final_mixture :
  ∀ x : ℝ,
  x > 0 →
  0.4 * (5 + x) = 0.9 * x →
  0.9 * x = 3.6 :=
by
  sorry

end NUMINAMATH_CALUDE_water_in_final_mixture_l510_51074


namespace NUMINAMATH_CALUDE_opposite_and_reciprocal_expression_l510_51017

theorem opposite_and_reciprocal_expression (a b c d : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) : 
  (a + b) / 2 - c * d = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_and_reciprocal_expression_l510_51017


namespace NUMINAMATH_CALUDE_divisibility_by_three_l510_51003

theorem divisibility_by_three (a b : ℤ) (h : 3 ∣ (a * b)) :
  ¬(¬(3 ∣ a) ∧ ¬(3 ∣ b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l510_51003


namespace NUMINAMATH_CALUDE_octagon_perimeter_l510_51051

/-- Represents an eight-sided polygon that can be divided into a rectangle and a square --/
structure OctagonWithRectAndSquare where
  rectangle_area : ℕ
  square_area : ℕ
  sum_perimeter : ℕ
  h1 : square_area > rectangle_area
  h2 : square_area * rectangle_area = 98
  h3 : ∃ (a b : ℕ), rectangle_area = a * b ∧ a > 0 ∧ b > 0
  h4 : ∃ (s : ℕ), square_area = s * s ∧ s > 0

/-- The perimeter of the octagon is 32 --/
theorem octagon_perimeter (oct : OctagonWithRectAndSquare) : oct.sum_perimeter = 32 := by
  sorry

end NUMINAMATH_CALUDE_octagon_perimeter_l510_51051
