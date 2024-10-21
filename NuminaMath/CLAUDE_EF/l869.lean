import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l869_86918

noncomputable def f (ω φ x : Real) : Real := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem function_properties
  (ω φ α : Real)
  (h_ω_pos : ω > 0)
  (h_φ_bound : -Real.pi / 2 ≤ φ ∧ φ ≤ Real.pi / 2)
  (h_symmetric : ∀ x, f ω φ (2 * Real.pi / 3 - x) = f ω φ (2 * Real.pi / 3 + x))
  (h_period : ∀ x, f ω φ (x + Real.pi) = f ω φ x)
  (h_f_alpha : f ω φ (α / 2) = Real.sqrt 3 / 4)
  (h_α_bound : Real.pi / 6 < α ∧ α < 2 * Real.pi / 3) :
  ω = 2 ∧ φ = -Real.pi / 6 ∧ Real.sin α = (Real.sqrt 3 + Real.sqrt 15) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l869_86918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bulb_box_probabilities_l869_86969

/-- Represents a box of light bulbs -/
structure BulbBox where
  total : ℕ
  defective : ℕ
  non_defective : ℕ
  h1 : defective + non_defective = total

/-- Probability of drawing two defective bulbs -/
def prob_two_defective (box : BulbBox) : ℚ :=
  (box.defective * box.defective : ℚ) / (box.total * box.total)

/-- Probability of drawing one defective and one non-defective bulb -/
def prob_one_defective_one_non (box : BulbBox) : ℚ :=
  (2 * box.defective * box.non_defective : ℚ) / (box.total * box.total)

/-- Probability of drawing at least one non-defective bulb -/
def prob_at_least_one_non_defective (box : BulbBox) : ℚ :=
  1 - prob_two_defective box

/-- The main theorem about probabilities in the bulb box problem -/
theorem bulb_box_probabilities (box : BulbBox) 
  (h2 : box.total = 6) (h3 : box.defective = 2) (h4 : box.non_defective = 4) :
  prob_two_defective box = 1/9 ∧
  prob_one_defective_one_non box = 4/9 ∧
  prob_at_least_one_non_defective box = 8/9 := by
  sorry

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bulb_box_probabilities_l869_86969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implication_l869_86968

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.sin x + Real.cos x

noncomputable def g (a : ℝ) (x : ℝ) := Real.sin x + a * Real.cos x

def symmetric_about (h : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, h (c + x) = h (c - x)

theorem symmetry_implication (a : ℝ) :
  symmetric_about (f a) (π / 6) → symmetric_about (g a) (π / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implication_l869_86968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l869_86948

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the points
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (6, 2)

-- Define the line
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

-- Define the theorem
theorem circle_and_line_properties :
  ∃ (M : ℝ × ℝ) (r : ℝ),
    -- Circle M passes through A and B
    A ∈ Circle M r ∧ B ∈ Circle M r ∧
    -- CD is on the perpendicular bisector of AB
    ∃ (C D : ℝ × ℝ),
      C ∈ Circle M r ∧ D ∈ Circle M r ∧
      -- |CD| = 2√10
      (C.1 - D.1)^2 + (C.2 - D.2)^2 = 40 ∧
      -- Equation of line CD: 2x - y - 5 = 0
      (∀ (x y : ℝ), (x, y) ∈ Line C D ↔ 2*x - y - 5 = 0) ∧
      -- Equation of circle M
      ((M = (5, 5) ∧ r^2 = 10) ∨ (M = (3, 1) ∧ r^2 = 10)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l869_86948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_email_phone_text_relationship_l869_86903

/-- Represents the number of phone messages -/
def P : ℕ := sorry

/-- Represents the number of emails -/
def E : ℕ := sorry

/-- Represents the number of texts -/
def T : ℕ := sorry

/-- Theorem stating the relationship between emails and phone messages,
    the total number of emails and phone messages,
    and the relationship between texts and emails -/
theorem email_phone_text_relationship 
  (h1 : E = 9 * P - 7)
  (h2 : E + P = 93)
  (h3 : T = E^2 + 64) :
  E = 83 ∧ T = 6953 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_email_phone_text_relationship_l869_86903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_intersect_Q_l869_86901

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | x ≤ 2}

def P_real : Set ℝ := {1, 2, 3, 4}

theorem P_intersect_Q : P_real ∩ Q = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_intersect_Q_l869_86901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_midpoint_x_coordinate_l869_86983

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line type -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Point on a parabola -/
def PointOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  p.eq x y

/-- Midpoint of two points -/
noncomputable def Midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

/-- Perpendicular bisector passes through a point -/
def PerpendicularBisectorThrough (a b m : ℝ × ℝ) : Prop :=
  let mid := Midpoint a b
  (m.2 - mid.2) * (b.1 - a.1) = (m.1 - mid.1) * (b.2 - a.2)

/-- Main theorem -/
theorem parabola_midpoint_x_coordinate
  (p : Parabola)
  (h_eq : p.eq = fun x y ↦ y^2 = 4*x)
  (h_focus : p.focus = (1, 0))
  (a b : ℝ × ℝ)
  (h_a_on_p : PointOnParabola p a.1 a.2)
  (h_b_on_p : PointOnParabola p b.1 b.2)
  (h_not_parallel : a.2 ≠ b.2)
  (h_perp_bisector : PerpendicularBisectorThrough a b (4, 0)) :
  (Midpoint a b).1 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_midpoint_x_coordinate_l869_86983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_of_children_l869_86911

theorem average_weight_of_children (boys_count : ℕ) (girls_count : ℕ) 
  (boys_avg_weight : ℝ) (girls_avg_weight : ℝ) :
  boys_count = 8 →
  girls_count = 5 →
  boys_avg_weight = 160 →
  girls_avg_weight = 110 →
  let total_weight := boys_count * boys_avg_weight + girls_count * girls_avg_weight
  let total_count := boys_count + girls_count
  (Int.floor ((total_weight / total_count) + 0.5) : ℝ) = 141 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_of_children_l869_86911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_2011_l869_86928

-- Define the sequence as a function
def sequenceA (n : ℕ) : ℕ :=
  (10^n - 1) / 9

-- Theorem stating that at least one number in the sequence is divisible by 2011
theorem divisible_by_2011 :
  ∃ k ∈ Finset.range 2012, sequenceA k % 2011 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_2011_l869_86928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_double_probability_l869_86939

/-- A domino is a pair of integers -/
def Domino := ℕ × ℕ

/-- A complete set of dominoes with integers from 0 to 12 -/
def CompleteSet : Set Domino :=
  {d | d.1 ≤ 12 ∧ d.2 ≤ 12 ∧ (d.1 < d.2 ∨ d.1 = d.2)}

/-- A double is a domino with the same integer on both squares -/
def IsDouble (d : Domino) : Prop := d.1 = d.2

/-- The number of dominoes in a complete set -/
def SetSize : ℕ := 91

/-- The number of doubles in a complete set -/
def DoubleCount : ℕ := 13

theorem domino_double_probability :
  (DoubleCount : ℚ) / SetSize = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_double_probability_l869_86939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_five_halves_l869_86944

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x) else 0  -- We define f explicitly only for [0,1]

-- State the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

-- State the theorem
theorem f_neg_five_halves : f (-5/2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_five_halves_l869_86944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_product_prime_factors_l869_86922

theorem divisors_product_prime_factors (n : ℕ) (hn : n = 60) :
  (Finset.prod (Nat.divisors n) id).factorization.support.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_product_prime_factors_l869_86922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_special_properties_l869_86996

theorem triangle_with_special_properties (a b c : ℕ) (r ra rb rc : ℕ) (area : ℕ) :
  -- Conditions
  r = 1 ∧                                  -- incircle radius is 1
  area = (a + b + c) / 2 * r ∧             -- area formula using incircle radius
  ra = area / ((b + c - a) / 2) ∧          -- excircle radius formula
  rb = area / ((a + c - b) / 2) ∧
  rc = area / ((a + b - c) / 2) ∧
  1 / ra + 1 / rb + 1 / rc = 1             -- sum of reciprocals of excircle radii
  →
  -- Conclusion
  a = 3 ∧ b = 4 ∧ c = 5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_special_properties_l869_86996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l869_86946

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of the foci -/
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 3, 0)

/-- Definition of the dot product of vectors PF1 and PF2 -/
noncomputable def dot_product_PF1_PF2 (x y : ℝ) : ℝ :=
  (x + Real.sqrt 3) * (x - Real.sqrt 3) + y * (-y)

/-- Definition of the fixed point M -/
def M : ℝ × ℝ := (0, 2)

/-- Definition of acute angle -/
def is_acute_angle (A B : ℝ × ℝ) : Prop :=
  (A.1 * B.1 + A.2 * B.2) > 0

/-- Main theorem -/
theorem ellipse_properties :
  (∀ x y : ℝ, is_on_ellipse x y →
    dot_product_PF1_PF2 x y ≤ 1 ∧ dot_product_PF1_PF2 x y ≥ -2) ∧
  (∀ k : ℝ, (∃ A B : ℝ × ℝ,
    is_on_ellipse A.1 A.2 ∧ is_on_ellipse B.1 B.2 ∧
    A ≠ B ∧
    (A.2 - M.2) = k * (A.1 - M.1) ∧
    (B.2 - M.2) = k * (B.1 - M.1) ∧
    is_acute_angle A B) →
    (-2 < k ∧ k < -Real.sqrt 3 / 2) ∨ (Real.sqrt 3 / 2 < k ∧ k < 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l869_86946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_polynomial_form_l869_86906

/-- A polynomial that satisfies the given equation for infinitely many real values of x. -/
def SatisfyingPolynomial (f : ℝ → ℂ) : Prop :=
  ∃ S : Set ℝ, Set.Infinite S ∧ ∀ x ∈ S, f (3 * x) / f x = 729 * (x - 3) / (x - 243)

/-- The form of the polynomial that satisfies the equation. -/
def PolynomialForm (f : ℝ → ℂ) : Prop :=
  ∃ a : ℂ, ∀ x : ℝ, f x = a * (x - 243) * (x - 81) * (x - 27) * (x - 9) * x^2

/-- Theorem stating that any satisfying polynomial must be of the specified form. -/
theorem satisfying_polynomial_form (f : ℝ → ℂ) :
  SatisfyingPolynomial f → PolynomialForm f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_polynomial_form_l869_86906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_length_is_zero_l869_86999

/-- Triangle DEF with specific properties -/
structure SpecialTriangle where
  /-- Point D of the triangle -/
  D : ℝ × ℝ
  /-- Point E of the triangle -/
  E : ℝ × ℝ
  /-- Point F of the triangle -/
  F : ℝ × ℝ
  /-- Length of side DE -/
  DE_length : dist D E = 10
  /-- Length of side EF -/
  EF_length : dist E F = 12
  /-- Length of side FD -/
  FD_length : dist F D = 14
  /-- Point K on altitude DK -/
  K : ℝ × ℝ
  /-- DK is perpendicular to EF -/
  DK_altitude : (D.1 - K.1) * (E.1 - F.1) + (D.2 - K.2) * (E.2 - F.2) = 0
  /-- Point G on DE -/
  G : ℝ × ℝ
  /-- Point H on DF -/
  H : ℝ × ℝ
  /-- EG is an angle bisector -/
  EG_bisector : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ G = (t * D.1 + (1 - t) * E.1, t * D.2 + (1 - t) * E.2)
  /-- FH is an angle bisector -/
  FH_bisector : ∃ s : ℝ, 0 < s ∧ s < 1 ∧ H = (s * D.1 + (1 - s) * F.1, s * D.2 + (1 - s) * F.2)
  /-- EG is a median -/
  EG_median : dist D G = dist G E
  /-- FH is a median -/
  FH_median : dist D H = dist H F
  /-- Point M where EG intersects DK -/
  M : ℝ × ℝ
  /-- Point N where FH intersects DK -/
  N : ℝ × ℝ
  /-- M is on EG -/
  M_on_EG : ∃ u : ℝ, 0 ≤ u ∧ u ≤ 1 ∧ M = (u * E.1 + (1 - u) * G.1, u * E.2 + (1 - u) * G.2)
  /-- N is on FH -/
  N_on_FH : ∃ v : ℝ, 0 ≤ v ∧ v ≤ 1 ∧ N = (v * F.1 + (1 - v) * H.1, v * F.2 + (1 - v) * H.2)
  /-- M is on DK -/
  M_on_DK : ∃ w : ℝ, 0 ≤ w ∧ w ≤ 1 ∧ M = (w * D.1 + (1 - w) * K.1, w * D.2 + (1 - w) * K.2)
  /-- N is on DK -/
  N_on_DK : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ N = (x * D.1 + (1 - x) * K.1, x * D.2 + (1 - x) * K.2)

/-- The length of MN in a SpecialTriangle is 0 -/
theorem mn_length_is_zero (t : SpecialTriangle) : dist t.M t.N = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_length_is_zero_l869_86999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_difference_l869_86984

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_difference (a b : V) 
  (ha : ‖a‖ = 2)
  (hb : ‖b‖ = 1)
  (hab : ‖a + b‖ = Real.sqrt 3) :
  ‖a - b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_difference_l869_86984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_calculation_l869_86907

/-- Taxi fare function -/
noncomputable def taxiFare (a : ℝ) : ℝ :=
  if a ≤ 3 then 5 else 5 + 1.2 * (a - 3)

/-- Theorem: Taxi fare calculation for distances greater than 3 km -/
theorem taxi_fare_calculation (a : ℝ) (h : a > 3) :
  taxiFare a = 1.2 * a + 1.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_calculation_l869_86907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l869_86986

theorem ticket_price_possibilities : 
  let possible_prices := {x : ℕ | x > 0 ∧ 36 % x = 0 ∧ 54 % x = 0}
  Finset.card (Finset.filter (λ x => x > 0 ∧ 36 % x = 0 ∧ 54 % x = 0) (Finset.range 55)) = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l869_86986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l869_86994

/-- Given a function f: ℝ → ℝ satisfying certain properties, prove that f(3) = 3 -/
theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 1) = f (3 - x)) 
  (h2 : f 1 = 3) : 
  f 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l869_86994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l869_86900

-- Define the initial angle
def initial_angle : ℚ := 60

-- Define the rotation angle
def rotation_angle : ℚ := 630

-- Function to calculate the resulting acute angle after rotation
noncomputable def resulting_acute_angle (initial : ℚ) (rotation : ℚ) : ℚ :=
  let total := (initial + rotation) % 360
  min total (360 - total)

-- Theorem statement
theorem rotation_result :
  resulting_acute_angle initial_angle rotation_angle = 30 := by
  -- Unfold the definitions
  unfold resulting_acute_angle
  unfold initial_angle
  unfold rotation_angle
  -- Simplify the arithmetic
  simp
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l869_86900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_pow_eighteen_nine_pow_nine_twelve_pow_twelve_count_k_values_correct_l869_86960

/-- The number of positive integer values of k for which 18^18 is the LCM of 9^9, 12^12, and k -/
def count_k_values : ℕ := 19

/-- 18^18 is equal to 2^18 * 3^36 -/
theorem eighteen_pow_eighteen : 18^18 = 2^18 * 3^36 := by sorry

/-- 9^9 is equal to 3^18 -/
theorem nine_pow_nine : 9^9 = 3^18 := by sorry

/-- 12^12 is equal to 2^24 * 3^12 -/
theorem twelve_pow_twelve : 12^12 = 2^24 * 3^12 := by sorry

theorem count_k_values_correct :
  count_k_values = (Finset.range 19).card ∧
  ∀ k : ℕ, k > 0 →
    (∃ a b : ℕ, k = 2^a * 3^b ∧ a ≤ 18 ∧ b = 36) ↔
    Nat.lcm (Nat.lcm (9^9) (12^12)) k = 18^18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_pow_eighteen_nine_pow_nine_twelve_pow_twelve_count_k_values_correct_l869_86960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_l869_86936

-- Define the function f(x) as noncomputable due to Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x * (x - a))

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, Monotone (f a)) → a ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_l869_86936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l869_86920

/-- Represents the number of sides in a convex polygon. -/
def n : ℕ := sorry

/-- Represents the common difference in the arithmetic sequence of interior angles. -/
noncomputable def common_difference : ℝ := 10

/-- Represents the largest interior angle of the polygon. -/
noncomputable def largest_angle : ℝ := 180

/-- Represents the sum of interior angles of a polygon with n sides. -/
noncomputable def sum_of_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Represents the first term in the arithmetic sequence of interior angles. -/
noncomputable def first_angle : ℝ := largest_angle - common_difference * (n - 1)

/-- Represents the sum of angles using the arithmetic sequence formula. -/
noncomputable def sum_from_sequence : ℝ := n * (first_angle + largest_angle) / 2

theorem polygon_sides_count :
  sum_of_angles n = sum_from_sequence → n = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l869_86920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_finishes_eleventh_l869_86937

/-- Represents the position of a racer in the race. -/
def Position := Fin 15

/-- Represents a racer in the race. -/
inductive Racer
| Ian
| Maria
| Zack
| Olivia
| Nathan
| Leo

/-- Represents the finishing order of the race. -/
def FinishingOrder := Racer → Position

def race_result (finish : FinishingOrder) : Prop :=
  (finish Racer.Nathan).val - (finish Racer.Maria).val = 7 ∧
  (finish Racer.Olivia).val - (finish Racer.Zack).val = 2 ∧
  (finish Racer.Ian).val - (finish Racer.Maria).val = 3 ∧
  (finish Racer.Zack).val - (finish Racer.Leo).val = 3 ∧
  (finish Racer.Leo).val - (finish Racer.Nathan).val = 2 ∧
  (finish Racer.Olivia).val = 9

theorem maria_finishes_eleventh (finish : FinishingOrder) 
  (h : race_result finish) : (finish Racer.Maria).val = 11 := by
  sorry

#check maria_finishes_eleventh

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_finishes_eleventh_l869_86937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l869_86931

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 4}

def B : Set ℝ := {y : ℝ | ∃ x ∈ A, y = x + 1}

theorem complement_intersection_theorem :
  (Set.univ \ A) ∩ (Set.univ \ B) = Iic (-1) ∪ Ici 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l869_86931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l869_86982

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := sorry

-- Define the points P and Q
def P : ℝ × ℝ := (-2, 4)
def Q : ℝ × ℝ := (4, 4)

-- Define the diameter of the circle
noncomputable def diameter : ℝ := 2 * Real.sqrt 10

-- Theorem statement
theorem circle_equation :
  (P ∈ circle_C) ∧ (Q ∈ circle_C) ∧ (diameter = 2 * Real.sqrt 10) →
  (∃ (h : ℝ) (k : ℝ), 
    (circle_C = {(x, y) | (x - 1)^2 + (y - h)^2 = 10}) ∧ 
    (h = 3 ∨ h = 5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l869_86982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_35_l869_86974

/-- Represents a solution to the equation with unique single-digit numbers -/
structure Solution :=
  (a b c d e : Nat)
  (unique : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (single_digit : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10)
  (equation_holds : ∃ (f g h i j : Nat), 
    f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10 ∧
    Finset.card (Finset.range 10 ∩ {a, b, c, d, e, f, g, h, i, j}) = 10 ∧
    a * (b + 10 * c) * (d + e + f + 10 * g) = 2014)

/-- The maximum sum of five distinct single-digit numbers in any valid solution is 35 -/
theorem max_sum_is_35 : 
  (∀ s : Solution, s.a + s.b + s.c + s.d + s.e ≤ 35) ∧ 
  (∃ s : Solution, s.a + s.b + s.c + s.d + s.e = 35) := by
  sorry

#check max_sum_is_35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_35_l869_86974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_image_parametric_number_l869_86930

/-- A triangular pyramid with specific properties -/
structure TriangularPyramid where
  base_is_equilateral : Bool
  lateral_face_perpendicular : Bool
  lateral_face_is_equilateral : Bool

/-- The image of the pyramid as a quadrilateral -/
structure PyramidImage where
  is_quadrilateral : Bool
  has_diagonals : Bool

/-- The parametric number of a geometric object -/
def parametric_number (i : PyramidImage) : ℕ := 5

/-- Theorem stating that the parametric number of the pyramid image is 5 -/
theorem pyramid_image_parametric_number 
  (p : TriangularPyramid) 
  (i : PyramidImage) 
  (h1 : p.base_is_equilateral = true)
  (h2 : p.lateral_face_perpendicular = true)
  (h3 : p.lateral_face_is_equilateral = true)
  (h4 : i.is_quadrilateral = true)
  (h5 : i.has_diagonals = true) :
  parametric_number i = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_image_parametric_number_l869_86930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l869_86945

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x - 2)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x < -1 ∨ x > 2

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x y, domain x → domain y → x < y → x < -1 → y < -1 → f y < f x :=
by
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l869_86945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l869_86979

/-- Given a hyperbola with equation y² - x²/3 = 1, 
    its asymptotic line equation is y = ±(√3/3)x -/
theorem hyperbola_asymptote (x y : ℝ) :
  (y^2 - x^2 / 3 = 1) →
  ∃ (k : ℝ), k = Real.sqrt 3 / 3 ∧ (y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l869_86979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_to_polar_circle_l869_86952

/-- A circle C in the Cartesian plane -/
structure CartesianCircle where
  /-- The Cartesian equation of the circle -/
  equation : ℝ → ℝ → Prop

/-- A circle C in the polar coordinate system -/
structure PolarCircle where
  /-- The polar equation of the circle -/
  equation : ℝ → ℝ → Prop

/-- The transformation from Cartesian to polar coordinates -/
def cartesianToPolar (x y ρ θ : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

/-- Theorem: The polar equation of a circle with Cartesian equation x² + y² - 2x = 0 is ρ = 2cosθ -/
theorem cartesian_to_polar_circle :
  ∀ (C : CartesianCircle),
    (∀ x y, C.equation x y ↔ x^2 + y^2 - 2*x = 0) →
    ∃ (P : PolarCircle),
      (∀ ρ θ, P.equation ρ θ ↔ ρ = 2 * Real.cos θ) ∧
      (∀ x y ρ θ, cartesianToPolar x y ρ θ → (C.equation x y ↔ P.equation ρ θ)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_to_polar_circle_l869_86952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_150_degrees_l869_86909

theorem csc_150_degrees : 1 / Real.sin (150 * π / 180) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_150_degrees_l869_86909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l869_86935

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l869_86935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_film_radius_for_given_dimensions_l869_86962

/-- The radius of a circular film formed by liquid X on water -/
noncomputable def film_radius (length width height thickness : ℝ) : ℝ :=
  Real.sqrt ((length * width * height) / (Real.pi * thickness))

/-- Theorem stating the radius of the circular film for given dimensions -/
theorem film_radius_for_given_dimensions :
  film_radius 6 3 12 0.1 = Real.sqrt (2160 / Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_film_radius_for_given_dimensions_l869_86962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l869_86988

theorem polynomial_remainder (p : ℝ → ℝ) (r : ℝ → ℝ) :
  (∀ x, ∃ q₁ : ℝ → ℝ, p x = (x - 3) * q₁ x + 2) →
  (∀ x, ∃ q₂ : ℝ → ℝ, p x = (x + 1) * q₂ x - 2) →
  (∀ x, ∃ q₃ : ℝ → ℝ, p x = (x - 4) * q₃ x + 5) →
  (∀ x, ∃ q : ℝ → ℝ, p x = (x - 3) * (x + 1) * (x - 4) * q x + r x) →
  r 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l869_86988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l869_86975

theorem imaginary_part_of_z : Complex.im ((2 : ℂ) + Complex.I * Complex.I) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l869_86975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l869_86973

-- Define the line l: x = my + 2
def line (m : ℝ) (x y : ℝ) : Prop := x = m * y + 2

-- Define the circle M: x^2 + 2x + y^2 + 2y = 0
def circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 + 2*y = 0

-- Define the tangency condition
def is_tangent (m : ℝ) : Prop := ∃ x y : ℝ, line m x y ∧ circle_equation x y ∧
  ∀ x' y' : ℝ, line m x' y' → circle_equation x' y' → (x = x' ∧ y = y')

-- Theorem statement
theorem tangent_line_slope :
  ∀ m : ℝ, is_tangent m → (m = 1 ∨ m = -7) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l869_86973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_in_list_l869_86957

def number_list : List Nat := [33, 35, 37, 39, 41]

def is_prime (n : Nat) : Bool :=
  if n ≤ 1 then false
  else
    let sqrt_n := Nat.sqrt n
    (List.range (sqrt_n - 1)).all (fun m => n % (m + 2) ≠ 0)

def arithmetic_mean (l : List Nat) : Rat :=
  (l.sum : Rat) / l.length

theorem arithmetic_mean_of_primes_in_list :
  arithmetic_mean (number_list.filter is_prime) = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_in_list_l869_86957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_abundant_not_multiple_of_4_l869_86902

def isAbundant (n : ℕ) : Prop :=
  n > 0 ∧ (Finset.sum (Nat.properDivisors n) id) > n

def isMultipleOf4 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * k

theorem smallest_abundant_not_multiple_of_4 :
  (∀ m : ℕ, m < 18 → ¬(isAbundant m ∧ ¬isMultipleOf4 m)) ∧
  (isAbundant 18 ∧ ¬isMultipleOf4 18) := by
  sorry

#check smallest_abundant_not_multiple_of_4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_abundant_not_multiple_of_4_l869_86902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l869_86964

-- Define the initial and final parabolas
def initial_parabola (x : ℝ) : ℝ := 2 * (x + 1)^2 - 3
def final_parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the translation
def translate (f : ℝ → ℝ) (h v : ℝ) : ℝ → ℝ := λ x ↦ f (x - h) + v

-- Theorem statement
theorem parabola_translation :
  ∀ x, translate initial_parabola 1 3 x = final_parabola x := by
  sorry

#check parabola_translation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l869_86964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_preservation_impossibility_l869_86998

/-- Represents the operations that can be performed on a pair of natural numbers -/
inductive Operation
  | swap
  | sum_first
  | abs_diff

/-- Applies an operation to a pair of natural numbers -/
def apply_operation (op : Operation) (pair : ℕ × ℕ) : ℕ × ℕ :=
  match op with
  | Operation.swap => (pair.2, pair.1)
  | Operation.sum_first => (pair.1 + pair.2, pair.2)
  | Operation.abs_diff => (pair.1, Int.natAbs (pair.1 - pair.2))

/-- The initial pair of numbers on the card -/
def initial_pair : ℕ × ℕ := (1037, 1159)

/-- The desired pair of numbers -/
def desired_pair : ℕ × ℕ := (611, 1081)

/-- Applies a list of operations to a pair of numbers -/
def apply_operations (ops : List Operation) (pair : ℕ × ℕ) : ℕ × ℕ :=
  ops.foldl (fun p op => apply_operation op p) pair

theorem gcd_preservation_impossibility :
  ∀ (ops : List Operation),
    Nat.gcd (apply_operations ops initial_pair).1
            (apply_operations ops initial_pair).2 =
    Nat.gcd initial_pair.1 initial_pair.2 ∧
    Nat.gcd initial_pair.1 initial_pair.2 ≠ Nat.gcd desired_pair.1 desired_pair.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_preservation_impossibility_l869_86998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2019_l869_86913

def f : ℕ → ℚ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | n + 2 => f (n + 1) + 1 / ((n + 2) * (n + 1))

theorem f_2019 : f 2019 = 4037 / 2019 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2019_l869_86913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_westbound_vehicles_approximation_l869_86967

/-- Approximates the number of westbound vehicles in a highway section -/
theorem westbound_vehicles_approximation
  (eastbound_speed : ℝ)
  (westbound_speed : ℝ)
  (observation_time : ℝ)
  (stop_time : ℝ)
  (observed_vehicles : ℕ)
  (section_length : ℝ)
  (h1 : eastbound_speed = 70)
  (h2 : westbound_speed = 60)
  (h3 : observation_time = 10 / 60)  -- Convert to hours
  (h4 : stop_time = 2 / 60)          -- Convert to hours
  (h5 : observed_vehicles = 15)
  (h6 : section_length = 150)
  : Int.floor ((observed_vehicles : ℝ) * section_length / 
    ((eastbound_speed + westbound_speed) * (observation_time - stop_time))) = 130 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_westbound_vehicles_approximation_l869_86967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l869_86990

/-- Proves that the average speed of a car is (3150 / 85) km/hr given specific uphill and downhill speeds and distances. -/
theorem car_average_speed (uphill_speed : ℝ) (downhill_speed : ℝ) (uphill_distance : ℝ) (downhill_distance : ℝ)
    (h1 : uphill_speed = 30)
    (h2 : downhill_speed = 70)
    (h3 : uphill_distance = 100)
    (h4 : downhill_distance = 50) :
    (uphill_distance + downhill_distance) / ((uphill_distance / uphill_speed) + (downhill_distance / downhill_speed)) = 3150 / 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l869_86990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l869_86943

noncomputable def P : ℝ × ℝ := (Real.sqrt 3, 1)

theorem rotation_result (Q : ℝ × ℝ) :
  (Q.1 = -1 ∧ Q.2 = Real.sqrt 3) ↔
  (Q.1^2 + Q.2^2 = P.1^2 + P.2^2 ∧ Q.1 * P.1 + Q.2 * P.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l869_86943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_when_k_is_1_range_of_k_when_complement_A_subset_complement_B_l869_86915

-- Define set A
def A : Set ℝ := {a : ℝ | ∀ x : ℝ, x^2 + 2*a*x + 4 > 0}

-- Define set B
def B (k : ℝ) : Set ℝ := {x : ℝ | 1 < (x + k) / 2 ∧ (x + k) / 2 < 2}

-- Theorem for part 1
theorem intersection_A_complement_B_when_k_is_1 :
  A ∩ (Set.univ \ B 1) = Set.Ico (-2) 1 := by sorry

-- Theorem for part 2
theorem range_of_k_when_complement_A_subset_complement_B :
  ∀ k : ℝ, (Set.univ \ A) ⊂ (Set.univ \ B k) → 2 ≤ k ∧ k ≤ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_when_k_is_1_range_of_k_when_complement_A_subset_complement_B_l869_86915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l869_86929

def U : Set ℤ := {x | -2 < x ∧ x ≤ 3 ∧ x ≥ 0}
def A : Set ℤ := {3}

theorem complement_of_A_in_U : Set.compl A ∩ U = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l869_86929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l869_86904

-- Define the ellipse C
def ellipse_C (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 / a^2 + p.2^2 / b^2 = 1}

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ a^2 / b^2 = 3 ∧
  ∃ (c : ℝ), c^2 = a^2 - b^2 ∧ b * c = 5 * Real.sqrt 2 / 3

-- Define the line intersecting the ellipse
def intersecting_line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * (p.1 + 1)}

-- Main theorem
theorem ellipse_properties (a b : ℝ) (h : conditions a b) :
  -- 1) Standard equation
  ellipse_C a b = ellipse_C (Real.sqrt 5) (Real.sqrt (5/3)) ∧
  -- 2) Slope when midpoint x-coordinate is -1/2
  (∀ k : ℝ, (∃ A B : ℝ × ℝ, A ∈ ellipse_C a b ∩ intersecting_line k ∧
                           B ∈ ellipse_C a b ∩ intersecting_line k ∧
                           (A.1 + B.1) / 2 = -1/2) →
    k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3) ∧
  -- 3) Constant dot product
  ∀ k : ℝ, ∀ A B : ℝ × ℝ,
    A ∈ ellipse_C a b ∩ intersecting_line k →
    B ∈ ellipse_C a b ∩ intersecting_line k →
    let M : ℝ × ℝ := (-7/3, 0)
    let MA : ℝ × ℝ := (A.1 - M.1, A.2 - M.2)
    let MB : ℝ × ℝ := (B.1 - M.1, B.2 - M.2)
    MA.1 * MB.1 + MA.2 * MB.2 = 4/9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l869_86904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l869_86980

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := 1/2 * t.b * t.c * Real.sin t.A

/-- Triangle ABC is right-angled -/
def is_right_angled (t : Triangle) : Prop := t.B = Real.pi / 2 ∨ t.C = Real.pi / 2 ∨ t.A = Real.pi / 2

/-- Triangle ABC is isosceles -/
def is_isosceles (t : Triangle) : Prop := t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

theorem triangle_properties (t : Triangle) :
  (t.a = 2 * Real.sqrt 3 ∧ t.A = Real.pi / 3 ∧ area t = 2 * Real.sqrt 3 →
    (t.b = 2 ∧ t.c = 4) ∨ (t.b = 4 ∧ t.c = 2)) ∧
  (Real.sin (t.C - t.B) = Real.sin (2 * t.B) - Real.sin t.A →
    is_right_angled t ∨ is_isosceles t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l869_86980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l869_86965

/-- Parabola definition -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Focus point -/
def F : ℝ × ℝ := (2, 0)

/-- Directrix intersection with x-axis -/
def K : ℝ × ℝ := (-2, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Point P satisfies the condition |PK| = √2|PF| -/
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  distance P K = Real.sqrt 2 * distance P F

/-- Area of triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

theorem parabola_triangle_area :
  ∀ P : ℝ × ℝ, parabola P.1 P.2 → satisfies_condition P →
  triangle_area P K F = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l869_86965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l869_86941

-- Define the ellipse
def ellipse : ℝ × ℝ → Prop := λ p => let (x, y) := p; x^2/16 + y^2/36 = 1

-- Define the parabola (we don't know its exact equation, but we know it exists)
structure Parabola where
  equation : ℝ × ℝ → Prop

-- Define the property of sharing a common focus
def share_focus (e : ℝ × ℝ → Prop) (p : Parabola) : Prop := sorry

-- Define the property of the parabola's directrix being along the minor axis of the ellipse
def directrix_on_minor_axis (e : ℝ × ℝ → Prop) (p : Parabola) : Prop := sorry

-- Define the property of intersecting at exactly two points
def intersect_at_two_points (e : ℝ × ℝ → Prop) (p : Parabola) : Prop := sorry

-- Define the distance between intersection points
noncomputable def distance_between_intersections (e : ℝ × ℝ → Prop) (p : Parabola) : ℝ := sorry

-- Theorem statement
theorem intersection_distance :
  ∀ (p : Parabola),
  share_focus ellipse p →
  directrix_on_minor_axis ellipse p →
  intersect_at_two_points ellipse p →
  distance_between_intersections ellipse p = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l869_86941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l869_86989

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the sets A and B
def A : Set ℂ := {1, i}
def B : Set ℂ := {-1/i, (1-i)^2/2}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {1, i, -i} := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l869_86989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_range_l869_86912

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the side lengths
noncomputable def side_length (a b : ℝ) : ℝ :=
  Real.sqrt ((a - b)^2)

-- Define the angle measure
noncomputable def angle_measure (a b c : ℝ) : ℝ :=
  Real.arccos ((side_length b c)^2 + (side_length a c)^2 - (side_length a b)^2) / (2 * side_length b c * side_length a c)

theorem angle_A_range (t : Triangle) :
  side_length t.B t.C = 2 →
  side_length t.A t.C = 2 * Real.sqrt 2 →
  0 < angle_measure t.B t.A t.C ∧ angle_measure t.B t.A t.C ≤ π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_range_l869_86912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l869_86919

-- Define set A
def A : Set ℝ := {y | ∃ x : ℝ, y = Real.exp x}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - x - 6 ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l869_86919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_negative_five_l869_86997

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 12) / (x + 5)

theorem vertical_asymptote_at_negative_five :
  ∃ (M : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    ∀ (x : ℝ), 0 < |x + 5| ∧ |x + 5| < δ → |f x| > M :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_negative_five_l869_86997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_squared_trilinear_coordinates_l869_86914

/-- Given a triangle with angles α, β, γ, and two points M and N with absolute trilinear coordinates
    (x₁, y₁, z₁) and (x₂, y₂, z₂) respectively, this theorem states that the square of the distance
    between M and N is given by the formula:
    MN² = (cos α / (sin β sin γ)) * (x₁ - x₂)² + (cos β / (sin γ sin α)) * (y₁ - y₂)² + (cos γ / (sin α sin β)) * (z₁ - z₂)² -/
theorem distance_squared_trilinear_coordinates
  (α β γ : ℝ) (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) :
  α + β + γ = π →
  ∃ (MN : ℝ),
    MN^2 = (Real.cos α / (Real.sin β * Real.sin γ)) * (x₁ - x₂)^2 +
           (Real.cos β / (Real.sin γ * Real.sin α)) * (y₁ - y₂)^2 +
           (Real.cos γ / (Real.sin α * Real.sin β)) * (z₁ - z₂)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_squared_trilinear_coordinates_l869_86914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winner_average_jump_is_22_feet_l869_86958

/-- Represents an athlete's jump distances in feet -/
structure AthleteJumps where
  long : ℝ
  triple : ℝ
  high : ℝ

/-- Calculates the average jump distance for an athlete -/
noncomputable def averageJump (jumps : AthleteJumps) : ℝ :=
  (jumps.long + jumps.triple + jumps.high) / 3

/-- Determines the winner's average jump distance -/
noncomputable def winnerAverageJump (athlete1 : AthleteJumps) (athlete2 : AthleteJumps) : ℝ :=
  max (averageJump athlete1) (averageJump athlete2)

theorem winner_average_jump_is_22_feet 
  (athlete1 : AthleteJumps) 
  (athlete2 : AthleteJumps)
  (h1 : athlete1 = { long := 26, triple := 30, high := 7 })
  (h2 : athlete2 = { long := 24, triple := 34, high := 8 }) :
  winnerAverageJump athlete1 athlete2 = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_winner_average_jump_is_22_feet_l869_86958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_successful_transplants_approx_26_67_l869_86924

/-- Represents a plot with seeds, germination rate, and transplant success rate -/
structure Plot where
  seeds : ℕ
  germination_rate : ℝ
  transplant_success_rate : ℝ

/-- Calculates the number of successfully transplanted seeds for a plot -/
def successful_transplants (p : Plot) : ℝ :=
  p.seeds * p.germination_rate * p.transplant_success_rate

/-- The list of plots with their respective data -/
def plots : List Plot := [
  { seeds := 300, germination_rate := 0.25, transplant_success_rate := 0.90 },
  { seeds := 200, germination_rate := 0.35, transplant_success_rate := 0.85 },
  { seeds := 400, germination_rate := 0.45, transplant_success_rate := 0.80 },
  { seeds := 350, germination_rate := 0.15, transplant_success_rate := 0.95 },
  { seeds := 150, germination_rate := 0.50, transplant_success_rate := 0.70 }
]

/-- The total number of seeds planted across all plots -/
def total_seeds : ℕ := (plots.map (λ p => p.seeds)).sum

/-- The total number of successfully transplanted seeds across all plots -/
def total_successful : ℝ := (plots.map successful_transplants).sum

/-- The theorem stating that the percentage of successfully transplanted seeds is approximately 26.67% -/
theorem percentage_successful_transplants_approx_26_67 :
  ∃ ε > 0, |((total_successful / (total_seeds : ℝ)) * 100 - 26.67)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_successful_transplants_approx_26_67_l869_86924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l869_86992

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The first line -/
noncomputable def line1 : Line2D := { point := (2, 4), direction := (3, -2) }

/-- The second line -/
noncomputable def line2 : Line2D := { point := (-1, 5), direction := (6, 1) }

/-- A point lies on a line if there exists a parameter t such that
    the point equals the line's point plus t times the direction vector -/
def pointOnLine (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p.1 = l.point.1 + t * l.direction.1 ∧ p.2 = l.point.2 + t * l.direction.2

/-- The intersection point of the two lines -/
noncomputable def intersectionPoint : ℝ × ℝ := (1/5, 26/5)

/-- Theorem: The intersection point lies on both lines and is unique -/
theorem intersection_point_correct :
  pointOnLine intersectionPoint line1 ∧
  pointOnLine intersectionPoint line2 ∧
  ∀ p : ℝ × ℝ, pointOnLine p line1 ∧ pointOnLine p line2 → p = intersectionPoint :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l869_86992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_concurrent_lines_and_barycentric_coords_l869_86972

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the incircle tangency points
noncomputable def incircle_tangency_points (t : Triangle) : ℝ × ℝ × ℝ :=
  ((t.b + t.c - t.a) / 2, (t.a + t.c - t.b) / 2, (t.a + t.b - t.c) / 2)

-- Define the excircle tangency points (same as incircle for this problem)
noncomputable def excircle_tangency_points (t : Triangle) : ℝ × ℝ × ℝ :=
  incircle_tangency_points t

-- Theorem statement
theorem triangle_concurrent_lines_and_barycentric_coords (t : Triangle) :
  let (x, y, z) := incircle_tangency_points t
  -- 1. AP, BQ, and CR are concurrent
  ∃ point : ℝ × ℝ × ℝ, True ∧
  -- 2. AP', BQ', and CR' are concurrent
  ∃ point' : ℝ × ℝ × ℝ, True ∧
  -- 3. Barycentric coordinates of both intersection points
  point = (1/x, 1/y, 1/z) ∧ point' = (1/x, 1/y, 1/z) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_concurrent_lines_and_barycentric_coords_l869_86972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_motion_l869_86971

/-- Represents the speed of a particle at the nth mile -/
noncomputable def speed (n : ℕ) (k : ℝ) (w : ℝ) : ℝ :=
  k / ((n - 1)^2 + 2*w)

/-- Represents the time taken to traverse the nth mile -/
noncomputable def time (n : ℕ) (k : ℝ) (w : ℝ) : ℝ :=
  1 / speed n k w

theorem particle_motion (n : ℕ) (k : ℝ) (w : ℝ) 
  (h1 : w = 12)  -- Weight is 12 pounds
  (h2 : time 2 k w = 2)  -- Second mile takes 2 hours
  (h3 : k = 25/2)  -- Derived from conditions
  : time n k w = 2*((n-1)^2 + 24) / 25 := by
  sorry

#check particle_motion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_motion_l869_86971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inverse_inradius_l869_86942

/-- A triangle with vertices on lattice points -/
structure LatticeTriangle where
  A : ℤ × ℤ
  B : ℤ × ℤ
  C : ℤ × ℤ

/-- The distance between two points -/
noncomputable def distance (p q : ℤ × ℤ) : ℝ :=
  Real.sqrt (((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2) : ℝ)

/-- The perimeter of a triangle -/
noncomputable def perimeter (t : LatticeTriangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- The area of a triangle -/
noncomputable def area (t : LatticeTriangle) : ℝ :=
  let s := perimeter t / 2
  Real.sqrt (s * (s - distance t.A t.B) * (s - distance t.B t.C) * (s - distance t.C t.A))

/-- The inradius of a triangle -/
noncomputable def inradius (t : LatticeTriangle) : ℝ :=
  2 * area t / perimeter t

theorem max_inverse_inradius :
  ∀ t : LatticeTriangle,
    distance t.A t.B = 1 →
    perimeter t < 17 →
    (1 / inradius t) ≤ 1 + 5 * Real.sqrt 2 + Real.sqrt 65 := by
  sorry

#check max_inverse_inradius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inverse_inradius_l869_86942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_point_l869_86993

noncomputable section

-- Define the points A and C
def A : Fin 3 → ℝ := ![-2, 8, 10]
def C : Fin 3 → ℝ := ![2, 4, 8]

-- Define the plane equation
def plane_eq (x : Fin 3 → ℝ) : Prop := x 0 + x 1 + x 2 = 10

-- Define the point B
def B : Fin 3 → ℝ := ![10/13, 60/13, 100/13]

-- Define a function to check if a point lies on the plane
def on_plane (p : Fin 3 → ℝ) : Prop := plane_eq p

-- Define a function to check if three points are collinear
def collinear (p q r : Fin 3 → ℝ) : Prop :=
  ∃ (t : ℝ), (r 0 - p 0) = t * (q 0 - p 0) ∧
             (r 1 - p 1) = t * (q 1 - p 1) ∧
             (r 2 - p 2) = t * (q 2 - p 2)

-- Theorem stating that B satisfies the conditions of the problem
theorem light_reflection_point :
  on_plane B ∧
  ∃ (D : Fin 3 → ℝ), collinear A B D ∧ collinear C B D ∧
    (D 0 + A 0 = 2 * B 0) ∧ (D 1 + A 1 = 2 * B 1) ∧ (D 2 + A 2 = 2 * B 2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_point_l869_86993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l869_86927

/-- Profit function for "Zhongou Jun Tai" -/
noncomputable def S (x : ℝ) : ℝ := x / 5

/-- Profit function for "Yongying Currency" -/
noncomputable def T (x : ℝ) : ℝ := 2 * Real.sqrt x / 5

/-- Total investment amount in ten thousand yuan -/
def total_investment : ℝ := 5

/-- Total profit function -/
noncomputable def total_profit (x : ℝ) : ℝ := S x + T (total_investment - x)

theorem max_profit :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ total_investment ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ total_investment →
  total_profit y ≤ total_profit x ∧
  total_profit x = 1.2 := by
  sorry

#eval total_investment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l869_86927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l869_86932

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 4 - 2 * Real.sin x * Real.cos x - (Real.sin x) ^ 4

theorem f_properties :
  ∀ x : ℝ,
  (0 < x ∧ x < π ∧ f x = -Real.sqrt 2 / 2) →
    (x = 5 * π / 24 ∨ x = 13 * π / 24) ∧
  (0 ≤ x ∧ x ≤ π / 2) →
    (f x ≥ -Real.sqrt 2 ∧ (f x = -Real.sqrt 2 ↔ x = 3 * π / 8)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l869_86932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eval_sqrt_sum_l869_86966

noncomputable def f (x : ℝ) : ℝ := x^6 - 2 * Real.sqrt 2006 * x^5 - x^4 + x^3 - 2 * Real.sqrt 2007 * x^2 + 2 * x - Real.sqrt 2006

theorem f_eval_sqrt_sum : f (Real.sqrt 2006 + Real.sqrt 2007) = Real.sqrt 2007 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eval_sqrt_sum_l869_86966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l869_86905

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = Set.Iic (-4) ∪ Set.Ici 2 :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = Set.Ioi (-3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l869_86905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l869_86940

/-- Given a line with parametric equations x = 1 + (1/2)t and y = 2 + (√3/2)t,
    prove that its inclination angle is 60°. -/
theorem line_inclination_angle :
  ∃ (α : ℝ), α = 60 * Real.pi / 180 ∧
  Real.tan α = (Real.sqrt 3) / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l869_86940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_diff_l869_86978

theorem max_cos_diff (x y : ℝ) (h : Real.sin x - Real.sin y = 3/4) :
  ∃ (max_val : ℝ), max_val = 23/32 ∧ 
  ∀ (z w : ℝ), Real.sin z - Real.sin w = 3/4 → Real.cos (x - y) ≤ max_val :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_diff_l869_86978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_theorem_l869_86970

/-- Definition of a right-angled triangle -/
def RightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ c^2 = a^2 + b^2

/-- Pythagorean theorem for a right-angled triangle -/
theorem pythagorean_theorem (a b c : ℝ) (h : RightTriangle a b c) : a^2 + b^2 = c^2 := by
  rcases h with ⟨ha, hb, hc, heq⟩
  exact heq.symm


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_theorem_l869_86970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_roots_l869_86954

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticRoots where
  x₁ : ℝ
  x₂ : ℝ

/-- 
Given a parabola y = ax^2 + bx + c, if the distance between its intersection points 
with the x-axis is 6 and its axis of symmetry is x = -2, then the solutions to 
ax^2 + bx + c = 0 are x₁ = -5 and x₂ = 1.
-/
theorem parabola_roots (p : Parabola) 
  (h_distance : ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ x₂ - x₁ = 6 ∧ p.a * x₁^2 + p.b * x₁ + p.c = 0 ∧ p.a * x₂^2 + p.b * x₂ + p.c = 0)
  (h_symmetry : -p.b / (2 * p.a) = -2) :
  ∃ (roots : QuadraticRoots), roots.x₁ = -5 ∧ roots.x₂ = 1 ∧
    p.a * roots.x₁^2 + p.b * roots.x₁ + p.c = 0 ∧ 
    p.a * roots.x₂^2 + p.b * roots.x₂ + p.c = 0 ∧ 
    roots.x₁ < roots.x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_roots_l869_86954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_g_of_x_l869_86956

theorem find_g_of_x (x : ℝ) :
  ∃ g : ℝ → ℝ, (7 * x^4 - 4 * x^3 + 2 * x - 5 + g x = 5 * x^3 - 3 * x^2 + 4 * x - 1) ∧
    g x = -7 * x^4 + 9 * x^3 - 3 * x^2 + 2 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_g_of_x_l869_86956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_xi_l869_86933

-- Define the contents of the bags
def bag_A : (ℕ × ℕ) := (3, 2)  -- (red balls, white balls)
def bag_B : (ℕ × ℕ) := (2, 3)  -- (red balls, white balls)

-- Define ξ as a function that returns the number of red balls in bag A after the swap
def ξ (a b : ℕ × ℕ) : ℕ := sorry

-- Define the probability of drawing a red ball from bag A
def prob_red_A : ℚ := 3 / 5

-- Define the probability of drawing a white ball from bag A
def prob_white_A : ℚ := 2 / 5

-- Define the probability of drawing a red ball from bag B
def prob_red_B : ℚ := 2 / 5

-- Define the probability of drawing a white ball from bag B
def prob_white_B : ℚ := 3 / 5

-- Define the expected value of ξ
def E_ξ : ℚ := 14 / 5

-- Theorem statement
theorem expected_value_xi : E_ξ = 14 / 5 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_xi_l869_86933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_48_l869_86951

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (8, -1)
def C : ℝ × ℝ := (12, 7)

-- Function to calculate the area of a triangle given its vertices
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Theorem stating that the area of the triangle with vertices A, B, and C is 48
theorem triangle_area_is_48 : triangleArea A B C = 48 := by
  -- Expand the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp [A, B, C]
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_48_l869_86951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_sign_consistent_vector_l869_86934

theorem exist_sign_consistent_vector (A : Matrix (Fin 3) (Fin 3) ℝ)
  (h_diag : ∀ i, A i i > 0)
  (h_off_diag : ∀ i j, i ≠ j → A i j < 0) :
  ∃ c : Fin 3 → ℝ, (∀ i, c i > 0) ∧
    ((∀ i, (A.mulVec c) i < 0) ∨
     (∀ i, (A.mulVec c) i > 0) ∨
     (∀ i, (A.mulVec c) i = 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_sign_consistent_vector_l869_86934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l869_86981

/-- Given two triangles ABC and AMC where MC intersects AB at point O, 
    prove that if |AM| + |MC| = |AB| + |BC| and |AB| = |BC|, then |OB| > |OM|. -/
theorem triangle_inequality (A B C M O : EuclideanSpace ℝ (Fin 2)) : 
  ‖M - C‖ + ‖A - M‖ = ‖A - B‖ + ‖B - C‖ →
  ‖A - B‖ = ‖B - C‖ →
  ‖O - B‖ > ‖O - M‖ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l869_86981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_A_l869_86925

theorem triangle_sin_A (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧ 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a = 1 ∧ b = Real.sqrt 2 ∧
  2 * B = A + C →
  Real.sin A = Real.sqrt 6 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_A_l869_86925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_decreasing_l869_86921

/-- Inverse proportional function -/
noncomputable def inverse_prop (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_prop_decreasing (k : ℝ) (x₁ x₂ m n : ℝ) 
  (h_k : k > 0) 
  (h_x : x₁ < x₂) 
  (h_m : inverse_prop k x₁ = m) 
  (h_n : inverse_prop k x₂ = n) : 
  m > n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_decreasing_l869_86921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_selecting_A_and_B_l869_86976

/-- The probability of selecting both A and B when randomly choosing 3 students from a group of 5 students (including A and B) -/
theorem probability_selecting_A_and_B :
  let total_students : ℕ := 5
  let selected_students : ℕ := 3
  let prob_A_and_B : ℚ := 3 / 10
  prob_A_and_B = (Nat.choose (total_students - 2) (selected_students - 2)) / (Nat.choose total_students selected_students) := by
  -- Definitions
  let total_students : ℕ := 5
  let selected_students : ℕ := 3
  let prob_A_and_B : ℚ := 3 / 10

  -- Proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_selecting_A_and_B_l869_86976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_focus_coordinates_l869_86963

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The equation of the hyperbola -/
def Hyperbola.equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The right focus of a hyperbola -/
noncomputable def Hyperbola.right_focus (h : Hyperbola) : ℝ × ℝ :=
  (Real.sqrt (h.a^2 + h.b^2), 0)

/-- Theorem: The right focus of the hyperbola (x^2/9) - (y^2/16) = 1 is (5, 0) -/
theorem right_focus_coordinates :
  let h : Hyperbola := { a := 3, b := 4 }
  h.right_focus = (5, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_focus_coordinates_l869_86963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l869_86953

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * Real.pi * x)^2

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 8 * Real.pi^2 * x := by
  -- The proof is omitted for now
  sorry

#check derivative_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l869_86953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_when_a_is_negative_one_monotonic_range_of_a_l869_86991

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (a * x^2 + 3)

-- Part 1
theorem extremum_when_a_is_negative_one :
  let f₁ := f (-1)
  ∃ (min max : ℝ), (∀ x, f₁ x ≥ min) ∧ (∀ x, f₁ x ≤ max) ∧
    (∃ x₁, f₁ x₁ = min) ∧ (∃ x₂, f₁ x₂ = max) ∧
    min = -6 * Real.exp (-3) ∧ max = 2 * Real.exp 1 := by
  sorry

-- Part 2
theorem monotonic_range_of_a :
  {a : ℝ | ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x ≤ y → f a x ≤ f a y} =
  {a : ℝ | a ≥ -3/8} ∪ {a : ℝ | a ≤ -1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_when_a_is_negative_one_monotonic_range_of_a_l869_86991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_l869_86917

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def opposite_direction (a b : V) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ b = k • a

theorem incorrect_statement (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hop : opposite_direction a b) (hnorm : ‖a‖ < ‖b‖) :
  ¬ (∃ (k : ℝ), k > 0 ∧ a + b = k • a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_l869_86917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_range_f_or_g_and_not_both_implies_a_range_l869_86916

open Set Real

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 * Real.exp (3 * a * x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x - a / x + 2 * log x

-- Define the property of f being monotonically increasing on (0,2]
def f_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y ∧ y ≤ 2 → f a x ≤ f a y

-- Define the property of g having an extreme value
def g_has_extreme (a : ℝ) : Prop :=
  ∃ x, x > 0 ∧ ∀ y, y > 0 → g a x ≥ g a y ∨ g a x ≤ g a y

theorem f_monotone_implies_a_range (a : ℝ) :
  f_monotone_increasing a → a ∈ Ici (-1/2) := by sorry

theorem f_or_g_and_not_both_implies_a_range (a : ℝ) :
  (f_monotone_increasing a ∨ g_has_extreme a) ∧
  ¬(f_monotone_increasing a ∧ g_has_extreme a) →
  a ∈ Ioo (-1) (-1/2) ∪ Ici 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_range_f_or_g_and_not_both_implies_a_range_l869_86916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_probability_is_two_thirds_l869_86908

/-- A circular spinner with 6 regions -/
structure Spinner where
  x : ℚ
  region_count : ℕ := 6
  equal_angle_count : ℕ := 4
  fixed_angle1 : ℚ := 20
  fixed_angle2 : ℚ := 140

/-- The sum of all central angles in the spinner equals 360° -/
axiom angle_sum (s : Spinner) : s.fixed_angle1 + s.fixed_angle2 + s.equal_angle_count * s.x = 360

/-- The shaded region consists of the largest angle and two x° angles -/
def shaded_region_sum (s : Spinner) : ℚ := s.fixed_angle2 + 2 * s.x

/-- The probability of the arrow stopping on a shaded region -/
def shaded_probability (s : Spinner) : ℚ := shaded_region_sum s / 360

/-- Theorem: The probability of the arrow stopping on a shaded region is 2/3 -/
theorem shaded_probability_is_two_thirds (s : Spinner) : shaded_probability s = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_probability_is_two_thirds_l869_86908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l869_86995

-- Define a type for the statements
inductive Statement
  | one
  | two
  | three
  | four
  | five

def is_correct (s : Statement) : Bool :=
  match s with
  | Statement.one => false
  | Statement.two => false
  | Statement.three => false
  | Statement.four => true
  | Statement.five => true

def count_correct_statements : Nat :=
  (List.filter is_correct [Statement.one, Statement.two, Statement.three, Statement.four, Statement.five]).length

theorem correct_statements_count : count_correct_statements = 2 := by
  rfl

#eval count_correct_statements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l869_86995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l869_86977

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + Real.cos x

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 π ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 π → f y ≤ f x) ∧
  f x = π - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l869_86977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_l869_86955

noncomputable def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm_v := Real.sqrt (v.1^2 + v.2^2)
  let u := (v.1 / norm_v, v.2 / norm_v)
  ![![u.1^2, u.1 * u.2],
    ![u.1 * u.2, u.2^2]]

theorem det_projection_matrix_zero (v : ℝ × ℝ) (hv : v = (3, 4)) :
  Matrix.det (projection_matrix v) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_l869_86955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_ratio_l869_86950

/-- The ellipse in our problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

/-- The right focus of the ellipse -/
def F : ℝ × ℝ := (2, 0)

/-- A point on the ellipse -/
def point_on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

/-- A chord of the ellipse -/
def chord (A B : ℝ × ℝ) : Prop :=
  point_on_ellipse A ∧ point_on_ellipse B ∧ A ≠ B

/-- A chord passing through the focus F -/
def chord_through_focus (A B : ℝ × ℝ) : Prop :=
  chord A B ∧ ∃ t : ℝ, (1 - t) • A + t • B = F

/-- A chord not perpendicular to the x-axis -/
def not_perpendicular_to_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 ≠ B.1

/-- The perpendicular bisector of a chord -/
def perpendicular_bisector (A B N : ℝ × ℝ) : Prop :=
  N.2 = 0 ∧ (N.1 - A.1) * (B.1 - A.1) + (N.2 - A.2) * (B.2 - A.2) = 0 ∧
  (N.1 - (A.1 + B.1) / 2)^2 + (N.2 - (A.2 + B.2) / 2)^2 = 
  ((A.1 - B.1) / 2)^2 + ((A.2 - B.2) / 2)^2

/-- The ratio of distances -/
noncomputable def distance_ratio (A B N : ℝ × ℝ) : ℝ :=
  Real.sqrt ((N.1 - F.1)^2 + (N.2 - F.2)^2) / 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem ellipse_chord_ratio (A B N : ℝ × ℝ) :
  chord_through_focus A B →
  not_perpendicular_to_x_axis A B →
  perpendicular_bisector A B N →
  distance_ratio A B N = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_ratio_l869_86950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_octahedron_side_length_proof_l869_86959

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with side length 2 -/
structure Cube where
  sideLength : ℝ := 2
  q₁ : Point3D
  q₂ : Point3D
  q₃ : Point3D
  q₄ : Point3D
  q₁' : Point3D
  q₂' : Point3D
  q₃' : Point3D
  q₄' : Point3D

/-- Represents a regular octahedron inscribed in the cube -/
structure Octahedron where
  vertices : Fin 6 → Point3D

/-- The theorem stating the side length of the inscribed octahedron -/
theorem octahedron_side_length (c : Cube) (o : Octahedron) : ℝ :=
  3 * Real.sqrt 2 / 2

/-- Proof of the octahedron side length -/
theorem octahedron_side_length_proof (c : Cube) (o : Octahedron) :
    octahedron_side_length c o = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_octahedron_side_length_proof_l869_86959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_to_line_l869_86985

/-- The line l: 4x + 3y + 5 = 0 -/
def line_l (x y : ℝ) : Prop := 4 * x + 3 * y + 5 = 0

/-- The circle (x-1)² + (y-2)² = 1 -/
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |4 * x + 3 * y + 5| / Real.sqrt (4^2 + 3^2)

theorem min_distance_point_to_line :
  ∀ x y : ℝ, circle_eq x y → 2 ≤ distance_to_line x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_to_line_l869_86985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_of_cubic_polynomial_l869_86923

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluation of a cubic polynomial at a given point -/
noncomputable def CubicPolynomial.eval (p : CubicPolynomial) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The sum of roots of a cubic polynomial -/
noncomputable def CubicPolynomial.sumOfRoots (p : CubicPolynomial) : ℝ := -p.b / p.a

/-- Theorem: If P(x^3 - x) ≥ P(x^2 + x + 1) for all real x, 
    then the sum of roots of P is -b/a -/
theorem sum_of_roots_of_cubic_polynomial 
  (p : CubicPolynomial) 
  (h : ∀ x : ℝ, p.eval (x^3 - x) ≥ p.eval (x^2 + x + 1)) : 
  p.sumOfRoots = -p.b / p.a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_of_cubic_polynomial_l869_86923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_symmetry_l869_86961

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x^2 - 10*x + 25)

theorem f_domain_and_symmetry :
  (∀ x, f x ≠ 0 ↔ x ≠ 5) ∧
  (∀ a : ℝ, f (5 - a) = f (5 + a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_symmetry_l869_86961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_counting_problem_l869_86947

def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_monotonic_increasing (s : List ℕ) : Prop :=
  s.length = 3 ∧ s.Sorted (· < ·)

theorem digit_counting_problem :
  (∃ (S : Finset ℕ), S.card = 52 ∧ 
    (∀ n ∈ S, is_three_digit n ∧ is_even n ∧
      (∃ a b c, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ 
        a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
        n = 100 * a + 10 * b + c))) ∧
  (∃ (T : Finset (List ℕ)), T.card = 20 ∧
    (∀ s ∈ T, is_monotonic_increasing s ∧
      (∀ x ∈ s, x ∈ digits))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_counting_problem_l869_86947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_theorem_l869_86987

/-- Calculates the overall loss percent given initial stock value, profit margin, theft percentages, and tax rate -/
noncomputable def overall_loss_percent (initial_stock : ℝ) (profit_margin : ℝ) (theft1 : ℝ) (theft2 : ℝ) (theft3 : ℝ) (tax_rate : ℝ) : ℝ :=
  let remaining_after_theft1 := initial_stock * (1 - theft1)
  let remaining_after_theft2 := remaining_after_theft1 * (1 - theft2)
  let remaining_after_theft3 := remaining_after_theft2 * (1 - theft3)
  let loss := initial_stock - remaining_after_theft3
  (loss / initial_stock) * 100

/-- Theorem stating that under the given conditions, the overall loss percent is 58% -/
theorem store_loss_theorem :
  ∀ (initial_stock : ℝ),
  initial_stock > 0 →
  overall_loss_percent initial_stock 0.1 0.2 0.3 0.25 0.12 = 58 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_theorem_l869_86987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_properties_l869_86910

/-- A pyramid with a square base and specific lateral face orientations -/
structure SpecialPyramid where
  a : ℝ  -- Side length of the square base
  l : ℝ  -- Median lateral edge length
  a_pos : 0 < a  -- Assumption that a is positive
  l_pos : 0 < l  -- Assumption that l is positive

/-- The volume of the special pyramid -/
noncomputable def volume (p : SpecialPyramid) : ℝ :=
  (1/3) * p.a^2 * (p.l / Real.sqrt 2)

/-- The total surface area of the special pyramid -/
noncomputable def totalSurfaceArea (p : SpecialPyramid) : ℝ :=
  p.a^2 + p.a * Real.sqrt 2 * p.l

/-- Theorem stating the volume and total surface area of the special pyramid -/
theorem special_pyramid_properties (p : SpecialPyramid) :
  (volume p, totalSurfaceArea p) = ((1/3) * p.a^2 * (p.l / Real.sqrt 2), p.a^2 + p.a * Real.sqrt 2 * p.l) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_properties_l869_86910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l869_86949

theorem greatest_integer_fraction : 
  ⌊(3^110 + 2^110 : ℝ) / (3^106 + 2^106 : ℝ)⌋ = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l869_86949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l869_86926

-- Define the function f(x) = 3x + ln x - 5
noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.log x - 5

-- Theorem statement
theorem root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l869_86926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l869_86938

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((1 / (1 - Complex.I)) * Complex.I) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l869_86938
