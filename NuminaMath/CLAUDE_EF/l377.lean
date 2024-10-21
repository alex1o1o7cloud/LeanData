import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l377_37714

/-- Parametric representation of an ellipse --/
noncomputable def ellipse (t : ℝ) : ℝ × ℝ :=
  ((3 * (Real.cos t - 2)) / (3 - Real.sin t), (4 * (Real.sin t - 6)) / (3 - Real.sin t))

/-- Coefficients of the ellipse equation --/
def ellipseCoeffs : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ := (64, 1024, 144, -6144, -1728, 5184)

theorem ellipse_equation (A B C D E F : ℤ) :
  (A = 64 ∧ B = 1024 ∧ C = 144 ∧ D = -6144 ∧ E = -1728 ∧ F = 5184) →
  (∀ t : ℝ, let (x, y) := ellipse t
    A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0) →
  Int.gcd |A| (Int.gcd |B| (Int.gcd |C| (Int.gcd |D| (Int.gcd |E| |F|)))) = 1 →
  |A| + |B| + |C| + |D| + |E| + |F| = 14288 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l377_37714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_average_speed_l377_37798

/-- Represents the flight phases of a rocket --/
structure RocketFlight where
  initial_speed : ℝ
  initial_time : ℝ
  acceleration_start_speed : ℝ
  acceleration_end_speed : ℝ
  acceleration_time : ℝ
  hover_time : ℝ
  plummet_distance : ℝ
  plummet_time : ℝ

/-- Calculates the average speed of a rocket during its entire flight --/
noncomputable def average_speed (flight : RocketFlight) : ℝ :=
  let initial_distance := flight.initial_speed * flight.initial_time
  let acceleration_distance := ((flight.acceleration_start_speed + flight.acceleration_end_speed) / 2) * flight.acceleration_time
  let plummet_distance := flight.plummet_distance
  let total_distance := initial_distance + acceleration_distance + plummet_distance
  let total_time := flight.initial_time + flight.acceleration_time + flight.hover_time + flight.plummet_time
  total_distance / total_time

/-- The theorem stating the average speed of the rocket --/
theorem rocket_average_speed :
  let flight : RocketFlight := {
    initial_speed := 150,
    initial_time := 12,
    acceleration_start_speed := 150,
    acceleration_end_speed := 200,
    acceleration_time := 8,
    hover_time := 4,
    plummet_distance := 600,
    plummet_time := 3
  }
  ∃ ε > 0, |average_speed flight - 140.74| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_average_speed_l377_37798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_focus_coincide_parabola_ellipse_focus_value_l377_37760

/-- The value of p for which the focus of the parabola y² = 2px coincides with 
    the left focus of the ellipse x²/6 + y²/2 = 1 -/
def parabola_ellipse_focus_p : ℝ := -4

/-- The equation of the parabola -/
def is_parabola (x y p : ℝ) : Prop := y^2 = 2*p*x

/-- The equation of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

/-- The x-coordinate of the focus of a parabola y² = 2px -/
noncomputable def parabola_focus_x (p : ℝ) : ℝ := p/2

/-- The coordinates of the left focus of the ellipse x²/6 + y²/2 = 1 -/
def ellipse_left_focus : ℝ × ℝ := (-2, 0)

theorem parabola_ellipse_focus_coincide :
  parabola_focus_x parabola_ellipse_focus_p = (ellipse_left_focus.1) := by
  sorry

theorem parabola_ellipse_focus_value :
  parabola_ellipse_focus_p = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_focus_coincide_parabola_ellipse_focus_value_l377_37760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pages_copied_l377_37765

/-- Given a cost of 5 cents per page, a maximum spending limit of $45, and a total budget of $50,
    prove that the maximum number of pages that can be copied is 900. -/
theorem max_pages_copied (cost_per_page : ℚ) (max_spend : ℚ) (total_budget : ℚ) : 
  cost_per_page = 5 / 100 →
  max_spend = 45 →
  total_budget = 50 →
  ⌊max_spend / cost_per_page⌋ = 900 := by
  intros h_cost h_max h_total
  -- The proof steps would go here
  sorry

#check max_pages_copied

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pages_copied_l377_37765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_a_value_l377_37777

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  hpos_a : 0 < a
  hpos_b : 0 < b
  hecc : e = Real.sqrt 5 / 2

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle structure -/
structure Triangle where
  A : Point
  O : Point
  F : Point

/-- The main theorem -/
theorem hyperbola_a_value (C : Hyperbola) (T : Triangle) :
  (C.a * C.b = 2) →  -- Area of triangle AOF is 1
  (C.a = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_a_value_l377_37777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_301_l377_37710

/-- Factorization of x^4 + 400 -/
noncomputable def factor (x : ℝ) : ℝ := (x^2 - 10*x + 20) * (x^2 + 10*x + 20)

/-- The expression to be computed -/
noncomputable def expression : ℝ :=
  (factor 15 * factor 30 * factor 45 * factor 60 * factor 75) /
  (factor 5 * factor 20 * factor 35 * factor 50 * factor 65)

theorem expression_equals_301 : expression = 301 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_301_l377_37710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_fraction_maximum_l377_37719

theorem trigonometric_fraction_maximum : 
  ∃ (x : ℝ), (Real.sin x)^4 + (Real.cos x)^4 + 2 = (10/9) * ((Real.sin x)^6 + (Real.cos x)^6 + 2) ∧
  ∀ (y : ℝ), (Real.sin y)^4 + (Real.cos y)^4 + 2 ≤ (10/9) * ((Real.sin y)^6 + (Real.cos y)^6 + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_fraction_maximum_l377_37719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equality_l377_37775

theorem log_sum_equality : 
  Real.log 4 + Real.log 9 + 2 * Real.sqrt ((Real.log 6)^2 - Real.log 36 + 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equality_l377_37775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_divisor_of_large_number_l377_37799

/-- The smallest prime divisor of 5^(7^(10^(7^10))) + 1 is 2 -/
theorem smallest_prime_divisor_of_large_number : 
  ∃ p, Nat.Prime p ∧ p ∣ (5^(7^(10^(7^10))) + 1) ∧ p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_divisor_of_large_number_l377_37799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l377_37715

-- Define a triangle with side lengths and angles
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.a^2 - t.a * t.b - 2 * t.b^2 = 0

-- Part 1: If B = π/6, then C = π/3
theorem part_one (t : Triangle) (h : triangle_conditions t) (hB : t.B = π/6) :
  t.C = π/3 := by sorry

-- Part 2: If C = 2π/3 and c = 14, then the area of the triangle is 14√3
theorem part_two (t : Triangle) (h : triangle_conditions t) (hC : t.C = 2*π/3) (hc : t.c = 14) :
  (1/2) * t.a * t.b * Real.sin t.C = 14 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l377_37715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l377_37770

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Point on a hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Intersection points of tangent with asymptotes -/
structure TangentIntersections (h : Hyperbola) (m : PointOnHyperbola h) where
  a : ℝ × ℝ
  b : ℝ × ℝ

/-- Area of a triangle given three points -/
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry

/-- Predicate to check if a point is the circumcenter of a triangle -/
def is_circumcenter (p c1 c2 c3 : ℝ × ℝ) : Prop :=
  sorry

/-- Theorem about hyperbola properties -/
theorem hyperbola_properties (h : Hyperbola) (m : PointOnHyperbola h) 
  (t : TangentIntersections h m) : 
  (∃ (s : ℝ), s = h.a * h.b ∧ 
    s = area_triangle (0, 0) t.a t.b) ∧
  (∃ (p : ℝ × ℝ), 
    is_circumcenter p (0, 0) t.a t.b ∧
    h.a^2 * p.1^2 - h.b^2 * p.2^2 = (h.a^2 + h.b^2)^2 / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l377_37770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_PF_PA_l377_37784

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the focus F
def F : ℝ × ℝ := (0, 2)

-- Define point A
def A : ℝ × ℝ := (0, -2)

-- Define a point P on the parabola
def P : ℝ × ℝ → Prop := λ p => parabola p.1 p.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Statement of the theorem
theorem min_ratio_PF_PA :
  ∃ (min : ℝ), min = Real.sqrt 2 / 2 ∧
  ∀ (p : ℝ × ℝ), P p → (distance p F) / (distance p A) ≥ min :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_PF_PA_l377_37784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_constants_proof_l377_37725

/-- Equilibrium constant for reaction 1: NH₄I(s) ⇌ NH₃(g) + HI(g) -/
noncomputable def K₁ : ℝ := 9

/-- Equilibrium constant for reaction 2: 2HI(g) ⇌ H₂(g) + I₂(g) -/
noncomputable def K₂ : ℝ := 25 / 9

/-- Equilibrium constant for reaction: H₂(g) + I₂(g) ⇌ 2HI(g) -/
noncomputable def K : ℝ := 9 / 25

/-- Equilibrium concentration of H₂ in mol·L⁻¹ -/
noncomputable def c_H₂ : ℝ := 0.5

/-- Equilibrium concentration of HI in mol·L⁻¹ -/
noncomputable def c_HI : ℝ := 4

/-- Equilibrium concentration of I₂ in mol·L⁻¹ -/
noncomputable def c_I₂ : ℝ := c_H₂

/-- Equilibrium concentration of NH₃ in mol·L⁻¹ -/
noncomputable def c_NH₃ : ℝ := c_HI - 2 * c_H₂

theorem equilibrium_constants_proof :
  K₁ = c_NH₃ * c_HI ∧
  K₂ = (c_H₂ * c_I₂) / (c_HI ^ 2) ∧
  K = 1 / K₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_constants_proof_l377_37725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_hyperbola_standard_equation_l377_37766

-- Parabola
def parabola_equation (p : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 = 2 * p * y

def directrix (k : ℝ) : ℝ → ℝ → Prop :=
  fun x y => y = k

theorem parabola_standard_equation :
  ∃ p : ℝ, p > 0 ∧ 
  (∀ x y, parabola_equation p x y ↔ x^2 = 4 * y) ∧
  directrix (-1) 0 (-1) :=
sorry

-- Hyperbola
def ellipse_equation (a b : ℝ) : ℝ → ℝ → Prop :=
  fun x y => y^2 / a^2 + x^2 / b^2 = 1

def hyperbola_equation (a b : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 / a^2 - y^2 / b^2 = 1

def point_on_curve (f : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  f x y

theorem hyperbola_standard_equation :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  (∀ x y, ellipse_equation 36 27 x y → 
    ∃ c : ℝ, c > 0 ∧ (x = 0 → (y = c ∨ y = -c))) ∧
  point_on_curve (hyperbola_equation a b) (Real.sqrt 15) 4 ∧
  (∀ x y, hyperbola_equation a b x y ↔ x^2 / 5 - y^2 / 4 = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_hyperbola_standard_equation_l377_37766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l377_37733

-- Define the lines p and q
def p (x : ℝ) : ℝ := -2 * x + 8
def q (x : ℝ) : ℝ := -3 * x + 8

-- Define the region R as the area between p and q in the 1st quadrant
def R : Set (ℝ × ℝ) :=
  {(x, y) | 0 ≤ x ∧ 0 ≤ y ∧ y ≤ p x ∧ q x ≤ y}

-- Define the region S as the area below p in the 1st quadrant
def S : Set (ℝ × ℝ) :=
  {(x, y) | 0 ≤ x ∧ 0 ≤ y ∧ y ≤ p x}

-- Theorem statement
theorem probability_between_lines :
  ∃ (μ : MeasureTheory.Measure (ℝ × ℝ)), μ R / μ S = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l377_37733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_on_line_l377_37743

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := u.1 * v.1 + u.2 * v.2
  let norm_sq := u.1 * u.1 + u.2 * u.2
  (dot / norm_sq * u.1, dot / norm_sq * u.2)

def line_equation (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + b

theorem vector_on_line (v : ℝ × ℝ) :
  proj (3, -4) v = (9/5, -12/5) →
  line_equation (3/4) (-15/4) v.1 v.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_on_line_l377_37743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_29_equals_170_l377_37796

-- Define the function f
noncomputable def f (y : ℝ) : ℝ :=
  let x := (y - 3) / 2
  (x - 3) * (x + 4)

-- State the theorem
theorem f_29_equals_170 : f 29 = 170 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_29_equals_170_l377_37796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_empty_l377_37794

open Set

theorem complement_intersection_empty (U A B : Set ℕ) : 
  U = {1, 2, 3} → A = {1, 2} → B = {1, 3} → 
  (U \ A) ∩ (U \ B) = ∅ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_empty_l377_37794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l377_37788

-- Define the circle
def our_circle (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the point P
noncomputable def P : ℝ × ℝ := (-3, -3/2)

-- Define a line passing through P
def line_through_P (a b c : ℝ) : Prop :=
  a * P.1 + b * P.2 + c = 0

-- Define the chord length
def chord_length : ℝ := 8

-- Theorem statement
theorem line_equation :
  ∃ (a b c : ℝ),
    line_through_P a b c ∧
    (∀ (x y : ℝ), our_circle x y → (a * x + b * y + c = 0 → chord_length = 8)) ∧
    ((a = 1 ∧ b = 0 ∧ c = 3) ∨ (a = 3 ∧ b = 4 ∧ c = 15)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l377_37788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l377_37726

/-- The distance between two points in a 2D plane. -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: The distance between the points (-3, 5) and (4, -9) is √245. -/
theorem distance_between_specific_points :
  distance (-3) 5 4 (-9) = Real.sqrt 245 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l377_37726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_theorem_l377_37776

-- Define a rectangle
structure Rectangle where
  width : ℕ
  height : ℕ

-- Define a property for odd integers
def isOdd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

-- Define a property for integer side lengths
def hasIntegerSideLengths (r : Rectangle) : Prop :=
  r.width > 0 ∧ r.height > 0

-- Define a property for distances from sides
def distancesFromSides (small : Rectangle) (big : Rectangle) : (ℕ × ℕ × ℕ × ℕ) → Prop
  | (left, right, top, bottom) => 
    left + small.width + right = big.width ∧
    top + small.height + bottom = big.height

-- Define a property for all odd or all even distances
def allOddOrAllEven : (ℕ × ℕ × ℕ × ℕ) → Prop
  | (a, b, c, d) => (isOdd a ∧ isOdd b ∧ isOdd c ∧ isOdd d) ∨
                    (¬isOdd a ∧ ¬isOdd b ∧ ¬isOdd c ∧ ¬isOdd d)

-- The main theorem
theorem rectangle_division_theorem (R : Rectangle) 
  (h1 : isOdd R.width) (h2 : isOdd R.height) 
  (h3 : hasIntegerSideLengths R) :
  ∃ (S : Rectangle), hasIntegerSideLengths S ∧
    ∃ (d : ℕ × ℕ × ℕ × ℕ), distancesFromSides S R d ∧ allOddOrAllEven d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_theorem_l377_37776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l377_37722

-- Define the basic geometric objects
structure Point := (x y : ℝ)
structure Circle := (center : Point) (radius : ℝ)
structure Line := (a b c : ℝ)

-- Define the circles and points
variable (ω ω₁ ω₂ : Circle)
variable (O₁ O₂ D E F A B : Point)
variable (t : Line)

-- Define auxiliary functions
def is_externally_tangent (c1 c2 : Circle) (p : Point) : Prop := sorry
def is_internally_tangent (c1 c2 : Circle) (p : Point) : Prop := sorry
def is_tangent (l : Line) (c : Circle) (p : Point) : Prop := sorry
def is_diameter (c : Circle) (p1 p2 : Point) : Prop := sorry
def is_perpendicular (l1 l2 : Line) : Prop := sorry
def same_side_of_line (l : Line) (p1 p2 p3 : Point) : Prop := sorry
def lies_on (p : Point) (l : Line) : Prop := sorry

-- Define the conditions
axiom externally_tangent : ω₁.center = O₁ ∧ ω₂.center = O₂ ∧ is_externally_tangent ω₁ ω₂ D
axiom internally_tangent : is_internally_tangent ω ω₁ E ∧ is_internally_tangent ω ω₂ F
axiom common_tangent : is_tangent t ω₁ D ∧ is_tangent t ω₂ D
axiom diameter_perpendicular : is_diameter ω A B ∧ is_perpendicular (Line.mk 0 0 0) t
axiom same_side : same_side_of_line t A E O₁

-- Define the lines
noncomputable def line_AO₁ : Line := Line.mk 0 0 0
noncomputable def line_BO₂ : Line := Line.mk 0 0 0
noncomputable def line_EF : Line := Line.mk 0 0 0

-- The theorem to be proved
theorem lines_concurrent : ∃ P : Point, 
  lies_on P line_AO₁ ∧ lies_on P line_BO₂ ∧ lies_on P line_EF ∧ lies_on P t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l377_37722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_ABP_l377_37724

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B P : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (P.2 - A.2) - (P.1 - A.1) * (B.2 - A.2))

/-- The minimum area of triangle ABP given points A and B and P on a circle -/
theorem min_area_triangle_ABP :
  let A : ℝ × ℝ := (0, -3)
  let B : ℝ × ℝ := (4, 0)
  let circle : Set (ℝ × ℝ) := {P | P.1^2 + P.2^2 - 2*P.2 = 0}
  ∃ (min_area : ℝ), min_area = 11/2 ∧
    ∀ P ∈ circle, area_triangle A B P ≥ min_area :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_ABP_l377_37724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_15_l377_37772

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Add this case to cover Nat.zero
  | 1 => 1
  | n + 1 => 2 * sequence_a n + 1

theorem a_4_equals_15 : sequence_a 4 = 15 := by
  -- Unfold the definition of sequence_a
  unfold sequence_a
  -- Evaluate the sequence step by step
  simp
  -- The proof is complete
  rfl

#eval sequence_a 4  -- This will evaluate to 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_15_l377_37772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l377_37786

def spinner1 : Finset ℕ := {1, 4, 6}
def spinner2 : Finset ℕ := {3, 5, 7}

def is_multiple_of_4 (n : ℕ) : Prop := ∃ k, n = 4 * k

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (spinner1.product spinner2).filter (λ p => (p.1 + p.2) % 4 == 0)

theorem spinner_probability :
  (favorable_outcomes.card : ℚ) / (spinner1.card * spinner2.card) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l377_37786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l377_37783

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sin (ω * x) * Real.cos (ω * x)

-- State the theorem
theorem min_omega_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ x₀ : ℝ, ∀ x : ℝ, f ω x₀ ≤ f ω x ∧ f ω x ≤ f ω (x₀ + 2023 * Real.pi)) :
  ω ≥ 1 / 4046 ∧ (∀ ω' > 0, (∃ x₀ : ℝ, ∀ x : ℝ, f ω' x₀ ≤ f ω' x ∧ f ω' x ≤ f ω' (x₀ + 2023 * Real.pi)) → ω' ≥ 1 / 4046) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l377_37783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_is_5000_l377_37729

/-- Represents the financial transaction described in the problem -/
structure FinancialTransaction where
  borrowing_rate : ℚ  -- 4% per annum
  lending_rate : ℚ    -- 5% per annum
  duration : ℕ        -- 2 years
  yearly_gain : ℚ     -- Rs. 50 per year

/-- Calculates the borrowed amount given the financial transaction details -/
def calculate_borrowed_amount (t : FinancialTransaction) : ℚ :=
  t.yearly_gain * t.duration / (t.lending_rate - t.borrowing_rate)

/-- Theorem stating that under the given conditions, the borrowed amount is 5000 -/
theorem borrowed_amount_is_5000 :
  ∀ (t : FinancialTransaction),
    t.borrowing_rate = 4/100 →
    t.lending_rate = 5/100 →
    t.duration = 2 →
    t.yearly_gain = 50 →
    calculate_borrowed_amount t = 5000 := by
  sorry

#eval calculate_borrowed_amount {
  borrowing_rate := 4/100,
  lending_rate := 5/100,
  duration := 2,
  yearly_gain := 50
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_is_5000_l377_37729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_run_pentagon_running_time_pentagon_l377_37749

/-- The time taken to run around a pentagon field -/
theorem time_to_run_pentagon (perimeter : ℝ) (speed_kmh : ℝ) (time_sec : ℝ) : 
  perimeter = 250 → 
  speed_kmh = 12 → 
  time_sec = perimeter / (speed_kmh * 1000 / 3600) → 
  time_sec = 75 := by
  sorry

/-- Function to calculate time given distance and speed -/
noncomputable def calculate_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- The perimeter of the pentagon field -/
def pentagon_perimeter : ℝ := 30 + 40 + 50 + 60 + 70

/-- Convert km/h to m/s -/
noncomputable def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

/-- Main theorem about running time around the pentagon field -/
theorem running_time_pentagon (speed_kmh : ℝ) (time_sec : ℝ) :
  speed_kmh = 12 →
  time_sec = calculate_time pentagon_perimeter (kmh_to_ms speed_kmh) →
  time_sec = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_run_pentagon_running_time_pentagon_l377_37749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l377_37778

noncomputable section

-- Define the points
def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (8, 0)

-- Define D as the midpoint of AB
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define E as the midpoint of BC
noncomputable def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the lines AE and CD
noncomputable def lineAE (x : ℝ) : ℝ := (E.2 - A.2) / (E.1 - A.1) * (x - A.1) + A.2
noncomputable def lineCD (x : ℝ) : ℝ := (D.2 - C.2) / (D.1 - C.1) * (x - C.1) + C.2

-- Define F as the intersection point of AE and CD
noncomputable def F : ℝ × ℝ :=
  let x := (lineCD 0 - lineAE 0) / ((E.2 - A.2) / (E.1 - A.1) - (D.2 - C.2) / (D.1 - C.1))
  (x, lineAE x)

theorem intersection_point_sum :
  F.1 + F.2 = 14/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l377_37778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_e_equals_two_l377_37763

theorem floor_e_equals_two : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_e_equals_two_l377_37763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_9800_minus_53_cube_root_l377_37755

theorem sqrt_9800_minus_53_cube_root (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt 9800 - 53 : ℝ) = (Real.sqrt (a : ℝ) - (b : ℝ))^3 →
  a + b = 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_9800_minus_53_cube_root_l377_37755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l377_37731

theorem calculation_proof : (1/3)⁻¹ + Real.sqrt 12 - |Real.sqrt 3 - 2| - (Real.pi - 2023)^0 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l377_37731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_opposite_five_people_l377_37789

def num_people : ℕ := 5

def num_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

def num_favorable_outcomes (n : ℕ) : ℕ := 2 * Nat.factorial (n - 2)

def probability_opposite (n : ℕ) : ℚ :=
  (num_favorable_outcomes n : ℚ) / (num_arrangements n : ℚ)

theorem probability_opposite_five_people :
  probability_opposite num_people = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_opposite_five_people_l377_37789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l377_37720

theorem expression_value : ((((3:ℚ)+2)⁻¹+1)⁻¹+2)⁻¹+2 = 40/17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l377_37720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_estimation_l377_37741

/-- The number of turtles tagged on June 1 -/
def tagged_june : ℕ := 50

/-- The number of turtles captured on October 1 -/
def captured_october : ℕ := 80

/-- The number of tagged turtles found in the October 1 capture -/
def tagged_october : ℕ := 4

/-- The proportion of turtles that left the pond between June 1 and October 1 -/
def migration_rate : ℚ := 1/5

/-- The proportion of turtles in October that were not in the pond on June 1 -/
def new_turtles_rate : ℚ := 3/10

/-- The estimated number of turtles in the pond on June 1 -/
def estimated_turtles_june : ℕ := 700

theorem turtle_estimation :
  let october_original := captured_october - Int.floor (new_turtles_rate * captured_october)
  let tagged_ratio : ℚ := tagged_october / october_original
  tagged_ratio = tagged_june / estimated_turtles_june :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_estimation_l377_37741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_13_divisors_and_12_factor_l377_37735

def has_exactly_k_divisors (n k : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = k

theorem smallest_number_with_13_divisors_and_12_factor : ∃ n : ℕ,
  n > 0 ∧ 12 ∣ n ∧ has_exactly_k_divisors n 13 ∧
  ∀ m < n, 12 ∣ m → ¬(has_exactly_k_divisors m 13) ∧
  n = 156 := by
  sorry

#check smallest_number_with_13_divisors_and_12_factor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_13_divisors_and_12_factor_l377_37735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l377_37727

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

-- State the theorem
theorem triangle_proof (abc : Triangle) 
  (h1 : abc.b * Real.sin (2 * abc.A) = Real.sqrt 3 * abc.a * Real.sin abc.B)
  (h2 : abc.b / abc.c = 3 * Real.sqrt 3 / 4)
  (h3 : abc.area = 3 * Real.sqrt 3) :
  abc.A = π / 6 ∧ abc.a = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l377_37727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l377_37704

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6) + 1

theorem function_properties (ω : ℝ) (h1 : ω > 0) (h2 : Real.pi / 2 = Real.pi / ω) :
  ω = 2 ∧ ∃ (M : ℝ), M = 3 ∧ ∀ x ∈ Set.Icc 0 (Real.pi / 2), f 2 x ≤ M := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l377_37704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l377_37797

/-- Predicate to check if a point is on the directrix -/
def is_on_directrix (point : ℝ × ℝ) : Prop := point.1 = 3

/-- Predicate to check if a point is the vertex of the parabola -/
def is_vertex (point : ℝ × ℝ) : Prop := point.1 = 0 ∧ point.2 = 0

/-- Predicate to check if a point is on the parabola -/
def is_on_parabola (p : ℝ) (point : ℝ × ℝ) : Prop := point.2^2 = -4*p*point.1

/-- A parabola with vertex at the origin and directrix x = 3 has the equation y^2 = -12x -/
theorem parabola_equation : 
  (∃ p : ℝ, p > 0 ∧ 
   (∀ point : ℝ × ℝ, is_on_directrix point → point.1 = 3) ∧
   (∀ point : ℝ × ℝ, is_vertex point → point.1 = 0 ∧ point.2 = 0) ∧
   (∀ point : ℝ × ℝ, is_on_parabola p point ↔ point.2^2 = -4*p*point.1)) →
  (∀ point : ℝ × ℝ, is_on_parabola 3 point ↔ point.2^2 = -12*point.1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l377_37797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_focus_l377_37795

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Theorem: Distance between point on parabola and its focus -/
theorem distance_point_focus (para : Parabola) 
  (h1 : Point.mk 2 2 ∈ {p : Point | p.y^2 = 2 * para.p * p.x}) : 
  distance (Point.mk 2 2) (Point.mk (para.p / 2) 0) = 5/2 := by
  sorry

#check distance_point_focus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_focus_l377_37795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_and_decreasing_functions_l377_37703

-- Define the interval (0,1)
def openInterval01 : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

-- Define evenness
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define decreasing on an interval
def isDecreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x > f y

-- Define the functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := -x^2
noncomputable def h (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)
noncomputable def k (x : ℝ) : ℝ := Real.cos x

theorem even_and_decreasing_functions :
  (isEven f ∧ isDecreasing f openInterval01) = false ∧
  (isEven g ∧ isDecreasing g openInterval01) = true ∧
  (isEven h ∧ isDecreasing h openInterval01) = false ∧
  (isEven k ∧ isDecreasing k openInterval01) = true :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_and_decreasing_functions_l377_37703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_inclination_l377_37737

/-- The inclination angle of a line passing through two points -/
noncomputable def inclination_angle (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.arctan ((y2 - y1) / (x2 - x1))

/-- Theorem: The inclination angle of a line passing through (1, 0) and (0, -1) is π/4 -/
theorem line_through_points_inclination :
  inclination_angle 1 0 0 (-1) = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_inclination_l377_37737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l377_37742

theorem negation_of_sin_leq_one :
  (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l377_37742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_relations_l377_37767

-- Define the sound pressure level formula
noncomputable def sound_pressure_level (p p₀ : ℝ) : ℝ := 20 * Real.log (p / p₀) / Real.log 10

-- Define the theorem
theorem sound_pressure_relations (p₀ p₁ p₂ p₃ : ℝ) 
  (h₀ : p₀ > 0)
  (h₁ : 60 ≤ sound_pressure_level p₁ p₀ ∧ sound_pressure_level p₁ p₀ ≤ 90)
  (h₂ : 50 ≤ sound_pressure_level p₂ p₀ ∧ sound_pressure_level p₂ p₀ ≤ 60)
  (h₃ : sound_pressure_level p₃ p₀ = 40) :
  (∃ (p₁' p₂' : ℝ), p₁' ≥ p₂' ∧ 
    60 ≤ sound_pressure_level p₁' p₀ ∧ sound_pressure_level p₁' p₀ ≤ 90 ∧
    50 ≤ sound_pressure_level p₂' p₀ ∧ sound_pressure_level p₂' p₀ ≤ 60) ∧
  p₃ = 100 * p₀ ∧
  p₁ ≤ 100 * p₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_relations_l377_37767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l377_37781

/-- Given a non-right-angled triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where c = 1, C = π/3, and sin C + sin(A-B) = 3sin(2B), the area of triangle ABC is (3√3)/28. -/
theorem triangle_area (A B C a b c : Real) : 
  c = 1 →
  C = π/3 →
  Real.sin C + Real.sin (A - B) = 3 * Real.sin (2 * B) →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 28 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l377_37781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_lines_l377_37752

theorem max_distance_between_lines (a b c : ℝ) :
  (∀ x, x^2 + x + c = 0 ↔ (x = a ∨ x = b)) →
  0 ≤ c →
  c ≤ 1/8 →
  let d := |a - b| / Real.sqrt 2
  ∃ (d_max : ℝ), d ≤ d_max ∧ d_max = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_lines_l377_37752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_malvina_card_sum_l377_37732

theorem malvina_card_sum (x : Real) : 
  0 < x → x < π / 2 → 
  ((Real.sin x = Real.cos x ∧ Real.tan x ≠ Real.sin x) ∨ 
   (Real.cos x = Real.tan x ∧ Real.sin x ≠ Real.cos x) ∨ 
   (Real.tan x = Real.sin x ∧ Real.cos x ≠ Real.sin x)) → 
  (if Real.sin x = Real.cos x then Real.tan x
   else if Real.cos x = Real.tan x then Real.sin x
   else Real.cos x) + 
  (if Real.cos x = Real.tan x then Real.sin x else 1) = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_malvina_card_sum_l377_37732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fraction_increase_l377_37790

theorem unique_fraction_increase : ∃! (x y : ℕ+), 
  (Nat.gcd x.val y.val = 1) ∧ 
  ((x.val + 1 : ℚ) / (y.val + 1) = 1.2 * (x.val : ℚ) / y.val) ∧
  (x.val = 5 ∧ y.val = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fraction_increase_l377_37790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_proof_l377_37705

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x + 1/a|

-- Part I
theorem solution_set (x : ℝ) : 
  (f 2 x > 3) ↔ (x < -11/4 ∨ x > 1/4) := by sorry

-- Part II
theorem inequality_proof (a m : ℝ) (ha : a > 0) :
  f a m + f a (-1/m) ≥ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_proof_l377_37705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robert_time_theorem_l377_37793

/-- The time taken for Robert to cover the one-mile stretch -/
noncomputable def robert_time (highway_length : ℝ) (highway_width : ℝ) (speed : ℝ) (break_duration : ℝ) (break_interval : ℝ) : ℝ :=
  let radius := highway_width / 2
  let num_semicircles := highway_length * 5280 / highway_width
  let distance := num_semicircles * Real.pi * radius
  let time_without_breaks := distance / (speed * 5280)
  let num_breaks := highway_length / break_interval
  let total_break_time := num_breaks * break_duration / 60
  time_without_breaks + total_break_time

/-- Theorem stating that Robert's time to cover the stretch is 6π + 5 minutes -/
theorem robert_time_theorem :
  robert_time 1 30 5 5 0.5 = 6 * Real.pi + 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robert_time_theorem_l377_37793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_and_length_l377_37792

-- Define the circle
noncomputable def circle_center : ℝ × ℝ := (1, 2)
noncomputable def circle_radius : ℝ := Real.sqrt 2

-- Define point P
def point_P : ℝ × ℝ := (2, -1)

-- Define the equations of the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := x + y - 1 = 0
def tangent_line_2 (x y : ℝ) : Prop := 7 * x - y - 15 = 0

-- Theorem statement
theorem tangent_lines_and_length :
  -- The equations of the tangent lines are correct
  (∃ (x y : ℝ), tangent_line_1 x y ∧ (x - 1)^2 + (y - 2)^2 = 2 ∧ (x, y) ≠ point_P) ∧
  (∃ (x y : ℝ), tangent_line_2 x y ∧ (x - 1)^2 + (y - 2)^2 = 2 ∧ (x, y) ≠ point_P) ∧
  -- The length of the tangent line is 2√2
  (let d := Real.sqrt ((point_P.1 - circle_center.1)^2 + (point_P.2 - circle_center.2)^2);
   Real.sqrt (d^2 - circle_radius^2) = 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_and_length_l377_37792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l377_37782

/-- The distance between two parallel lines ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- The first line: x + y - 1 = 0 -/
def line1 : ℝ → ℝ → Prop :=
  λ x y ↦ x + y - 1 = 0

/-- The second line: x + y + 1 = 0 -/
def line2 : ℝ → ℝ → Prop :=
  λ x y ↦ x + y + 1 = 0

theorem distance_between_lines :
  distance_parallel_lines 1 1 (-1) 1 = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l377_37782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l377_37762

/-- A function f(x) with a phase shift φ. -/
noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x) * Real.cos φ + Real.cos (2 * x) * Real.sin φ

/-- The condition that |φ| < π/2. -/
def phi_condition (φ : ℝ) : Prop := |φ| < Real.pi / 2

/-- The condition that the graph is shifted right by π/3 and symmetric about the origin. -/
def graph_condition (φ : ℝ) : Prop := ∃ (k : ℤ), φ = Real.pi * (↑k : ℝ) - Real.pi / 3

/-- The theorem stating the minimum value of f(x) on [0, π/2]. -/
theorem min_value_of_f (φ : ℝ) (h1 : phi_condition φ) (h2 : graph_condition φ) :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
    f x φ = -Real.sqrt 3 / 2 ∧
    ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y φ ≥ -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l377_37762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l377_37700

def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N : Set ℝ := {x | (2 : ℝ)^(x+1) ≥ 1}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l377_37700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soil_volume_correct_l377_37764

/-- Represents a symmetrical railway embankment with a perpendicular cut -/
structure Embankment where
  a : ℝ  -- bottom width of embankment
  b : ℝ  -- top width of embankment
  m : ℝ  -- height of embankment
  c : ℝ  -- bottom width of cut
  d : ℝ  -- top width of cut
  h_positive : 0 < m  -- height is positive
  h_widths : 0 < c ∧ c ≤ d ∧ d ≤ b ∧ b ≤ a  -- width constraints

/-- Volume of soil to be removed from the embankment -/
noncomputable def soil_volume (e : Embankment) : ℝ :=
  (e.m / 6) * (2 * e.a * e.c + 2 * e.b * e.d + e.a * e.d + e.b * e.c)

/-- Theorem stating that the calculated volume is correct -/
theorem soil_volume_correct (e : Embankment) : 
  soil_volume e = (e.m / 6) * (2 * e.a * e.c + 2 * e.b * e.d + e.a * e.d + e.b * e.c) := by
  -- Unfold the definition of soil_volume
  unfold soil_volume
  -- The equality follows directly from the definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soil_volume_correct_l377_37764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_opposite_parts_l377_37747

theorem complex_number_opposite_parts (a : ℝ) : 
  (Complex.re ((↑a * Complex.I + 2) * Complex.I) = 
   -Complex.im ((↑a * Complex.I + 2) * Complex.I)) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_opposite_parts_l377_37747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tina_pail_capacity_l377_37706

/-- Represents the capacity of Tina's pail in gallons -/
def tina_pail : ℝ := sorry

/-- Represents the capacity of Tommy's pail in gallons -/
def tommy_pail : ℝ := tina_pail + 2

/-- Represents the capacity of Timmy's pail in gallons -/
def timmy_pail : ℝ := 2 * tommy_pail

/-- The total amount of water in gallons after 3 trips each -/
def total_water : ℝ := 66

theorem tina_pail_capacity :
  3 * tina_pail + 3 * tommy_pail + 3 * timmy_pail = total_water →
  tina_pail = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tina_pail_capacity_l377_37706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_ratio_l377_37761

theorem journey_distance_ratio (total_distance : ℝ) (total_time : ℝ) 
  (speed1 : ℝ) (speed2 : ℝ) (distance1 : ℝ) (distance2 : ℝ) :
  total_distance = 448 →
  total_time = 20 →
  speed1 = 21 →
  speed2 = 24 →
  distance1 + distance2 = total_distance →
  distance1 / speed1 + distance2 / speed2 = total_time →
  distance1 / distance2 = 1 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

#check journey_distance_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_ratio_l377_37761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sign_sqrt_sum_squared_l377_37713

theorem opposite_sign_sqrt_sum_squared (x y : ℝ) :
  (Real.sqrt (x - 1) + |2*x + y - 6| = 0) → 
  Real.sqrt ((x + y)^2) = 5 ∨ Real.sqrt ((x + y)^2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sign_sqrt_sum_squared_l377_37713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_time_l377_37791

/-- Represents the properties of a car's travel. -/
structure CarTravel where
  speed : ℝ
  fuelEfficiency : ℝ
  tankCapacity : ℝ
  fuelUsedRatio : ℝ

/-- Calculates the travel time of a car given its properties. -/
noncomputable def travelTime (car : CarTravel) : ℝ :=
  (car.tankCapacity * car.fuelUsedRatio * car.fuelEfficiency) / car.speed

/-- Theorem stating that under the given conditions, the car travels for 5 hours. -/
theorem car_travel_time :
  let car : CarTravel := {
    speed := 60,
    fuelEfficiency := 30,
    tankCapacity := 12,
    fuelUsedRatio := 0.8333333333333334
  }
  travelTime car = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_time_l377_37791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l377_37701

theorem sin_graph_shift (x : ℝ) :
  Real.sin (2 * (x - π / 6)) = Real.sin (2 * x - π / 3) := by
  have h1 : 2 * (x - π / 6) = 2 * x - π / 3 := by
    ring
  rw [h1]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l377_37701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l377_37739

noncomputable def Hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 16 = 1

def center : ℝ × ℝ := (0, 0)
noncomputable def right_focus : ℝ × ℝ := (2 * Real.sqrt 5, 0)
noncomputable def eccentricity : ℝ := Real.sqrt 5

def fixed_line (x : ℝ) : Prop := x = -1

def left_vertex : ℝ × ℝ := (-2, 0)
def right_vertex : ℝ × ℝ := (2, 0)

theorem hyperbola_properties :
  ∀ (P M N : ℝ × ℝ),
    fixed_line P.1 →
    Hyperbola M.1 M.2 →
    Hyperbola N.1 N.2 →
    (∃ (t : ℝ), M.2 = (M.1 + 2) * (P.2 / (P.1 + 2))) →
    (∃ (t : ℝ), N.2 = (N.1 - 2) * (P.2 / (P.1 - 2))) →
    (∃ (m t : ℝ), N.2 - M.2 = m * (N.1 - M.1) ∧ N.2 = m * N.1 + t ∧ M.2 = m * M.1 + t) →
    N.2 - M.2 = -4 * (N.1 - M.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l377_37739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_archery_score_l377_37774

-- Define the points
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (2.5, 1)
def C : ℝ × ℝ := (-1, 4)
def D : ℝ × ℝ := (-4, -4)
def E : ℝ × ℝ := (6, 5)

-- Define the scoring function
noncomputable def score (p : ℝ × ℝ) : ℕ :=
  let d := Real.sqrt (p.1^2 + p.2^2)
  if d ≤ Real.sqrt 2 then 300
  else if d ≤ 3 then 100
  else if d ≤ 5 then 50
  else 0

-- Theorem statement
theorem archery_score : 
  (score A + score B + score C + score D + score E = 500) ∧ 
  (List.filter (fun p => Real.sqrt (p.1^2 + p.2^2) < Real.sqrt 2) [A, B, C, D, E]).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_archery_score_l377_37774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_our_series_sum_abs_sin_fourier_series_zero_our_series_sum_via_fourier_l377_37757

/-- The Fourier series expansion of |sin x| on [-π, π] --/
noncomputable def abs_sin_fourier_series (x : ℝ) : ℝ :=
  2 / Real.pi - (4 / Real.pi) * ∑' k, ((-1)^k * Real.cos (k * x)) / ((2 * k - 1) * (2 * k + 1))

/-- The series we want to sum --/
noncomputable def our_series : ℝ := ∑' k, 1 / ((2 * k - 1) * (2 * k + 1))

/-- Theorem stating that our series sums to 1/2 --/
theorem our_series_sum : our_series = 1 / 2 := by
  sorry

/-- Theorem relating our series to the Fourier series of |sin x| --/
theorem abs_sin_fourier_series_zero (x : ℝ) : abs_sin_fourier_series x = 0 := by
  sorry

/-- Main theorem proving that our series sums to 1/2 using the Fourier series --/
theorem our_series_sum_via_fourier : our_series = 1 / 2 := by
  have h : abs_sin_fourier_series 0 = 0 := abs_sin_fourier_series_zero 0
  -- Expand the definition and use the fact that cos(0) = 1
  have h2 : 2 / Real.pi - (4 / Real.pi) * our_series = 0 := by sorry
  -- Solve for our_series
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_our_series_sum_abs_sin_fourier_series_zero_our_series_sum_via_fourier_l377_37757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_kmn_equals_31_l377_37746

theorem sum_kmn_equals_31 
  (t : ℝ) 
  (k m n : ℕ+) 
  (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9/4)
  (h2 : (1 - Real.sin t) * (1 - Real.cos t) = m/n - Real.sqrt k)
  (h3 : Nat.Coprime m.val n.val) :
  k + m + n = 31 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_kmn_equals_31_l377_37746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l377_37751

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 2) + 1 / (x - 3)

def IsValidInput (f : ℝ → ℝ) (x : ℝ) : Prop := ∃ y, f x = y

theorem domain_of_f :
  {x : ℝ | IsValidInput f x} = {x : ℝ | x ≥ 2 ∧ x ≠ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l377_37751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_intersection_l377_37734

/-- The circle equation --/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The line equation --/
def line_eq (x y k : ℝ) : Prop := y = k * x - 3

/-- Theorem: k ≤ -2√2 is a sufficient condition for the circle and line to have common points --/
theorem sufficient_condition_for_intersection (k : ℝ) : 
  k ≤ -2 * Real.sqrt 2 → ∃ x y : ℝ, circle_eq x y ∧ line_eq x y k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_intersection_l377_37734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l377_37738

theorem trig_problem (α : Real) 
  (h1 : Real.sin α = 4/5) 
  (h2 : α > π/2 ∧ α < π) : 
  Real.sin (α - π/4) = 7*Real.sqrt 2/10 ∧ 
  Real.tan (2*α) = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l377_37738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_average_speed_l377_37753

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem cyclist_average_speed :
  let distance : ℝ := 120  -- miles
  let time : ℝ := 11/3     -- hours (3 hours and 40 minutes = 11/3 hours)
  average_speed distance time = 360/11 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry

#eval (360 : ℚ) / 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_average_speed_l377_37753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_factor_of_36_l377_37769

/-- The probability that a positive integer less than or equal to 36 is a factor of 36 -/
theorem prob_factor_of_36 : 
  (Finset.filter (λ n ↦ n ∣ 36) (Finset.range 37)).card / 36 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_factor_of_36_l377_37769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_product_216_prob_three_sixes_l377_37748

-- Define a standard die
def standard_die : Finset ℕ := Finset.range 6

-- Define the probability of an event on a standard die
def prob_event (event : Finset ℕ) : ℚ :=
  event.card / standard_die.card

-- Define the condition for the product to be 216
def product_216 (a b c : ℕ) : Prop :=
  a * b * c = 216

-- State the theorem
theorem prob_product_216 :
  prob_event {6} = 1 / 216 := by
  sorry

-- Additional theorem to connect the problem statement with the solution
theorem prob_three_sixes :
  (1 / 6 : ℚ) * (1 / 6 : ℚ) * (1 / 6 : ℚ) = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_product_216_prob_three_sixes_l377_37748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_plus_N_equals_five_l377_37716

-- Define the operation rules
noncomputable def oplus (a b : ℝ) : ℝ := a^(1/2) + b^(1/3)
noncomputable def otimes (a b : ℝ) : ℝ := Real.log (a^2) - Real.log (b^(1/2))

-- Define M and N
noncomputable def M : ℝ := oplus (9/4) (8/125)
noncomputable def N : ℝ := otimes (Real.sqrt 2) (1/25)

-- Theorem statement
theorem M_plus_N_equals_five : M + N = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_plus_N_equals_five_l377_37716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_weight_after_five_years_fish_weight_starts_decreasing_after_five_years_l377_37758

/-- Fish weight growth rate in the first year -/
noncomputable def initial_growth_rate : ℝ := 2

/-- Growth rate multiplier for subsequent years -/
noncomputable def growth_rate_multiplier : ℝ := 1/2

/-- Annual weight decrease rate due to pollution -/
noncomputable def pollution_decrease_rate : ℝ := 1/10

/-- Fish weight after n years relative to initial weight -/
noncomputable def fish_weight : ℕ → ℝ
| 0 => 1
| n+1 => (fish_weight n) * (1 + (initial_growth_rate * (growth_rate_multiplier ^ n))) * (1 - pollution_decrease_rate)

theorem fish_weight_after_five_years :
  fish_weight 5 = 405/32 := by
  sorry

theorem fish_weight_starts_decreasing_after_five_years :
  ∀ n : ℕ, n < 5 → fish_weight (n+1) > fish_weight n ∧
  fish_weight 6 < fish_weight 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_weight_after_five_years_fish_weight_starts_decreasing_after_five_years_l377_37758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_formula_l377_37721

-- Define the pyramid
structure Pyramid where
  a : ℝ  -- Length of AB
  b : ℝ  -- Length of AD
  h : ℝ  -- Height of the pyramid (SC)
  ha : a > 0
  hb : b > 0
  hh : h > 0

-- Define the dihedral angle
noncomputable def dihedralAngle (p : Pyramid) : ℝ :=
  Real.pi - Real.arcsin ((p.h * Real.sqrt (p.a^2 + p.b^2 + p.h^2)) /
    (Real.sqrt (p.a^2 + p.h^2) * Real.sqrt (p.b^2 + p.h^2)))

-- Theorem statement
theorem dihedral_angle_formula (p : Pyramid) :
  dihedralAngle p = Real.pi - Real.arcsin ((p.h * Real.sqrt (p.a^2 + p.b^2 + p.h^2)) /
    (Real.sqrt (p.a^2 + p.h^2) * Real.sqrt (p.b^2 + p.h^2))) :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_formula_l377_37721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f4_is_even_and_decreasing_l377_37744

-- Define the interval [0,+∞)
def nonnegative_reals : Set ℝ := {x : ℝ | x ≥ 0}

-- Define the functions
def f1 : ℝ → ℝ := λ x => x
noncomputable def f2 : ℝ → ℝ := λ x => 2^x
def f3 : ℝ → ℝ := λ x => x^2
def f4 : ℝ → ℝ := λ x => -x^2

-- Define even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define decreasing function on an interval
def is_decreasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f y < f x

theorem only_f4_is_even_and_decreasing :
  (is_even f4 ∧ is_decreasing_on f4 nonnegative_reals) ∧
  (¬(is_even f1 ∧ is_decreasing_on f1 nonnegative_reals)) ∧
  (¬(is_even f2 ∧ is_decreasing_on f2 nonnegative_reals)) ∧
  (¬(is_even f3 ∧ is_decreasing_on f3 nonnegative_reals)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f4_is_even_and_decreasing_l377_37744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expenses_opposite_of_income_l377_37785

/-- Represents the monetary value in yuan -/
structure Yuan where
  value : Int

/-- Represents income in yuan -/
def income (amount : Int) : Yuan :=
  ⟨amount⟩

/-- Represents expenses in yuan -/
def expenses (amount : Int) : Yuan :=
  ⟨-amount⟩

/-- Theorem stating that if income of 5 yuan is +5, then expenses of 5 yuan is -5 -/
theorem expenses_opposite_of_income :
  income 5 = ⟨5⟩ → expenses 5 = ⟨-5⟩ :=
by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expenses_opposite_of_income_l377_37785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_surface_area_l377_37759

/-- The surface area of a regular tetrahedron with edge length a -/
noncomputable def tetrahedron_surface_area (a : ℝ) : ℝ :=
  Real.sqrt 3 * a^2

/-- Theorem: The surface area of a regular tetrahedron with edge length a is √3 * a^2 -/
theorem regular_tetrahedron_surface_area (a : ℝ) (h : a > 0) :
  tetrahedron_surface_area a = Real.sqrt 3 * a^2 := by
  -- Unfold the definition of tetrahedron_surface_area
  unfold tetrahedron_surface_area
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_surface_area_l377_37759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swim_time_proof_l377_37728

/-- Represents the time taken to swim against a current -/
noncomputable def swim_against_current (swim_speed still_water_speed current_speed : ℝ) (distance : ℝ) : ℝ :=
  distance / (swim_speed - current_speed)

/-- Proves that the time taken to swim against the current is equal to the given time -/
theorem swim_time_proof (swim_speed still_water_speed current_speed time distance : ℝ) 
  (h1 : swim_speed = 4)
  (h2 : current_speed = 2)
  (h3 : time = 8)
  (h4 : swim_against_current swim_speed still_water_speed current_speed distance = time) :
  swim_against_current swim_speed still_water_speed current_speed distance = 8 := by
  sorry

#check swim_time_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swim_time_proof_l377_37728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_amount_to_give_l377_37730

/-- The amount LeRoy must give to Bernardo to equalize shared costs -/
noncomputable def amount_to_give (L B C X : ℝ) : ℝ := (B + C - X - 2*L) / 3

/-- Theorem stating that the amount LeRoy must give to Bernardo is correct -/
theorem correct_amount_to_give (L B C X : ℝ) :
  let total_shared := L + B + C - X
  let equal_share := total_shared / 3
  amount_to_give L B C X = equal_share - L := by
  sorry

#check correct_amount_to_give

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_amount_to_give_l377_37730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_stack_height_five_layers_l377_37717

/-- The height of a pile of cylindrical pipes packed in a triangular pattern -/
noncomputable def pipeStackHeight (diameter : ℝ) (layers : ℕ) : ℝ :=
  let triangleHeight := diameter * Real.sqrt 3 / 2
  let completeTriangles := (layers - 1) / 2
  completeTriangles * 3 * triangleHeight

theorem pipe_stack_height_five_layers (diameter : ℝ) (h : diameter = 12) :
  pipeStackHeight diameter 5 = 18 * Real.sqrt 3 := by
  sorry

#check pipe_stack_height_five_layers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_stack_height_five_layers_l377_37717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_ngon_uniqueness_l377_37711

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Rotate two points around their midpoint -/
def rotateAroundMidpoint (a b : Point2D) (angle : ℝ) : (Point2D × Point2D) :=
  sorry

/-- Check if n points form a regular n-gon -/
def isRegularNGon (points : List Point2D) : Prop :=
  sorry

/-- The center of a regular n-gon -/
noncomputable def centerOfNGon (points : List Point2D) : Point2D :=
  sorry

/-- The size (radius) of a regular n-gon -/
noncomputable def sizeOfNGon (points : List Point2D) : ℝ :=
  sorry

/-- A sequence of rotations -/
def RotationSequence := List (Nat × Nat × ℝ)

/-- Apply a sequence of rotations to a list of points -/
def applyRotations (points : List Point2D) (rotations : RotationSequence) : List Point2D :=
  sorry

theorem regular_ngon_uniqueness
  (n : Nat)
  (initialPoints : List Point2D)
  (rotations : RotationSequence)
  (h1 : n ≥ 3)
  (h2 : initialPoints.length = n)
  (h3 : ∃ finalPoints, 
    (finalPoints.length = n) ∧ 
    (isRegularNGon finalPoints) ∧ 
    (finalPoints = applyRotations initialPoints rotations)) :
  ∀ p q : List Point2D,
    (p.length = n) → 
    (q.length = n) → 
    (isRegularNGon p) → 
    (isRegularNGon q) → 
    (p = applyRotations initialPoints rotations) →
    (q = applyRotations initialPoints rotations) →
    (centerOfNGon p = centerOfNGon q ∧ sizeOfNGon p = sizeOfNGon q) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_ngon_uniqueness_l377_37711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l377_37718

-- Define sets A and B
def A : Set ℝ := {x | (2 : ℝ)^x > 1}
def B : Set ℝ := {x | x^2 - x - 2 < 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l377_37718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l377_37736

/-- For a parabola with equation y^2 = 4x, the distance from its focus to its directrix is 2. -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y^2 = 4*x → (∃ (p : ℝ), p = 2 ∧ p = 2*x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l377_37736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_six_terms_l377_37745

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h1 : ∀ n, a n > 0
  h2 : q > 1
  h3 : a 3 + a 5 = 20
  h4 : a 2 * a 6 = 64
  h5 : ∀ n, a (n + 1) = a n * q

/-- Sum of first n terms of a geometric sequence -/
noncomputable def sumGeometric (g : GeometricSequence) (n : ℕ) : ℝ :=
  (g.a 1) * (1 - g.q^n) / (1 - g.q)

/-- Theorem: The sum of the first 6 terms of the specific geometric sequence is 63 -/
theorem sum_of_six_terms (g : GeometricSequence) : sumGeometric g 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_six_terms_l377_37745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_theorem_l377_37712

theorem reciprocal_sum_theorem (m n p : ℕ+) : 
  Nat.gcd m.val n.val = 26 →
  Nat.gcd (Nat.gcd m.val n.val) p.val = 26 →
  Nat.lcm m.val n.val = 6930 →
  Nat.lcm (Nat.lcm m.val n.val) p.val = 6930 →
  m.val + n.val + p.val = 150 →
  (1 : ℚ) / m.val + (1 : ℚ) / n.val + (1 : ℚ) / p.val = (1 : ℚ) / 320166 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_theorem_l377_37712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_odd_sum_probability_l377_37723

def SpinnerA : Finset ℕ := {1, 4, 6}
def SpinnerB : Finset ℕ := {1, 3, 5, 7}

def is_odd (n : ℕ) : Bool := n % 2 = 1

def total_outcomes : ℕ := SpinnerA.card * SpinnerB.card

def odd_sum_outcomes : ℕ := Finset.filter (fun p => is_odd (p.1 + p.2)) (SpinnerA.product SpinnerB) |>.card

theorem spinner_odd_sum_probability :
  (odd_sum_outcomes : ℚ) / total_outcomes = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_odd_sum_probability_l377_37723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_points_exist_l377_37779

/-- Represents a football team in the tournament -/
structure Team where
  id : Nat
  wins : Nat
  draws : Nat
  losses : Nat

/-- The tournament setup and conditions -/
structure Tournament where
  teams : Finset Team
  team_count : teams.card = 16
  match_count : ∀ t ∈ teams, t.wins + t.draws + t.losses = 15
  win_loss_condition : ∀ t ∈ teams, t.wins ≥ 5 ∧ t.losses ≥ 5

/-- Calculate the points for a team -/
def points (t : Team) : Nat :=
  3 * t.wins + t.draws

/-- Theorem stating that there are at least two teams with the same points -/
theorem same_points_exist (tournament : Tournament) :
  ∃ t1 t2 : Team, t1 ∈ tournament.teams ∧ t2 ∈ tournament.teams ∧ t1 ≠ t2 ∧ points t1 = points t2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_points_exist_l377_37779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_2_pow_2012_l377_37787

def units_digit (n : ℕ) : ℕ := n % 10

def power_of_two_units_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | _ => 8

theorem units_digit_of_2_pow_2012 :
  units_digit (2^2012) = power_of_two_units_digit 2012 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_2_pow_2012_l377_37787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l377_37771

/-- The area of a triangle with given inradius, circumradius, and angle condition -/
theorem triangle_area (X Y Z : ℝ) (r R : ℝ) (h1 : r = 7) (h2 : R = 25)
  (h3 : 2 * Real.cos Y = Real.cos X + Real.cos Z) : 
  ∃ A : ℝ, A = 133 ∧ A = r * (X + Y + Z) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l377_37771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_iff_geometric_progression_l377_37740

theorem rational_iff_geometric_progression (x : ℚ) : 
  ∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (x + ↑b)^2 = (x + ↑a) * (x + ↑c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_iff_geometric_progression_l377_37740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l377_37768

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3) + Real.sin (2 * x - Real.pi / 3)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l377_37768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_as_many_heads_fair_coin_l377_37750

/-- The number of coin flips for the first person -/
def n₁ : ℕ := 5

/-- The number of coin flips for the second person -/
def n₂ : ℕ := 6

/-- The probability of getting heads on a single flip of a fair coin -/
def p : ℚ := 1/2

/-- The probability of getting at least as many heads in n₁ fair coin flips as in n₂ fair coin flips -/
noncomputable def prob_at_least_as_many_heads (n₁ n₂ : ℕ) (p : ℚ) : ℚ :=
  sorry

/-- Theorem stating that the probability of getting at least as many heads in 5 fair coin flips
    as in 6 fair coin flips is 1/2 -/
theorem prob_at_least_as_many_heads_fair_coin :
  prob_at_least_as_many_heads n₁ n₂ p = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_as_many_heads_fair_coin_l377_37750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangement_two_rows_photo_arrangement_two_rows_with_conditions_photo_arrangement_single_row_with_group_photo_arrangement_single_row_with_separation_l377_37707

-- Common definitions
def n : ℕ := 7  -- Total number of students

-- Part 1
theorem photo_arrangement_two_rows (front_row back_row : ℕ) 
  (h1 : front_row = 3) (h2 : back_row = 4) (h3 : front_row + back_row = n) :
  (n.factorial) = 5040 := by sorry

-- Part 2
theorem photo_arrangement_two_rows_with_conditions 
  (front_row back_row : ℕ) (a b : Fin n)
  (h1 : front_row = 3) (h2 : back_row = 4) (h3 : front_row + back_row = n) :
  (front_row * back_row * (n - 2).factorial) = 1440 := by sorry

-- Part 3
theorem photo_arrangement_single_row_with_group (a b c : Fin n) 
  (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (5 * 3).factorial = 720 := by sorry

-- Part 4
theorem photo_arrangement_single_row_with_separation (boys girls : ℕ)
  (h1 : boys = 4) (h2 : girls = 3) (h3 : boys + girls = n) :
  (boys.factorial * (boys + 1).descFactorial girls) = 1440 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangement_two_rows_photo_arrangement_two_rows_with_conditions_photo_arrangement_single_row_with_group_photo_arrangement_single_row_with_separation_l377_37707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_strategy_l377_37754

/-- Represents the player who has the winning strategy -/
inductive Winner
  | Sasho
  | Deni

/-- The game described in the problem -/
noncomputable def game (a : ℝ) : Winner :=
  if a > 1 ∨ (0 < a ∧ a < 1) then Winner.Sasho else Winner.Deni

/-- Theorem stating the winning conditions for Sasho and Deni -/
theorem winning_strategy (a : ℝ) (h : a ≠ 0 ∧ a ≠ 1) :
  (game a = Winner.Sasho ↔ a > 1 ∨ (0 < a ∧ a < 1)) ∧
  (game a = Winner.Deni ↔ a < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_strategy_l377_37754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_is_7_5_l377_37780

noncomputable section

/-- The number of degrees in a full circle (clock face) -/
def full_circle : ℝ := 360

/-- The number of degrees each hour represents -/
def degrees_per_hour : ℝ := full_circle / 12

/-- The number of degrees each minute represents -/
def degrees_per_minute : ℝ := full_circle / 60

/-- The position of the hour hand at 3:15 -/
def hour_hand_position : ℝ := 3 * degrees_per_hour + 15 * (degrees_per_hour / 60)

/-- The position of the minute hand at 3:15 -/
def minute_hand_position : ℝ := 15 * degrees_per_minute

/-- The smaller angle between the hour and minute hands at 3:15 -/
def clock_angle_at_3_15 : ℝ := min (abs (hour_hand_position - minute_hand_position)) (full_circle - abs (hour_hand_position - minute_hand_position))

theorem clock_angle_at_3_15_is_7_5 : clock_angle_at_3_15 = 7.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_is_7_5_l377_37780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_at_zero_l377_37708

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3)^x - x + 4

-- State the theorem
theorem f_decreasing_at_zero (x₀ x₁ x₂ : ℝ) : 
  f x₀ = 0 → 
  2 < x₁ → x₁ < x₀ → 
  x₀ < x₂ → 
  f x₁ > f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_at_zero_l377_37708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_makes_tiling_impossible_l377_37709

/-- Represents a tile type -/
inductive Tile
  | TwoByTwo
  | OneByFour
deriving Repr, DecidableEq

/-- Represents a rectangular grid -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a tiling of a grid -/
def Tiling := List Tile

/-- Checks if a tiling is valid for a given grid -/
def is_valid_tiling (g : Grid) (t : Tiling) : Prop :=
  sorry

/-- Theorem: Replacing a 2x2 tile with a 1x4 tile makes tiling impossible -/
theorem replacement_makes_tiling_impossible (g : Grid) (t : Tiling) :
  is_valid_tiling g t →
  ∃ (t' : Tiling), (t' = t.map (λ tile => if tile = Tile.TwoByTwo then Tile.OneByFour else tile)) →
  ¬ is_valid_tiling g t' := by
  sorry

#check replacement_makes_tiling_impossible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_makes_tiling_impossible_l377_37709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l377_37702

theorem triangle_properties (a b c A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  Real.sqrt 3 * c * Real.sin A = a * Real.cos C →
  c = Real.sqrt 7 * a →
  b = 2 * Real.sqrt 3 →
  C = π / 6 ∧ (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l377_37702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l377_37756

theorem trigonometric_equation_solution (x : ℝ) 
  (h1 : Real.cos (2 * x) ≠ 0) 
  (h2 : Real.sin x ≠ 0) : 
  (1 / Real.tan x + Real.tan (2 * x) + 1 = 4 * (Real.cos x)^2 + Real.sin (3 * x) / Real.sin x - 2 * Real.cos (2 * x)) ↔ 
  (∃ k : ℤ, x = Real.pi / 2 * (2 * k + 1)) ∨ 
  (∃ n : ℤ, x = Real.pi / 8 * (4 * n + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l377_37756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meiosis_and_fertilization_maintain_chromosome_count_l377_37773

/-- Represents a physiological activity in an organism -/
inductive PhysiologicalActivity
| Meiosis
| Fertilization
| Mitosis
| CellProliferation
| CellDifferentiation

/-- Represents an organism -/
structure Organism where
  isSexuallyReproducing : Bool

/-- Represents a cell in an organism -/
structure Cell where
  isReproductive : Bool
  chromosomeCount : ℕ

/-- Represents a generation of organisms -/
structure Generation where
  somaticCells : List Cell

/-- Function to check if a physiological activity maintains constant chromosome count -/
def maintainsConstantChromosomeCount (activity : PhysiologicalActivity) : Bool :=
  sorry

/-- Theorem stating that meiosis and fertilization maintain constant chromosome count -/
theorem meiosis_and_fertilization_maintain_chromosome_count 
  (organism : Organism) 
  (generations : List Generation) : 
  organism.isSexuallyReproducing → 
  (∀ (g₁ g₂ : Generation), g₁ ∈ generations → g₂ ∈ generations → 
    ∀ (c₁ c₂ : Cell), c₁ ∈ g₁.somaticCells → c₂ ∈ g₂.somaticCells → 
    c₁.chromosomeCount = c₂.chromosomeCount) → 
  maintainsConstantChromosomeCount PhysiologicalActivity.Meiosis ∧ 
  maintainsConstantChromosomeCount PhysiologicalActivity.Fertilization :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meiosis_and_fertilization_maintain_chromosome_count_l377_37773
